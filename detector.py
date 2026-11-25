"""Core data validation, storage, and Isolation Forest anomaly detection logic.

All business logic is intentionally separated from Flask route handlers in
`app.py` so it can be unit-tested cleanly and maintained independently.
"""

from __future__ import annotations

from collections import OrderedDict
import io
import math
from pathlib import Path
import threading
import uuid
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
MIN_ROW_COUNT = 10
MAX_STORED_DATASETS = 25

GROUND_TRUTH_ALIASES = frozenset({"_ground_truth", "ground_truth", "is_anomaly"})

EXAMPLE_CATALOG: dict[str, dict[str, str]] = {
    "credit_card_transactions.csv": {
        "title": "Credit Card Transactions",
        "description": (
            "1,000 card purchases with amount, hour of day, and merchant risk "
            "score (~2% injected fraud-like outliers)."
        ),
        "default_contamination": "0.02",
    },
    "network_traffic.csv": {
        "title": "Network Traffic Flows",
        "description": (
            "800 network flow records with packet rate, average packet size, "
            "and duration (~3% injected attack bursts)."
        ),
        "default_contamination": "0.03",
    },
    "manufacturing_quality.csv": {
        "title": "Manufacturing Quality Sensors",
        "description": (
            "650 production batch readings with temperature, vibration, and "
            "pressure (~2.5% defective-batch outliers and 6 missing readings)."
        ),
        "default_contamination": "0.025",
    },
}


class ValidationError(ValueError):
    """Raised when uploaded data or detection parameters fail validation."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class DatasetNotFoundError(KeyError):
    """Raised when a requested dataset_id is missing or expired from memory."""


class InMemoryDatasetStore:
    """Thread-safe bounded LRU cache keeping uploaded datasets in memory only."""

    def __init__(self, capacity: int = MAX_STORED_DATASETS) -> None:
        self._capacity = capacity
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, dataset_record: dict[str, Any]) -> str:
        """Store a parsed dataset under a new UUID and return its ID."""
        dataset_id = uuid.uuid4().hex
        with self._lock:
            self._store[dataset_id] = dataset_record
            self._store.move_to_end(dataset_id)
            while len(self._store) > self._capacity:
                self._store.popitem(last=False)
        return dataset_id

    def get(self, dataset_id: str) -> dict[str, Any]:
        """Retrieve a stored dataset by ID or raise DatasetNotFoundError."""
        if not dataset_id or not isinstance(dataset_id, str):
            raise DatasetNotFoundError("Missing dataset_id.")
        with self._lock:
            record = self._store.get(dataset_id)
            if record is None:
                raise DatasetNotFoundError(
                    "Dataset session expired or not found. Please upload or select a dataset again."
                )
            self._store.move_to_end(dataset_id)
            return record


dataset_store = InMemoryDatasetStore()


def _clean_json_value(value: Any) -> Any:
    """Convert numpy/pandas scalars and NaN/Inf values to JSON-safe Python primitives."""
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f_val = float(value)
        if math.isnan(f_val) or math.isinf(f_val):
            return None
        return round(f_val, 6)
    if pd.isna(value):
        return None
    return str(value)


def _df_to_safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize DataFrame rows to dicts with NaN converted to None."""
    records: list[dict[str, Any]] = []
    columns = list(df.columns)
    for row in df.itertuples(index=False, name=None):
        records.append({col: _clean_json_value(val) for col, val in zip(columns, row)})
    return records


def _extract_ground_truth(df: pd.DataFrame) -> tuple[str | None, pd.Series | None]:
    """Locate an optional binary ground-truth column if present in the CSV."""
    for col in df.columns:
        if col.strip().lower() in GROUND_TRUTH_ALIASES:
            series = df[col]
            # Accept numeric 0/1 or string normal/anomaly
            if pd.api.types.is_numeric_dtype(series):
                valid = series.dropna()
                if len(valid) > 0 and set(valid.unique()).issubset({0, 1, 0.0, 1.0}):
                    return col, series.fillna(0).astype(int)
            else:
                normalized = series.astype(str).str.strip().str.lower()
                mapping = {"1": 1, "true": 1, "anomaly": 1, "0": 0, "false": 0, "normal": 0}
                if set(normalized.unique()).issubset(mapping.keys()):
                    return col, normalized.map(mapping).astype(int)
    return None, None


def parse_csv_bytes(raw_bytes: bytes, filename: str) -> dict[str, Any]:
    """Validate uploaded CSV bytes and return structured metadata + parsed DataFrame."""
    if not raw_bytes or len(raw_bytes.strip()) == 0:
        raise ValidationError("The uploaded file is empty.")

    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise ValidationError(
            "File exceeds the 5 MB size limit.", status_code=413
        )

    if b"\x00" in raw_bytes:
        raise ValidationError("Invalid CSV file: binary content detected.")

    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValidationError(
            "File must be a valid UTF-8 encoded CSV."
        ) from exc

    try:
        df = pd.read_csv(io.StringIO(text))
    except pd.errors.EmptyDataError as exc:
        raise ValidationError("The CSV file contains no rows or headers.") from exc
    except (pd.errors.ParserError, ValueError) as exc:
        raise ValidationError(f"Unable to parse CSV file: {exc}") from exc

    # Clean column names and drop completely blank unnamed columns
    df.columns = [str(col).strip() for col in df.columns]
    unnamed_empty = [
        col for col in df.columns
        if col.startswith("Unnamed:") and df[col].isna().all()
    ]
    if unnamed_empty:
        df = df.drop(columns=unnamed_empty)

    if df.empty or len(df.columns) == 0:
        raise ValidationError("The CSV file contains no usable columns or rows.")

    if len(df) < MIN_ROW_COUNT:
        raise ValidationError(
            f"Dataset must contain at least {MIN_ROW_COUNT} rows (found {len(df)})."
        )

    gt_col, gt_series = _extract_ground_truth(df)
    visible_columns = [col for col in df.columns if col != gt_col]

    numeric_columns: list[str] = []
    non_numeric_columns: list[str] = []

    for col in visible_columns:
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            if not df[col].dropna().empty:
                numeric_columns.append(col)
                continue
        non_numeric_columns.append(col)

    if not numeric_columns:
        raise ValidationError(
            "No numeric columns found. Upload a CSV with at least 2 numeric columns."
        )

    missing_counts = {
        col: int(df[col].isna().sum())
        for col in numeric_columns
    }

    preview_df = df[visible_columns].head(5)

    return {
        "filename": filename,
        "df": df[visible_columns].copy(),
        "ground_truth": gt_series,
        "columns": visible_columns,
        "numeric_columns": numeric_columns,
        "non_numeric_columns": non_numeric_columns,
        "has_ground_truth": gt_series is not None,
        "row_count": int(len(df)),
        "missing_counts": missing_counts,
        "preview": _df_to_safe_records(preview_df),
    }


def build_dataset_metadata_response(
    dataset_id: str,
    record: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the JSON-serializable metadata payload for upload/example responses."""
    warnings: list[str] = []
    if len(record["numeric_columns"]) < 2:
        warnings.append(
            "Only 1 numeric column was detected. Isolation Forest requires at least 2 numeric columns."
        )

    payload: dict[str, Any] = {
        "dataset_id": dataset_id,
        "filename": record["filename"],
        "columns": record["columns"],
        "numeric_columns": record["numeric_columns"],
        "non_numeric_columns": record["non_numeric_columns"],
        "has_ground_truth": record["has_ground_truth"],
        "row_count": record["row_count"],
        "missing_counts": record["missing_counts"],
        "preview": record["preview"],
        "warnings": warnings,
    }
    if extra:
        payload.update(extra)
    return payload


def list_example_datasets() -> list[dict[str, Any]]:
    """Return metadata for all bundled example CSVs present in `data/`."""
    examples: list[dict[str, Any]] = []
    for filename, meta in EXAMPLE_CATALOG.items():
        path = (DATA_DIR / filename).resolve()
        if path.parent != DATA_DIR.resolve() or not path.is_file():
            continue
        df = pd.read_csv(path)
        visible_cols = [c for c in df.columns if c.strip().lower() not in GROUND_TRUTH_ALIASES]
        examples.append({
            "name": filename,
            "title": meta["title"],
            "description": meta["description"],
            "default_contamination": meta["default_contamination"],
            "row_count": int(len(df)),
            "columns": visible_cols,
        })
    return examples


def load_example_dataset(name: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Safely load a bundled example CSV by filename into the in-memory store."""
    if not name or name not in EXAMPLE_CATALOG:
        raise DatasetNotFoundError(f"Unknown example dataset: '{name}'.")

    target_path = (DATA_DIR / name).resolve()
    if target_path.parent != DATA_DIR.resolve() or not target_path.is_file():
        raise DatasetNotFoundError(f"Example dataset file not found: '{name}'.")

    raw_bytes = target_path.read_bytes()
    parsed = parse_csv_bytes(raw_bytes, filename=name)
    dataset_id = dataset_store.put(parsed)
    meta = EXAMPLE_CATALOG[name]
    return dataset_id, parsed, {
        "title": meta["title"],
        "description": meta["description"],
        "default_contamination": meta["default_contamination"],
    }


def _validate_detect_params(
    payload: dict[str, Any],
    numeric_columns: list[str],
) -> dict[str, Any]:
    """Validate and normalize detection request parameters."""
    if not isinstance(payload, dict):
        raise ValidationError("Request body must be a JSON object.")

    features = payload.get("features")
    if not isinstance(features, list) or len(features) < 2:
        raise ValidationError("Select at least 2 numeric feature columns.")

    # Deduplicate while preserving order
    seen: set[str] = set()
    clean_features: list[str] = []
    valid_numeric = set(numeric_columns)
    for feat in features:
        if not isinstance(feat, str) or feat not in valid_numeric:
            raise ValidationError(
                f"Invalid or non-numeric feature column: '{feat}'."
            )
        if feat not in seen:
            seen.add(feat)
            clean_features.append(feat)

    if len(clean_features) < 2:
        raise ValidationError("Select at least 2 distinct numeric feature columns.")

    raw_contamination = payload.get("contamination", "auto")
    if isinstance(raw_contamination, str) and raw_contamination.strip().lower() == "auto":
        contamination: str | float = "auto"
    else:
        if isinstance(raw_contamination, bool):
            raise ValidationError("Contamination must be 'auto' or a number between 0.001 and 0.5.")
        try:
            cont_val = float(raw_contamination)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "Contamination must be 'auto' or a number between 0.001 and 0.5."
            ) from exc
        if math.isnan(cont_val) or math.isinf(cont_val) or not (0.001 <= cont_val <= 0.5):
            raise ValidationError(
                "Contamination must be between 0.001 and 0.5 (or 'auto')."
            )
        contamination = cont_val

    raw_estimators = payload.get("n_estimators", 100)
    if isinstance(raw_estimators, bool) or isinstance(raw_estimators, float):
        if isinstance(raw_estimators, float) and not raw_estimators.is_integer():
            raise ValidationError("n_estimators must be an integer between 10 and 500.")
        if isinstance(raw_estimators, bool):
            raise ValidationError("n_estimators must be an integer between 10 and 500.")
    try:
        n_estimators = int(raw_estimators)
    except (TypeError, ValueError) as exc:
        raise ValidationError("n_estimators must be an integer between 10 and 500.") from exc
    if not (10 <= n_estimators <= 500):
        raise ValidationError("n_estimators must be between 10 and 500.")

    raw_seed = payload.get("random_state", 42)
    if raw_seed is None:
        random_state: int | None = None
    else:
        if isinstance(raw_seed, bool):
            raise ValidationError("random_state must be an integer or null.")
        try:
            random_state = int(raw_seed)
        except (TypeError, ValueError) as exc:
            raise ValidationError("random_state must be an integer or null.") from exc

    missing_strategy = str(payload.get("missing_strategy", "impute")).strip().lower()
    if missing_strategy not in {"impute", "drop"}:
        raise ValidationError("missing_strategy must be 'impute' or 'drop'.")

    return {
        "features": clean_features,
        "contamination": contamination,
        "n_estimators": n_estimators,
        "random_state": random_state,
        "missing_strategy": missing_strategy,
    }


def run_isolation_forest(
    dataset_record: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Preprocess selected features, fit IsolationForest, and return scored rows & summary."""
    params = _validate_detect_params(payload, dataset_record["numeric_columns"])
    features: list[str] = params["features"]
    missing_strategy: str = params["missing_strategy"]

    full_df: pd.DataFrame = dataset_record["df"].copy()
    gt_series: pd.Series | None = dataset_record.get("ground_truth")

    # Normalize +/-inf to NaN in selected model features
    feature_frame = full_df[features].replace([np.inf, -np.inf], np.nan)
    for feat in features:
        if feature_frame[feat].dropna().empty:
            raise ValidationError(
                f"Selected feature '{feat}' has no valid numeric values."
            )

    missing_cells = int(feature_frame.isna().sum().sum())
    rows_with_missing = int(feature_frame.isna().any(axis=1).sum())

    if missing_strategy == "drop":
        keep_mask = ~feature_frame.isna().any(axis=1)
        rows_dropped = int((~keep_mask).sum())
        kept_indices = full_df.index[keep_mask]
        feature_matrix = feature_frame.loc[keep_mask].to_numpy(dtype=float)
        if len(feature_matrix) < MIN_ROW_COUNT:
            raise ValidationError(
                f"Fewer than {MIN_ROW_COUNT} complete rows remain after dropping missing values."
            )
        missing_info = {
            "strategy": "drop",
            "missing_cells": missing_cells,
            "rows_affected": rows_dropped,
            "message": (
                f"Dropped {rows_dropped} row(s) with missing values ({missing_cells} cell(s))."
                if rows_dropped > 0
                else "No missing values found in selected features."
            ),
        }
        plot_df = full_df.loc[keep_mask]
        gt_eval = gt_series.loc[keep_mask] if gt_series is not None else None
    else:
        kept_indices = full_df.index
        if missing_cells > 0:
            imputer = SimpleImputer(strategy="median")
            feature_matrix = imputer.fit_transform(feature_frame.to_numpy(dtype=float))
            col_medians = {
                feat: _clean_json_value(val)
                for feat, val in zip(features, imputer.statistics_)
            }
            missing_info = {
                "strategy": "median-impute",
                "missing_cells": missing_cells,
                "rows_affected": rows_with_missing,
                "imputed_medians": col_medians,
                "message": (
                    f"Median-imputed {missing_cells} missing cell(s) across "
                    f"{rows_with_missing} row(s)."
                ),
            }
        else:
            feature_matrix = feature_frame.to_numpy(dtype=float)
            missing_info = {
                "strategy": "median-impute",
                "missing_cells": 0,
                "rows_affected": 0,
                "imputed_medians": {},
                "message": "No missing values found in selected features.",
            }
        plot_df = full_df
        gt_eval = gt_series

    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(feature_matrix)

    model = IsolationForest(
        n_estimators=params["n_estimators"],
        contamination=params["contamination"],
        random_state=params["random_state"],
    )
    raw_preds = model.fit_predict(scaled_matrix)
    raw_scores = model.score_samples(scaled_matrix)

    # Negate score_samples so higher score = more anomalous, then min-max scale to [0, 1]
    neg_scores = -raw_scores
    s_min = float(neg_scores.min())
    s_max = float(neg_scores.max())
    raw_threshold = float(-model.offset_)

    if s_max > s_min:
        norm_scores = (neg_scores - s_min) / (s_max - s_min)
        score_threshold = float(
            np.clip((raw_threshold - s_min) / (s_max - s_min), 0.0, 1.0)
        )
    else:
        norm_scores = np.zeros_like(neg_scores, dtype=float)
        score_threshold = 0.5

    labels = ["anomaly" if int(pred) == -1 else "normal" for pred in raw_preds]
    anomaly_count = sum(1 for lbl in labels if lbl == "anomaly")
    total_rows = len(labels)
    anomaly_pct = round((anomaly_count / total_rows) * 100.0, 2) if total_rows else 0.0

    # Prepare clean per-row records containing original columns + model coordinates fallback
    row_records = _df_to_safe_records(plot_df)
    results: list[dict[str, Any]] = []
    for pos, (orig_idx, row_vals, score, label) in enumerate(
        zip(kept_indices, row_records, norm_scores, labels)
    ):
        # Provide imputed feature coordinates for plotting if original had NaN
        imputed_vals = {
            feat: _clean_json_value(feature_matrix[pos, f_idx])
            for f_idx, feat in enumerate(features)
        }
        results.append({
            "index": int(orig_idx),
            "values": row_vals,
            "model_values": imputed_vals,
            "anomaly_score": round(float(score), 4),
            "label": label,
        })

    ground_truth_metrics: dict[str, Any] | None = None
    if gt_eval is not None:
        y_true = gt_eval.to_numpy(dtype=int)
        y_pred = np.array([1 if lbl == "anomaly" else 0 for lbl in labels], dtype=int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        ground_truth_metrics = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "actual_anomalies": int(np.sum(y_true == 1)),
        }

    return {
        "summary": {
            "total_rows": total_rows,
            "anomaly_count": anomaly_count,
            "anomaly_percentage": anomaly_pct,
            "score_threshold": round(score_threshold, 4),
            "features_used": features,
            "contamination": params["contamination"],
            "n_estimators": params["n_estimators"],
            "missing_handling": missing_info,
            "ground_truth_metrics": ground_truth_metrics,
        },
        "results": results,
    }
