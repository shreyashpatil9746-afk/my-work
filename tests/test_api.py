"""Pytest suite covering upload validation, examples, detection output, and parameter bounds."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app  # noqa: E402
from detector import MAX_UPLOAD_BYTES  # noqa: E402


@pytest.fixture()
def client():
    """Return a Flask test client with testing mode enabled."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def _make_csv_bytes(rows: int = 20, include_gt: bool = False, with_missing: bool = False) -> bytes:
    """Build deterministic valid CSV bytes for testing."""
    header = "f1,f2,f3,label_text"
    if include_gt:
        header += ",_ground_truth"
    lines = [header]
    for idx in range(rows):
        v1 = 10.0 + idx * 0.5
        v2 = 50.0 + (idx % 5) * 2.0
        v3 = 1.5 if idx < rows - 1 else 25.0
        if with_missing and idx == 2:
            v1_str = ""
        else:
            v1_str = f"{v1:.2f}"
        row = f"{v1_str},{v2:.2f},{v3:.2f},item_{idx}"
        if include_gt:
            row += ",1" if idx == rows - 1 else ",0"
        lines.append(row)
    return "\n".join(lines).encode("utf-8")


def _upload_sample(client, csv_bytes: bytes, filename: str = "sample.csv"):
    return client.post(
        "/api/upload",
        data={"file": (io.BytesIO(csv_bytes), filename)},
        content_type="multipart/form-data",
    )


def test_index_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"OutlierScope" in resp.data
    assert b"plotly" in resp.data.lower()


def test_examples_list_and_load(client):
    resp = client.get("/api/examples")
    assert resp.status_code == 200
    data = resp.get_json()
    names = {item["name"] for item in data["examples"]}
    assert {
        "credit_card_transactions.csv",
        "network_traffic.csv",
        "manufacturing_quality.csv",
    }.issubset(names)

    load_resp = client.get("/api/examples/credit_card_transactions.csv")
    assert load_resp.status_code == 200
    loaded = load_resp.get_json()
    assert loaded["row_count"] == 1000
    assert loaded["has_ground_truth"] is True
    assert set(loaded["numeric_columns"]) == {
        "amount",
        "hour_of_day",
        "merchant_risk_score",
    }
    assert "_ground_truth" not in loaded["numeric_columns"]

    missing_resp = client.get("/api/examples/../app.py")
    assert missing_resp.status_code == 404


def test_upload_validation_errors(client):
    # No file part
    resp_missing = client.post("/api/upload")
    assert resp_missing.status_code == 400
    assert "error" in resp_missing.get_json()

    # Non-CSV extension
    resp_ext = _upload_sample(client, b"a,b\n1,2\n", filename="data.txt")
    assert resp_ext.status_code == 400
    assert "csv" in resp_ext.get_json()["error"].lower()

    # Empty file
    resp_empty = _upload_sample(client, b"   \n", filename="empty.csv")
    assert resp_empty.status_code == 400
    assert "empty" in resp_empty.get_json()["error"].lower()

    # Binary content disguised as CSV
    resp_bin = _upload_sample(client, b"\x00\x01\x02\xff\xfe", filename="bad.csv")
    assert resp_bin.status_code == 400

    # No numeric columns
    text_only = "name,city\n" + "\n".join(f"user_{i},city_{i}" for i in range(12))
    resp_no_num = _upload_sample(client, text_only.encode("utf-8"), filename="text.csv")
    assert resp_no_num.status_code == 400
    assert "numeric" in resp_no_num.get_json()["error"].lower()

    # Fewer than 10 rows
    short_csv = _make_csv_bytes(rows=9)
    resp_short = _upload_sample(client, short_csv, filename="short.csv")
    assert resp_short.status_code == 400
    assert "10 rows" in resp_short.get_json()["error"]

    # Oversized file (> 5 MB)
    oversized = b"a,b\n" + (b"1.0,2.0\n" * ((MAX_UPLOAD_BYTES // 8) + 500))
    resp_big = _upload_sample(client, oversized, filename="huge.csv")
    assert resp_big.status_code == 413
    assert "5 MB" in resp_big.get_json()["error"]


def test_upload_valid_csv_returns_expected_schema(client):
    resp = _upload_sample(client, _make_csv_bytes(rows=15), filename="valid.csv")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["dataset_id"]
    assert data["filename"] == "valid.csv"
    assert data["row_count"] == 15
    assert data["numeric_columns"] == ["f1", "f2", "f3"]
    assert data["non_numeric_columns"] == ["label_text"]
    assert len(data["preview"]) == 5


def test_detect_output_shape_and_ground_truth(client):
    upload_resp = _upload_sample(
        client, _make_csv_bytes(rows=30, include_gt=True, with_missing=True)
    )
    dataset_id = upload_resp.get_json()["dataset_id"]

    # Default median-impute strategy
    resp = client.post(
        "/api/detect",
        json={
            "dataset_id": dataset_id,
            "features": ["f1", "f2", "f3"],
            "contamination": 0.05,
            "n_estimators": 50,
            "random_state": 42,
        },
    )
    assert resp.status_code == 200

    # Ensure strict valid JSON (no NaN/Infinity literals)
    def reject_nan(const):
        raise ValueError(f"Invalid JSON constant: {const}")

    data = json.loads(resp.data.decode("utf-8"), parse_constant=reject_nan)
    summary = data["summary"]
    results = data["results"]

    assert summary["total_rows"] == 30
    assert summary["anomaly_count"] >= 1
    assert 0.0 <= summary["anomaly_percentage"] <= 100.0
    assert 0.0 <= summary["score_threshold"] <= 1.0
    assert summary["missing_handling"]["strategy"] == "median-impute"
    assert summary["missing_handling"]["missing_cells"] == 1
    assert summary["ground_truth_metrics"] is not None
    assert 0.0 <= summary["ground_truth_metrics"]["precision"] <= 1.0
    assert 0.0 <= summary["ground_truth_metrics"]["recall"] <= 1.0

    assert len(results) == 30
    for row in results:
        assert isinstance(row["index"], int)
        assert isinstance(row["values"], dict)
        assert 0.0 <= row["anomaly_score"] <= 1.0
        assert row["label"] in {"anomaly", "normal"}

    # Drop missing rows strategy
    drop_resp = client.post(
        "/api/detect",
        json={
            "dataset_id": dataset_id,
            "features": ["f1", "f2"],
            "contamination": "auto",
            "n_estimators": 100,
            "missing_strategy": "drop",
        },
    )
    assert drop_resp.status_code == 200
    drop_data = drop_resp.get_json()
    assert drop_data["summary"]["total_rows"] == 29
    assert drop_data["summary"]["missing_handling"]["strategy"] == "drop"
    assert drop_data["summary"]["missing_handling"]["rows_affected"] == 1


def test_detect_parameter_bounds(client):
    upload_resp = _upload_sample(client, _make_csv_bytes(rows=20))
    dataset_id = upload_resp.get_json()["dataset_id"]

    # Unknown dataset_id
    resp_missing_id = client.post(
        "/api/detect",
        json={"dataset_id": "nonexistent-id", "features": ["f1", "f2"]},
    )
    assert resp_missing_id.status_code == 404

    # Fewer than 2 features
    resp_one_feat = client.post(
        "/api/detect",
        json={"dataset_id": dataset_id, "features": ["f1"]},
    )
    assert resp_one_feat.status_code == 400

    # Non-numeric feature rejected
    resp_bad_feat = client.post(
        "/api/detect",
        json={"dataset_id": dataset_id, "features": ["f1", "label_text"]},
    )
    assert resp_bad_feat.status_code == 400

    # Contamination out of bounds
    for bad_cont in [0.0, 0.0005, 0.51, -0.1, "invalid", True]:
        resp = client.post(
            "/api/detect",
            json={
                "dataset_id": dataset_id,
                "features": ["f1", "f2"],
                "contamination": bad_cont,
            },
        )
        assert resp.status_code == 400

    # Contamination valid boundaries
    for good_cont in [0.001, 0.5, "auto"]:
        resp = client.post(
            "/api/detect",
            json={
                "dataset_id": dataset_id,
                "features": ["f1", "f2"],
                "contamination": good_cont,
            },
        )
        assert resp.status_code == 200

    # n_estimators out of bounds
    for bad_est in [9, 501, 50.5, "100x", False]:
        resp = client.post(
            "/api/detect",
            json={
                "dataset_id": dataset_id,
                "features": ["f1", "f2"],
                "n_estimators": bad_est,
            },
        )
        assert resp.status_code == 400

    # n_estimators valid boundaries
    for good_est in [10, 500]:
        resp = client.post(
            "/api/detect",
            json={
                "dataset_id": dataset_id,
                "features": ["f1", "f2"],
                "n_estimators": good_est,
            },
        )
        assert resp.status_code == 200
