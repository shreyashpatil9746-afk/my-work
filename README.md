# OutlierScope — Isolation Forest Anomaly Explorer

OutlierScope is an interactive web application that lets non-technical and technical users upload any CSV dataset, tune **Isolation Forest** parameters in real time, and inspect detected anomalies on an interactive scatter plot, score distribution histogram, and sortable results table.

## Features

- **Zero-setup CSV ingestion**: Drag-and-drop or upload any UTF-8 CSV file (up to 5 MB) with automatic validation, numeric column detection, and in-memory storage (no permanent file persistence).
- **Three bundled example datasets**: Explore realistic credit card fraud, network attack burst, and manufacturing sensor datasets with optional ground-truth precision and recall feedback.
- **Interactive parameter tuning**: Adjust **Contamination** (`0.001`–`0.5` or `auto`), **Number of Trees (`n_estimators`)** (`10`–`500`), **Model Features**, and **Missing Value Handling** (`median-impute` or `drop`) with 300 ms debounced live re-scoring.
- **Interactive Plotly visualizations**:
  - **2D Scatter Plot** with blue normal observations, larger red anomaly markers, rich hover tooltips showing all selected features and normalized anomaly scores, zoom/pan, and legend toggles.
  - **Anomaly Score Histogram** (`0.0` to `1.0`) showing the exact decision threshold line.
- **Top 20 Anomalies Table & Export**: Sortable table of the highest-scoring rows plus one-click **Download results CSV** containing every original column, `anomaly_score`, and `label`.
- **Built-in Interpretability Panel**: Plain-language guidance on reading scores, setting contamination, balancing tree count vs. speed, and reviewing flagged candidates responsibly.

## Screenshot

![OutlierScope Anomaly Explorer Workspace](docs/screenshot-placeholder.png)
*(Placeholder: launch the app locally and capture the main workspace showing the left control panel, summary cards, scatter plot, score histogram, and top anomalies table.)*

## Local Setup

Requirements: **Python 3.10+**

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the Flask server
python app.py
```

By default, `app.py` binds to `0.0.0.0` on port `5000` (or the `PORT` environment variable if set). Open `http://localhost:5000` in your browser.

To run with Gunicorn locally:

```bash
gunicorn --bind 0.0.0.0:5000 app:app
```

## Usage Walkthrough (Example Datasets)

1. **Credit Card Transactions (`credit_card_transactions.csv`)**:
   - Loads 1,000 card purchases (`amount`, `hour_of_day`, `merchant_risk_score`) with ~2% injected fraud-like outliers.
   - Notice how setting Contamination near `0.020` cleanly isolates late-night, high-value, high-risk transactions. Watch **Ground-Truth Accuracy** update precision and recall as you slide contamination up or down.
2. **Network Traffic Flows (`network_traffic.csv`)**:
   - Contains 800 network flow records (`packets_per_sec`, `avg_packet_size`, `connection_duration`) with both high-rate volumetric floods and slow-rate long-duration connections.
   - Switch **X-Axis** and **Y-Axis** (for example `packets_per_sec` vs. `connection_duration`) while keeping all 3 features selected in **Model Features** to inspect multi-dimensional anomalies from different 2D angles.
3. **Manufacturing Quality Sensors (`manufacturing_quality.csv`)**:
   - Contains 650 batch readings (`temperature`, `vibration`, `pressure`) with defective-batch outliers and 6 intentional missing sensor cells.
   - Toggle **Missing Values** between **Fill with column median** (retains all 650 rows) and **Drop rows with missing values** (analyzes 644 complete rows) and check the summary card note.

*(Optional)* To regenerate the bundled CSVs deterministically:

```bash
python data/generate_examples.py
```

## API Reference

### `GET /api/examples`
Lists bundled example datasets.

**Sample Response (`200 OK`)**:
```json
{
  "examples": [
    {
      "name": "credit_card_transactions.csv",
      "title": "Credit Card Transactions",
      "description": "1,000 card purchases...",
      "default_contamination": "0.02",
      "row_count": 1000,
      "columns": ["amount", "hour_of_day", "merchant_risk_score"]
    }
  ]
}
```

### `GET /api/examples/<name>`
Loads a bundled CSV into in-memory storage and returns its schema and preview.

### `POST /api/upload`
Accepts `multipart/form-data` with field `file` (`.csv`, max `5 MB`, minimum `10` rows, at least `1` numeric column).

**Sample Request**:
```bash
curl -X POST http://localhost:5000/api/upload \
  -F "file=@data/credit_card_transactions.csv"
```

**Sample Response (`200 OK`)**:
```json
{
  "dataset_id": "9f12c4e2c7a44d7e8a9c5b21a73b8d10",
  "filename": "credit_card_transactions.csv",
  "columns": ["amount", "hour_of_day", "merchant_risk_score"],
  "numeric_columns": ["amount", "hour_of_day", "merchant_risk_score"],
  "non_numeric_columns": [],
  "has_ground_truth": true,
  "row_count": 1000,
  "missing_counts": {"amount": 0, "hour_of_day": 0, "merchant_risk_score": 0},
  "preview": [{"amount": 42.15, "hour_of_day": 14.2, "merchant_risk_score": 0.18}],
  "warnings": []
}
```

### `POST /api/detect`
Runs `StandardScaler` + `IsolationForest` on the selected features.

**Sample Request**:
```json
{
  "dataset_id": "9f12c4e2c7a44d7e8a9c5b21a73b8d10",
  "features": ["amount", "hour_of_day", "merchant_risk_score"],
  "contamination": 0.02,
  "n_estimators": 100,
  "missing_strategy": "impute",
  "random_state": 42
}
```

**Sample Response (`200 OK`)**:
```json
{
  "summary": {
    "total_rows": 1000,
    "anomaly_count": 20,
    "anomaly_percentage": 2.0,
    "score_threshold": 0.5612,
    "features_used": ["amount", "hour_of_day", "merchant_risk_score"],
    "contamination": 0.02,
    "n_estimators": 100,
    "missing_handling": {
      "strategy": "median-impute",
      "missing_cells": 0,
      "rows_affected": 0,
      "message": "No missing values found in selected features."
    },
    "ground_truth_metrics": {
      "precision": 0.95,
      "recall": 0.95,
      "f1_score": 0.95,
      "true_positives": 19,
      "false_positives": 1,
      "false_negatives": 1,
      "true_negatives": 979,
      "actual_anomalies": 20
    }
  },
  "results": [
    {
      "index": 0,
      "values": {"amount": 42.15, "hour_of_day": 14.2, "merchant_risk_score": 0.18},
      "model_values": {"amount": 42.15, "hour_of_day": 14.2, "merchant_risk_score": 0.18},
      "anomaly_score": 0.1284,
      "label": "normal"
    }
  ]
}
```

## Parameter Tuning Guide

| Parameter | Range | Effect | Practical Tip |
| :--- | :--- | :--- | :--- |
| **Contamination** | `0.001` – `0.500`, or `auto` | Sets the expected proportion of outliers and determines the score cutoff. | Start with `0.01`–`0.05` if anomalies are rare, or use `auto` as an exploratory baseline. |
| **n_estimators** | `10` – `500` | Number of random isolation trees built in the ensemble. | `100`–`200` gives consistent scores; lower (`20`–`50`) speeds up large datasets. |
| **Features** | `>= 2` numeric columns | Dimensions standardized and evaluated by the forest. | Exclude IDs, timestamps-as-integers, or unrelated noise columns that dilute true outliers. |
| **Missing Values** | `impute` or `drop` | Median imputation fills gaps; drop removes incomplete rows. | Use `impute` to preserve rows when occasional sensors drop out; use `drop` if incomplete records are invalid. |

## Running Tests

```bash
pytest -v
```

## Deployment

### Primary: Render

1. Push this repository to GitHub or GitLab.
2. In the [Render Dashboard](https://dashboard.render.com/), click **New + -> Web Service** and connect your repository.
3. Configure the service:
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
4. Render automatically provides the `PORT` environment variable. Click **Create Web Service** and open your live URL once the build finishes.

*(Alternatively, choose **Docker** runtime on Render using the included `Dockerfile`.)*

### Alternative: Railway or Heroku

**Railway**:
1. Create a new project in [Railway](https://railway.app/) -> **Deploy from GitHub repo**.
2. Railway detects `Procfile` (`web: gunicorn app:app`) and `requirements.txt` automatically (or builds from `Dockerfile`).
3. Generate a public domain under **Settings -> Networking**.

**Heroku**:
```bash
heroku create your-app-name
git push heroku main
heroku open
```
`Procfile` and `runtime.txt` are already configured for Heroku's Python buildpack.

## Troubleshooting & Limitations

- **"Dataset session expired or not found"**: Uploaded datasets live strictly in server memory (up to 25 most recent datasets) and are cleared on server restart or across separate unpinned worker processes. If deploying with multiple Gunicorn workers (`--workers > 1`), use sticky sessions or `--workers 1 --threads 4` so requests hit the same in-memory store.
- **"Only 1 numeric column detected"**: Ensure numbers in your CSV are not formatted with currency symbols (`$1,200`) or percent signs (`12%`), which pandas reads as text.
- **Model assumptions**: Isolation Forest assumes anomalies are both *few* and *different* in feature space. It does not understand sequential time order unless time-derived features (e.g., `hour_of_day`, `rolling_avg`) are included as numeric columns, and flagged rows should always be verified by a domain reviewer.
