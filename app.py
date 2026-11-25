"""Flask application entrypoint for the Isolation Forest Anomaly Explorer."""

from __future__ import annotations

import os
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from detector import (
    MAX_UPLOAD_BYTES,
    DatasetNotFoundError,
    ValidationError,
    build_dataset_metadata_response,
    dataset_store,
    list_example_datasets,
    load_example_dataset,
    parse_csv_bytes,
    run_isolation_forest,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_err: RequestEntityTooLarge):
    """Return a clean JSON error if the uploaded request exceeds 5 MB."""
    return jsonify({"error": "File exceeds the 5 MB size limit."}), 413


@app.errorhandler(404)
def handle_not_found(_err):
    """Return JSON errors for missing API endpoints."""
    if request.path.startswith("/api/"):
        return jsonify({"error": "Endpoint not found."}), 404
    return jsonify({"error": "Resource not found."}), 404


@app.get("/")
def index():
    """Serve the single-page anomaly detection workspace."""
    return render_template("index.html")


@app.get("/api/examples")
def get_examples():
    """List bundled example datasets available in the `data/` directory."""
    examples = list_example_datasets()
    return jsonify({"examples": examples})


@app.get("/api/examples/<string:name>")
def get_example_dataset(name: str):
    """Load one bundled example dataset into memory and return its schema + preview."""
    try:
        dataset_id, parsed, extra = load_example_dataset(name)
    except DatasetNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValidationError as exc:
        return jsonify({"error": exc.message}), exc.status_code

    return jsonify(build_dataset_metadata_response(dataset_id, parsed, extra=extra))


@app.post("/api/upload")
def upload_csv():
    """Validate an uploaded CSV (max 5 MB) and store it in memory for detection."""
    if "file" not in request.files:
        return jsonify({"error": "No file field found in request."}), 400

    uploaded = request.files["file"]
    filename = (uploaded.filename or "").strip()
    if not filename:
        return jsonify({"error": "No file selected."}), 400

    if not filename.lower().endswith(".csv"):
        return jsonify({"error": "Only .csv files are supported."}), 400

    raw_bytes = uploaded.read()
    try:
        parsed = parse_csv_bytes(raw_bytes, filename=filename)
    except ValidationError as exc:
        return jsonify({"error": exc.message}), exc.status_code

    dataset_id = dataset_store.put(parsed)
    return jsonify(build_dataset_metadata_response(dataset_id, parsed))


@app.post("/api/detect")
def detect_anomalies():
    """Run Isolation Forest on the active dataset using the requested parameters."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be valid JSON."}), 400

    dataset_id = payload.get("dataset_id")
    if not dataset_id or not isinstance(dataset_id, str):
        return jsonify({"error": "A valid 'dataset_id' is required."}), 400

    try:
        dataset_record = dataset_store.get(dataset_id)
        output = run_isolation_forest(dataset_record, payload)
    except DatasetNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValidationError as exc:
        return jsonify({"error": exc.message}), exc.status_code

    return jsonify(output)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
