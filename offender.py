from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file

# Secondary detection model — HF Space with automatic local YOLO fallback.
# Import is guarded so the app starts even if ultralytics is not installed.
try:
    from hf_space_client import HFDetector
    _hf_detector: HFDetector | None = HFDetector()
except Exception as _hf_import_err:  # pragma: no cover
    logging.getLogger(__name__).warning(
        "HFDetector unavailable: %s — /detect endpoint will return 503.",
        _hf_import_err,
    )
    _hf_detector = None

# Advanced / optional — Vehicle Speed Estimation Space.
# Disabled automatically when HF_TOKEN is absent; never crashes the app.
try:
    from hf_space_client import SpeedEstimator
    _speed_estimator: SpeedEstimator | None = SpeedEstimator()
except Exception as _speed_import_err:  # pragma: no cover
    logging.getLogger(__name__).warning(
        "SpeedEstimator unavailable: %s — /speed_estimate endpoint will be disabled.",
        _speed_import_err,
    )
    _speed_estimator = None

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "offenders_data.json"
STATIC_DIR = BASE_DIR / "static"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

CATEGORY_META = {
    "triples": {"label": "Triple Riding", "default_offence": "Triple Riding"},
    "no_helmet": {"label": "No Helmet", "default_offence": "No Helmet"},
}
EDITABLE_FIELDS = {"name", "number_plate", "fine", "location", "fine_applied"}


def load_data() -> dict[str, dict[str, dict[str, Any]]]:
    if not DATA_FILE.exists():
        return {}

    try:
        with DATA_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def save_data(data: dict[str, dict[str, dict[str, Any]]]) -> None:
    with DATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def parse_session_time(folder_name: str) -> str:
    prefixes = ("TRIPLES_session_", "NO_HELMET_session_", "THRIPLES_session_", "session_")

    for prefix in prefixes:
        if folder_name.startswith(prefix):
            raw_timestamp = folder_name[len(prefix) : len(prefix) + 15]
            try:
                parsed = datetime.strptime(raw_timestamp, "%Y%m%d_%H%M%S")
                return parsed.strftime("%d %b %Y, %H:%M:%S")
            except ValueError:
                return "Unknown"

    return "Unknown"


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def default_record(category: str) -> dict[str, Any]:
    return {
        "name": "",
        "offence": CATEGORY_META[category]["default_offence"],
        "number_plate": "",
        "fine": 0,
        "location": "",
        "fine_applied": False,
    }


def get_offenders() -> dict[str, list[dict[str, Any]]]:
    data = load_data()
    updated = False
    offenders: dict[str, list[dict[str, Any]]] = {}

    for category in CATEGORY_META:
        category_dir = STATIC_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)
        offenders[category] = []
        data.setdefault(category, {})

        folders = sorted((path for path in category_dir.iterdir() if path.is_dir()), key=lambda path: path.name, reverse=True)

        for folder in folders:
            if folder.name not in data[category]:
                data[category][folder.name] = default_record(category)
                updated = True

            record = data[category][folder.name]
            images = sorted([file.name for file in folder.iterdir() if file.suffix.lower() in IMAGE_EXTENSIONS])

            offenders[category].append(
                {
                    "folder": folder.name,
                    "images": images,
                    "session_time": parse_session_time(folder.name),
                    "fine_applied": parse_bool(record.get("fine_applied")),
                    "name": record.get("name", ""),
                    "offence": record.get("offence", CATEGORY_META[category]["default_offence"]),
                    "number_plate": record.get("number_plate", ""),
                    "fine": int(record.get("fine", 0) or 0),
                    "location": record.get("location", ""),
                }
            )

    if updated:
        save_data(data)

    return offenders


def build_summary(offenders: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary = {
        "total_cases": 0,
        "fine_applied": 0,
        "fine_pending": 0,
        "total_fine_amount": 0,
        "categories": {},
    }

    for category, items in offenders.items():
        total_cases = len(items)
        fine_applied = sum(1 for offender in items if offender["fine_applied"])
        fine_pending = total_cases - fine_applied
        total_fine_amount = sum(int(offender.get("fine", 0) or 0) for offender in items)

        summary["categories"][category] = {
            "total_cases": total_cases,
            "fine_applied": fine_applied,
            "fine_pending": fine_pending,
            "total_fine_amount": total_fine_amount,
        }

        summary["total_cases"] += total_cases
        summary["fine_applied"] += fine_applied
        summary["fine_pending"] += fine_pending
        summary["total_fine_amount"] += total_fine_amount

    return summary


@app.route("/")
def index():
    offenders = get_offenders()
    summary = build_summary(offenders)
    return render_template("index.html", offenders=offenders, summary=summary, category_meta=CATEGORY_META)


@app.get("/healthz")
def health_check():
    return jsonify({"status": "ok"}), 200


@app.route("/update_offender", methods=["POST"])
def update_offender():
    payload = request.get_json(silent=True) or {}
    category = payload.get("category")
    folder = payload.get("folder")
    field = payload.get("field")
    value = payload.get("value")

    if not all([category, folder, field]):
        return jsonify({"status": "error", "message": "Missing data"}), 400

    if category not in CATEGORY_META:
        return jsonify({"status": "error", "message": "Invalid category"}), 400

    if field not in EDITABLE_FIELDS:
        return jsonify({"status": "error", "message": f"Field '{field}' is not editable"}), 400

    data = load_data()
    data.setdefault(category, {})
    data[category].setdefault(folder, default_record(category))

    if field == "fine":
        try:
            value = int(value)
        except (TypeError, ValueError):
            return jsonify({"status": "error", "message": "Fine must be a whole number"}), 400
        if value < 0:
            return jsonify({"status": "error", "message": "Fine cannot be negative"}), 400
    elif field == "fine_applied":
        value = parse_bool(value)
    else:
        value = str(value or "").strip()

    data[category][folder][field] = value
    save_data(data)

    updated_record = {
        **default_record(category),
        **data[category][folder],
        "fine_applied": parse_bool(data[category][folder].get("fine_applied")),
    }
    return jsonify({"status": "success", "record": updated_record})


@app.route("/detect", methods=["POST"])
def detect():
    """Secondary helmet-detection endpoint backed by the HF Gradio Space.

    Accepts JSON body:
      { "image_url": "https://..." }          — publicly reachable URL
      { "image_path": "static/no_helmet/…" }  — server-local file path

    Optional fields (all have defaults):
      conf_threshold  float  0.4
      image_size      int    640
      extract_text    bool   true
      ocr_model       str    "auto"

    Returns JSON:
      {
        "mode": "remote" | "local",
        "has_violation": bool,
        "detections": [{"label", "confidence", "x1", "y1", "x2", "y2"}, ...],
        "ocr_text": str,
        "stats": str,
        "annotated_image_url": str | null,
        "error": str | null
      }
    """
    if _hf_detector is None:
        return jsonify({"status": "error", "message": "Detection service unavailable"}), 503

    payload = request.get_json(silent=True) or {}
    image_url = payload.get("image_url")
    image_path = payload.get("image_path")

    if not image_url and not image_path:
        return jsonify({"status": "error", "message": "Provide 'image_url' or 'image_path'"}), 400

    # Resolve a relative image_path against BASE_DIR for safety
    if image_path:
        resolved = (BASE_DIR / image_path).resolve()
        if not str(resolved).startswith(str(BASE_DIR)):
            return jsonify({"status": "error", "message": "Path traversal not allowed"}), 400
        image_path = str(resolved)
        if not Path(image_path).is_file():
            return jsonify({"status": "error", "message": f"File not found: {image_path}"}), 404

    conf_threshold = float(payload.get("conf_threshold", 0.4))
    image_size = int(payload.get("image_size", 640))
    extract_text = bool(payload.get("extract_text", True))
    ocr_model = str(payload.get("ocr_model", "auto"))

    result = _hf_detector.detect(
        image_path=image_path,
        image_url=image_url,
        conf_threshold=conf_threshold,
        image_size=image_size,
        extract_text=extract_text,
        selected_ocr_model=ocr_model,
    )

    return jsonify(
        {
            "mode": result.mode,
            "has_violation": result.has_violation,
            "detections": result.detections,
            "ocr_text": result.ocr_text,
            "stats": result.stats,
            "annotated_image_url": result.annotated_image_url,
            "error": result.error,
        }
    )


@app.route("/speed_estimate", methods=["POST"])
def speed_estimate():
    """Advanced / optional endpoint — Vehicle Speed Estimation.

    Submits a local MP4 file to the HF Speed Estimation Space and returns
    vehicle counts and speed statistics.

    Because the Space runs on the free tier it may be sleeping and take
    up to ~60 s to respond on the first request.

    Accepts JSON body:
      { "video_path": "static/captured/traffic.mp4" }   (relative to BASE_DIR)

    Optional fields:
      model_choice             str    "yolov8n" | "yolov8s" | "yolov8m" | "yolov8l"
      line_position            int    480   (Y pixel for the counting line)
      confidence_threshold     float  0.3

    Returns JSON:
      {
        "available":          bool,
        "mode":               "remote" | "unavailable",
        "total_count":        int,
        "in_count":           int,
        "out_count":          int,
        "avg_speed_kmh":      float,
        "max_speed_kmh":      float,
        "min_speed_kmh":      float,
        "frames_processed":   int,
        "processing_time_s":  float,
        "model_used":         str,
        "output_video_url":   str | null,
        "stats_raw":          str,
        "error":              str | null
      }
    """
    if _speed_estimator is None:
        return jsonify(
            {
                "available": False,
                "error": "Speed estimation module failed to import — check server logs.",
            }
        ), 503

    payload = request.get_json(silent=True) or {}
    video_path = payload.get("video_path")

    if not video_path:
        return jsonify({"available": False, "error": "Provide 'video_path' in the request body."}), 400

    # Resolve and validate path
    resolved = (BASE_DIR / video_path).resolve()
    if not str(resolved).startswith(str(BASE_DIR)):
        return jsonify({"available": False, "error": "Path traversal not allowed."}), 400

    model_choice         = str(payload.get("model_choice", "yolov8n"))
    line_position        = int(payload.get("line_position", 480))
    confidence_threshold = float(payload.get("confidence_threshold", 0.3))

    result = _speed_estimator.estimate(
        video_path=str(resolved),
        model_choice=model_choice,
        line_position=line_position,
        confidence_threshold=confidence_threshold,
    )

    return jsonify(
        {
            "available":         result.available,
            "mode":              result.mode,
            "total_count":       result.total_count,
            "in_count":          result.in_count,
            "out_count":         result.out_count,
            "avg_speed_kmh":     result.avg_speed_kmh,
            "max_speed_kmh":     result.max_speed_kmh,
            "min_speed_kmh":     result.min_speed_kmh,
            "frames_processed":  result.frames_processed,
            "processing_time_s": result.processing_time_s,
            "model_used":        result.model_used,
            "output_video_url":  result.output_video_url,
            "stats_raw":         result.stats_raw,
            "error":             result.error,
        }
    )



@app.route("/demo_samples")
def demo_samples():
    """Return a JSON list of available dataset sample images grouped by class.

    Response:
      {
        "double_riding":  ["dataset_samples/double_riding/images/foo.jpg", ...],
        "single_rider":   [...],
        "triple_riding":  [...],
      }
    """
    samples_dir = BASE_DIR / "dataset_samples"
    result: dict[str, list[str]] = {}
    for category_dir in sorted(samples_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        images_dir = category_dir / "images"
        if not images_dir.exists():
            continue
        files = sorted(
            f"dataset_samples/{category_dir.name}/images/{f.name}"
            for f in images_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if files:
            result[category_dir.name] = files
    return jsonify(result)


@app.route("/dataset_samples/<path:filename>")
def serve_dataset_sample(filename: str):
    """Serve a file from the dataset_samples directory."""
    safe = (BASE_DIR / "dataset_samples" / filename).resolve()
    if not str(safe).startswith(str(BASE_DIR / "dataset_samples")):
        return jsonify({"error": "Forbidden"}), 403
    if not safe.is_file():
        return jsonify({"error": "Not found"}), 404
    return send_file(safe)


@app.route("/demo_detect", methods=["POST"])
def demo_detect():
    """Run helmet detection on a dataset sample image (URL path relative to BASE_DIR).

    Accepts JSON: { "sample_path": "dataset_samples/double_riding/images/foo.jpg" }
    Returns the same shape as /detect.
    """
    if _hf_detector is None:
        return jsonify({"status": "error", "message": "Detection service unavailable"}), 503

    payload = request.get_json(silent=True) or {}
    sample_path = payload.get("sample_path", "")

    resolved = (BASE_DIR / sample_path).resolve()
    if not str(resolved).startswith(str(BASE_DIR / "dataset_samples")):
        return jsonify({"status": "error", "message": "Forbidden path"}), 403
    if not resolved.is_file():
        return jsonify({"status": "error", "message": "Sample not found"}), 404

    result = _hf_detector.detect_from_path(
        str(resolved),
        conf_threshold=float(payload.get("conf_threshold", 0.4)),
        extract_text=bool(payload.get("extract_text", True)),
    )

    return jsonify({
        "mode": result.mode,
        "has_violation": result.has_violation,
        "detections": result.detections,
        "ocr_text": result.ocr_text,
        "stats": result.stats,
        "annotated_image_url": result.annotated_image_url,
        "error": result.error,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

