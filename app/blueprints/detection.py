"""app/blueprints/detection.py — Helmet detection routes (primary + demo)."""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request, send_file

from app.config import BASE_DIR, DATASET_DIR, IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

bp = Blueprint("detection", __name__)

# ── Lazy detector singleton ───────────────────────────────────────────────────
# Imported at module level but instantiated lazily so the app starts even
# when ultralytics / gradio_client are absent.

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        try:
            from hf_space_client import HFDetector
            _detector = HFDetector()
            logger.info("HFDetector initialised.")
        except Exception as exc:
            logger.warning("HFDetector unavailable: %s", exc)
            _detector = False          # sentinel: tried but failed
    return _detector if _detector is not False else None


# ── /detect ───────────────────────────────────────────────────────────────────

@bp.post("/detect")
def detect():
    """Run helmet + licence-plate detection on an image URL or local path.

    Request JSON:
      { "image_url": "https://..." }       — public image URL
      { "image_path": "static/..." }       — path relative to project root

    Response JSON: mode, has_violation, detections, ocr_text, stats,
                   annotated_image_url, error
    """
    detector = _get_detector()
    if detector is None:
        return jsonify({"status": "error", "message": "Detection service unavailable"}), 503

    payload    = request.get_json(silent=True) or {}
    image_url  = payload.get("image_url")
    image_path = payload.get("image_path")

    if not image_url and not image_path:
        return jsonify({"status": "error", "message": "Provide image_url or image_path"}), 400

    if image_path:
        resolved = (BASE_DIR / image_path).resolve()
        if not str(resolved).startswith(str(BASE_DIR)):
            return jsonify({"status": "error", "message": "Path traversal not allowed"}), 400
        if not resolved.is_file():
            return jsonify({"status": "error", "message": "File not found"}), 404
        image_path = str(resolved)

    result = detector.detect(
        image_url=image_url,
        image_path=image_path,
        conf_threshold=float(payload.get("conf_threshold", 0.4)),
        extract_text=bool(payload.get("extract_text", True)),
    )

    return jsonify({
        "mode":                result.mode,
        "has_violation":       result.has_violation,
        "detections":          result.detections,
        "ocr_text":            result.ocr_text,
        "stats":               result.stats,
        "annotated_image_url": result.annotated_image_url,
        "error":               result.error,
    })


# ── /demo_samples ─────────────────────────────────────────────────────────────

@bp.get("/demo_samples")
def demo_samples():
    """List sample images from dataset_samples/ grouped by class.

    Response: { "double_riding": ["dataset_samples/double_riding/images/x.jpg", ...], ... }
    """
    result: dict[str, list[str]] = {}
    if not DATASET_DIR.exists():
        return jsonify(result)

    for cat_dir in sorted(DATASET_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        img_dir = cat_dir / "images"
        if not img_dir.exists():
            continue
        files = sorted(
            f"dataset_samples/{cat_dir.name}/images/{f.name}"
            for f in img_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if files:
            result[cat_dir.name] = files
    return jsonify(result)


# ── /dataset_samples/<path> ───────────────────────────────────────────────────

@bp.get("/dataset_samples/<path:filename>")
def serve_dataset_sample(filename: str):
    """Serve a file from dataset_samples/ (path-traversal protected)."""
    safe = (DATASET_DIR / filename).resolve()
    if not str(safe).startswith(str(DATASET_DIR)):
        return jsonify({"error": "Forbidden"}), 403
    if not safe.is_file():
        return jsonify({"error": "Not found"}), 404
    return send_file(safe)


# ── /demo_detect ──────────────────────────────────────────────────────────────

@bp.post("/demo_detect")
def demo_detect():
    """Run detection on a dataset sample by its relative path.

    Request JSON: { "sample_path": "dataset_samples/double_riding/images/foo.jpg" }
    """
    detector = _get_detector()
    if detector is None:
        return jsonify({"status": "error", "message": "Detection service unavailable"}), 503

    payload     = request.get_json(silent=True) or {}
    sample_path = payload.get("sample_path", "")
    if not sample_path:
        return jsonify({"status": "error", "message": "sample_path is required"}), 400

    resolved = (BASE_DIR / sample_path).resolve()
    if not str(resolved).startswith(str(DATASET_DIR)):
        return jsonify({"status": "error", "message": "Forbidden path"}), 403
    if not resolved.is_file():
        return jsonify({"status": "error", "message": "Sample file not found"}), 404

    result = detector.detect_from_path(
        str(resolved),
        conf_threshold=float(payload.get("conf_threshold", 0.4)),
        extract_text=bool(payload.get("extract_text", True)),
    )

    return jsonify({
        "mode":                result.mode,
        "has_violation":       result.has_violation,
        "detections":          result.detections,
        "ocr_text":            result.ocr_text,
        "stats":               result.stats,
        "annotated_image_url": result.annotated_image_url,
        "error":               result.error,
    })
