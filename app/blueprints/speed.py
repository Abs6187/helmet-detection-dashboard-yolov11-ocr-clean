"""app/blueprints/speed.py — Vehicle speed estimation route (advanced/optional)."""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from app.config import BASE_DIR

logger = logging.getLogger(__name__)

bp = Blueprint("speed", __name__)

# ── Lazy estimator singleton ──────────────────────────────────────────────────

_estimator = None


def _get_estimator():
    global _estimator
    if _estimator is None:
        try:
            from hf_space_client import SpeedEstimator
            _estimator = SpeedEstimator()
            logger.info("SpeedEstimator initialised.")
        except Exception as exc:
            logger.warning("SpeedEstimator unavailable: %s", exc)
            _estimator = False
    return _estimator if _estimator is not False else None


# ── /speed_estimate ───────────────────────────────────────────────────────────

@bp.post("/speed_estimate")
def speed_estimate():
    """Submit a local MP4 to the HF Speed Estimation Space (optional feature).

    The Space runs on the free tier and hibernates — first call may take 30–60 s.
    If HF_TOKEN is absent or the Space is unreachable, returns {available: false}
    with a descriptive error — never a 5xx crash.

    Request JSON:
      {
        "video_path":           "static/captured/traffic.mp4",   // relative to project root
        "model_choice":         "yolov8n",                        // optional
        "line_position":        480,                              // optional
        "confidence_threshold": 0.3                               // optional
      }

    Response JSON:
      available, mode, total_count, in_count, out_count,
      avg_speed_kmh, max_speed_kmh, min_speed_kmh,
      frames_processed, processing_time_s, model_used,
      output_video_url, stats_raw, error
    """
    estimator = _get_estimator()
    if estimator is None:
        return jsonify({
            "available": False,
            "error": "Speed estimation module failed to import — check server logs.",
        }), 503

    payload    = request.get_json(silent=True) or {}
    video_path = payload.get("video_path")

    if not video_path:
        return jsonify({"available": False, "error": "video_path is required"}), 400

    resolved = (BASE_DIR / video_path).resolve()
    if not str(resolved).startswith(str(BASE_DIR)):
        return jsonify({"available": False, "error": "Path traversal not allowed"}), 400

    result = estimator.estimate(
        video_path=str(resolved),
        model_choice=str(payload.get("model_choice", "yolov8n")),
        line_position=int(payload.get("line_position", 480)),
        confidence_threshold=float(payload.get("confidence_threshold", 0.3)),
    )

    return jsonify({
        "available":          result.available,
        "mode":               result.mode,
        "total_count":        result.total_count,
        "in_count":           result.in_count,
        "out_count":          result.out_count,
        "avg_speed_kmh":      result.avg_speed_kmh,
        "max_speed_kmh":      result.max_speed_kmh,
        "min_speed_kmh":      result.min_speed_kmh,
        "frames_processed":   result.frames_processed,
        "processing_time_s":  result.processing_time_s,
        "model_used":         result.model_used,
        "output_video_url":   result.output_video_url,
        "stats_raw":          result.stats_raw,
        "error":              result.error,
    })
