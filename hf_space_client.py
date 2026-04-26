"""
hf_space_client.py
==================
Calls the Hugging Face Gradio Space:
  https://huggingface.co/spaces/Abs6187/Helmet-License-Plate-Detection

Strategy
--------
PRIMARY  → Remote Space via gradio_client  (requires HF_TOKEN env var)
FALLBACK → Local YOLO inference with best.pt (always available on disk)

The fallback is triggered automatically when:
  - HF_TOKEN env var is missing or empty
  - gradio_client package is not installed
  - The Space is unavailable / times out
  - Any network or API error occurs

Public API
----------
    from hf_space_client import HFDetector, DetectionResult

    detector = HFDetector()
    result   = detector.detect_from_path("static/no_helmet/snap.jpg")

    result.mode              # "remote" | "local"
    result.has_violation     # True if Without Helmet detected
    result.ocr_text          # "MH12AB1234"
    result.detections        # [{"label": "Without Helmet", "confidence": 0.87}, ...]
    result.annotated_image_url   # URL or None
    result.stats             # human-readable summary string
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPACE_ID = "Abs6187/Helmet-License-Plate-Detection"
SPACE_HOST = "https://abs6187-helmet-license-plate-detection.hf.space"

# Label used by the custom helmet model
WITHOUT_HELMET_LABEL = "Without Helmet"

# Default local model path — best.pt is committed to the repo
LOCAL_MODEL_PATH = str(Path(__file__).parent / "best.pt")

# Remote call timeout (seconds)
REMOTE_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Unified result from either remote or local detection."""

    mode: str = "unknown"
    """'remote' when the HF Space was used, 'local' when the fallback ran."""

    has_violation: bool = False
    """True when a 'Without Helmet' class is in detections."""

    detections: list[dict[str, Any]] = field(default_factory=list)
    """List of dicts: {label, confidence, x1, y1, x2, y2}"""

    annotated_image_url: str | None = None
    """URL of the annotated image (remote only; None for local)."""

    annotated_image_path: str | None = None
    """Local path to the annotated image (local mode) or temp download (remote)."""

    plate_image_urls: list[str] = field(default_factory=list)
    """Cropped licence plate image URLs (remote only)."""

    ocr_text: str = ""
    """OCR text from licence plates."""

    stats: str = ""
    """Human-readable detection summary."""

    error: str | None = None
    """Set to the error message if both remote AND local failed."""

    raw: Any = field(default=None, repr=False)
    """Raw response from gradio_client.predict() (remote) or YOLO result (local)."""


# ---------------------------------------------------------------------------
# Remote backend (HF Gradio Space)
# ---------------------------------------------------------------------------

class _RemoteBackend:
    """Thin wrapper around gradio_client that calls /yolov8_detect."""

    def __init__(self, hf_token: str) -> None:
        self._token = hf_token
        self._client: Any = None

    def _client_instance(self):
        if self._client is None:
            from gradio_client import Client  # type: ignore[import-untyped]
            self._client = Client(src=SPACE_ID, hf_token=self._token)
        return self._client

    def predict(
        self,
        *,
        image_path: str | None = None,
        image_url: str | None = None,
        image_size: float = 640,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.5,
        show_stats: bool = True,
        crop_plates: bool = True,
        extract_text: bool = True,
        ocr_on_no_helmet: bool = True,
        selected_ocr_model: str = "auto",
    ) -> DetectionResult:
        from gradio_client import handle_file  # type: ignore[import-untyped]

        if image_path:
            image_input = handle_file(image_path)
        elif image_url:
            image_input = handle_file(image_url)
        else:
            raise ValueError("Provide image_path or image_url.")

        client = self._client_instance()
        raw = client.predict(
            image=image_input,
            image_size=image_size,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            show_stats=show_stats,
            crop_plates=crop_plates,
            extract_text=extract_text,
            ocr_on_no_helmet=ocr_on_no_helmet,
            selected_ocr_model=selected_ocr_model,
            api_name="/yolov8_detect",
        )
        return _parse_remote_result(raw)


def _parse_remote_result(raw: tuple) -> DetectionResult:
    """Convert the 6-tuple from /yolov8_detect into a DetectionResult."""
    if not raw or len(raw) < 6:
        return DetectionResult(mode="remote", raw=raw)

    annotated_img, details_df, stats_str, plates_gallery, zip_file, ocr_str = raw

    # annotated image
    ann_url = ann_path = None
    if isinstance(annotated_img, dict):
        ann_url = annotated_img.get("url") or annotated_img.get("path")
        ann_path = annotated_img.get("path")
    elif isinstance(annotated_img, str):
        ann_url = annotated_img

    # detection rows
    detections: list[dict[str, Any]] = []
    if isinstance(details_df, dict):
        headers: list[str] = details_df.get("headers", [])
        for row in details_df.get("data", []):
            detections.append(dict(zip(headers, row)))

    # plate gallery
    plate_urls: list[str] = []
    if isinstance(plates_gallery, list):
        for item in plates_gallery:
            if isinstance(item, dict):
                img = item.get("image", {})
                url = img.get("url") or img.get("path") if isinstance(img, dict) else None
                if url:
                    plate_urls.append(url)

    has_violation = any(
        str(d.get("Class", d.get("label", ""))).strip() == WITHOUT_HELMET_LABEL
        for d in detections
    )

    return DetectionResult(
        mode="remote",
        has_violation=has_violation,
        detections=detections,
        annotated_image_url=ann_url,
        annotated_image_path=ann_path,
        plate_image_urls=plate_urls,
        ocr_text=ocr_str or "",
        stats=stats_str or "",
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Local fallback backend (YOLO on best.pt)
# ---------------------------------------------------------------------------

class _LocalBackend:
    """Runs YOLO locally using the committed best.pt model."""

    def __init__(self, model_path: str = LOCAL_MODEL_PATH) -> None:
        self._model_path = model_path
        self._model: Any = None

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO  # type: ignore[import-untyped]
            self._model = YOLO(self._model_path)
            logger.info("Local YOLO model loaded from: %s", self._model_path)
        return self._model

    def predict(
        self,
        *,
        image_path: str | None = None,
        image_url: str | None = None,
        conf_threshold: float = 0.4,
        image_size: int = 640,
        **_kwargs,
    ) -> DetectionResult:
        source = image_path or image_url
        if not source:
            raise ValueError("Provide image_path or image_url.")

        model = self._load_model()
        t0 = time.perf_counter()
        results = model.predict(source=source, conf=conf_threshold, imgsz=image_size, verbose=False)
        elapsed = time.perf_counter() - t0

        result = results[0]
        detections: list[dict[str, Any]] = []

        if result.boxes is not None and result.boxes.cls is not None:
            xyxy = result.boxes.xyxy.int().tolist()
            confs = result.boxes.conf.tolist()
            classes = result.boxes.cls.int().tolist()
            for box, conf, cls_id in zip(xyxy, confs, classes):
                label = result.names[int(cls_id)]
                x1, y1, x2, y2 = box
                detections.append(
                    {
                        "label": label,
                        "confidence": round(float(conf), 3),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )

        has_violation = any(d["label"] == WITHOUT_HELMET_LABEL for d in detections)
        counts: dict[str, int] = {}
        for d in detections:
            counts[d["label"]] = counts.get(d["label"], 0) + 1
        stats = (
            f"{len(detections)} detection(s) in {elapsed:.2f}s — "
            + ", ".join(f"{v}× {k}" for k, v in counts.items())
            if detections
            else f"No detections ({elapsed:.2f}s)"
        )

        return DetectionResult(
            mode="local",
            has_violation=has_violation,
            detections=detections,
            stats=stats,
            raw=result,
        )


# ---------------------------------------------------------------------------
# Main detector — tries remote first, falls back to local
# ---------------------------------------------------------------------------

class HFDetector:
    """Primary: HF Gradio Space. Fallback: local YOLO (best.pt).

    Parameters
    ----------
    hf_token:
        HF token. Reads HF_TOKEN env var by default.
        Pass an empty string to skip remote and always use local.
    local_model_path:
        Path to the local YOLO .pt file used as fallback.
    remote_timeout:
        Seconds before giving up on the remote call.
    """

    def __init__(
        self,
        hf_token: str | None = None,
        local_model_path: str = LOCAL_MODEL_PATH,
        remote_timeout: int = REMOTE_TIMEOUT,
    ) -> None:
        self._token = hf_token if hf_token is not None else os.environ.get("HF_TOKEN", "")
        self._remote = _RemoteBackend(self._token) if self._token else None
        self._local = _LocalBackend(local_model_path)
        self._timeout = remote_timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remote_available(self) -> bool:
        """Return True when remote should be attempted."""
        if not self._token:
            logger.info("HF_TOKEN not set — using local fallback.")
            return False
        try:
            import gradio_client  # noqa: F401  # type: ignore[import-untyped]
            return True
        except ImportError:
            logger.warning("gradio_client not installed — using local fallback.")
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        *,
        image_path: str | None = None,
        image_url: str | None = None,
        **kwargs,
    ) -> DetectionResult:
        """Run detection. Tries HF Space first; falls back to local YOLO.

        Parameters
        ----------
        image_path: absolute or relative path to a local image file.
        image_url:  publicly accessible image URL.
        **kwargs:   forwarded to both backends (conf_threshold, image_size, …).
        """
        if self._remote_available():
            try:
                logger.debug("Calling HF Space (remote)…")
                result = self._remote.predict(  # type: ignore[union-attr]
                    image_path=image_path,
                    image_url=image_url,
                    **kwargs,
                )
                logger.info("Remote detection succeeded. mode=remote")
                return result
            except Exception as exc:
                logger.warning(
                    "Remote HF Space call failed (%s). Falling back to local YOLO.", exc
                )

        # ── Fallback ──────────────────────────────────────────────────
        try:
            logger.debug("Running local YOLO fallback…")
            result = self._local.predict(
                image_path=image_path,
                image_url=image_url,
                **kwargs,
            )
            logger.info("Local fallback detection succeeded.")
            return result
        except Exception as exc:
            logger.error("Local YOLO fallback also failed: %s", exc)
            return DetectionResult(
                mode="local",
                error=str(exc),
                stats="Detection failed — check model file and image path.",
            )

    def detect_from_path(self, image_path: str, **kwargs) -> DetectionResult:
        """Convenience: pass a local file path."""
        return self.detect(image_path=image_path, **kwargs)

    def detect_from_url(self, image_url: str, **kwargs) -> DetectionResult:
        """Convenience: pass a public image URL."""
        return self.detect(image_url=image_url, **kwargs)


# ===========================================================================
# ADVANCED / OPTIONAL FEATURE
# Vehicle Speed Estimation + Counting
# HF Space: https://huggingface.co/spaces/Abs6187/Vehicle_Speed_Estimation_and_Counting
# ===========================================================================

SPEED_SPACE_ID = "Abs6187/Vehicle_Speed_Estimation_and_Counting"
SPEED_SPACE_HOST = "https://abs6187-vehicle-speed-estimation-and-counting.hf.space"

# The Space hibernates on the free tier — wake-up can take 30-60 s.
SPEED_REMOTE_TIMEOUT = 300  # seconds


@dataclass
class SpeedEstimationResult:
    """Result from the Vehicle Speed Estimation Space.

    The Space is OPTIONAL — if unavailable the route returns a graceful error
    rather than crashing the application.
    """

    available: bool = False
    """False when the Space is sleeping / unreachable or HF_TOKEN is absent."""

    mode: str = "unavailable"
    """'remote' when the Space responded, 'unavailable' otherwise."""

    # ── video ──────────────────────────────────────────────────────────────

    output_video_path: str | None = None
    """Local temp path to the annotated output video (MP4)."""

    output_video_url: str | None = None
    """URL served by HF for the annotated video (if available)."""

    # ── statistics ─────────────────────────────────────────────────────────

    total_count: int = 0
    avg_speed_kmh: float = 0.0
    max_speed_kmh: float = 0.0
    min_speed_kmh: float = 0.0
    in_count: int = 0
    out_count: int = 0
    frames_processed: int = 0
    processing_time_s: float = 0.0
    model_used: str = ""

    stats_raw: str = ""
    """Raw markdown stats string returned by the Space."""

    error: str | None = None
    """Error message when the call failed."""

    raw: Any = field(default=None, repr=False)


def _parse_speed_stats(stats_text: str) -> dict[str, Any]:
    """
    Loosely parse the markdown stats block returned by the Space.

    The block looks like::

        ### Vehicle Count Statistics
        - **Total Vehicles Detected:** 12
        - **Vehicles Entering (In):** 8
        - **Vehicles Exiting (Out):** 4

        ### Speed Analysis
        - **Average Speed:** 45.3 km/h
        - **Maximum Speed:** 72.1 km/h
        - **Minimum Speed:** 18.6 km/h

        ### Processing Information
        - **Frames Processed:** 900
        - **Processing Time:** 12.34 seconds
        - **Model Used:** yolov8n
        - **Detection Confidence:** 0.30
    """
    import re

    def _int(pattern: str) -> int:
        m = re.search(pattern, stats_text)
        return int(m.group(1)) if m else 0

    def _float(pattern: str) -> float:
        m = re.search(pattern, stats_text)
        return float(m.group(1)) if m else 0.0

    def _str(pattern: str) -> str:
        m = re.search(pattern, stats_text)
        return m.group(1).strip() if m else ""

    return {
        "total_count":       _int(r"Total Vehicles Detected.*?(\d+)"),
        "in_count":          _int(r"Vehicles Entering.*?(\d+)"),
        "out_count":         _int(r"Vehicles Exiting.*?(\d+)"),
        "avg_speed_kmh":     _float(r"Average Speed.*?([\d.]+)\s*km/h"),
        "max_speed_kmh":     _float(r"Maximum Speed.*?([\d.]+)\s*km/h"),
        "min_speed_kmh":     _float(r"Minimum Speed.*?([\d.]+)\s*km/h"),
        "frames_processed":  _int(r"Frames Processed.*?(\d+)"),
        "processing_time_s": _float(r"Processing Time.*?([\d.]+)\s*seconds"),
        "model_used":        _str(r"Model Used[^:]*:\*{0,2}\s*([\w.]+)"),
    }


class SpeedEstimator:
    """Optional client for the Vehicle Speed Estimation HF Space.

    The Space accepts a **video file** and returns an annotated video plus
    speed/count statistics.  Because the Space runs on the free tier it may
    be sleeping — the first request can take up to a minute to wake it up.

    Parameters
    ----------
    hf_token:
        HF token.  Reads HF_TOKEN env var by default.
        If absent the estimator is automatically disabled and every call
        returns ``SpeedEstimationResult(available=False)``.
    wake_timeout:
        How long (seconds) to wait for the Space to wake up and respond.
        Default 300 s (5 min) — generous because free-tier cold-starts are slow.
    """

    def __init__(
        self,
        hf_token: str | None = None,
        wake_timeout: int = SPEED_REMOTE_TIMEOUT,
    ) -> None:
        self._token = hf_token if hf_token is not None else os.environ.get("HF_TOKEN", "")
        self._timeout = wake_timeout
        self._client: Any = None
        self._enabled = bool(self._token)

        if not self._enabled:
            logger.info(
                "SpeedEstimator: HF_TOKEN not set — speed estimation will be unavailable."
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _client_instance(self):
        if self._client is None:
            from gradio_client import Client  # type: ignore[import-untyped]
            logger.info("SpeedEstimator: connecting to %s (may wake sleeping Space)…", SPEED_SPACE_ID)
            self._client = Client(src=SPEED_SPACE_ID, hf_token=self._token)
        return self._client

    @property
    def enabled(self) -> bool:
        """True when the token is present and gradio_client is installed."""
        if not self._token:
            return False
        try:
            import gradio_client  # noqa: F401  # type: ignore[import-untyped]
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        video_path: str,
        *,
        model_choice: str = "yolov8n",
        line_position: int = 480,
        confidence_threshold: float = 0.3,
    ) -> SpeedEstimationResult:
        """Submit a video to the Speed Estimation Space.

        Parameters
        ----------
        video_path:
            Absolute path to an MP4 file on the local filesystem.
        model_choice:
            YOLO model variant: ``"yolov8n"`` | ``"yolov8s"`` | ``"yolov8m"`` | ``"yolov8l"``.
        line_position:
            Y-coordinate of the counting line (100–1000, default 480).
        confidence_threshold:
            Detection confidence (0.1–0.9, default 0.3).

        Returns
        -------
        SpeedEstimationResult
            ``available=False`` if the Space is disabled or unreachable.
            ``available=True`` + populated stats on success.
        """
        if not self.enabled:
            return SpeedEstimationResult(
                available=False,
                error="Speed estimation unavailable: HF_TOKEN not set or gradio_client not installed.",
            )

        if not Path(video_path).is_file():
            return SpeedEstimationResult(
                available=False,
                error=f"Video file not found: {video_path}",
            )

        try:
            from gradio_client import handle_file  # type: ignore[import-untyped]

            client = self._client_instance()
            logger.info("SpeedEstimator: submitting video %s", video_path)

            raw = client.predict(
                video_file=handle_file(video_path),
                model_choice=model_choice,
                line_position=line_position,
                confidence_threshold=confidence_threshold,
                api_name="/estimate_vehicle_speed",
            )

            # raw = (output_video_path_or_dict, stats_markdown_str)
            out_video, stats_text = raw if len(raw) >= 2 else (None, "")

            # normalise video path/url
            out_path = out_url = None
            if isinstance(out_video, dict):
                out_path = out_video.get("path")
                out_url = out_video.get("url") or out_path
            elif isinstance(out_video, str):
                out_path = out_url = out_video

            parsed = _parse_speed_stats(stats_text or "")

            return SpeedEstimationResult(
                available=True,
                mode="remote",
                output_video_path=out_path,
                output_video_url=out_url,
                stats_raw=stats_text or "",
                error=None,
                raw=raw,
                **parsed,
            )

        except Exception as exc:
            logger.warning("SpeedEstimator: remote call failed — %s", exc)
            return SpeedEstimationResult(
                available=False,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Smoke-test  (python hf_space_client.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    TEST_URL = (
        "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png"
    )

    token = os.environ.get("HF_TOKEN", "")
    print(f"HF_TOKEN present : {'yes' if token else 'NO — will use local fallback'}")
    print(f"Space            : {SPACE_ID}")
    print(f"Local model      : {LOCAL_MODEL_PATH}\n")

    detector = HFDetector()
    result = detector.detect_from_url(TEST_URL, extract_text=True)

    print(f"Mode             : {result.mode}")
    print(f"Has violation    : {result.has_violation}")
    print(f"Stats            : {result.stats}")
    print(f"OCR text         : {result.ocr_text!r}")
    print(f"Ann. image URL   : {result.annotated_image_url}")
    print(f"Detections ({len(result.detections)}):")
    for d in result.detections:
        print(" ", json.dumps(d, default=str))
    if result.error:
        print(f"ERROR            : {result.error}")

    print("\n--- Speed Estimator ---")
    speed = SpeedEstimator()
    print(f"Enabled          : {speed.enabled}")
