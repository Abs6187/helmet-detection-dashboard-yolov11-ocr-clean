"""
tests/test_hf_space_client.py
==============================
Unit tests for hf_space_client.py — no network required, no heavy ML deps.
All remote calls are mocked.  The /detect route tests use the app factory.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_raw_remote_tuple(
    ann_url: str = "https://hf.space/annotated.jpg",
    details_headers: list[str] | None = None,
    details_data: list[list] | None = None,
    stats: str = "2 detections",
    plates: list | None = None,
    zip_path: str | None = None,
    ocr: str = "MH12AB1234",
) -> tuple:
    headers    = details_headers or ["Class", "Confidence", "X1", "Y1", "X2", "Y2"]
    data_rows  = details_data    or [["Without Helmet", 0.87, 10, 20, 100, 200]]
    details_df = {"headers": headers, "data": data_rows}
    ann_dict   = {"url": ann_url, "path": "/tmp/annotated.jpg"}
    gallery    = [{"image": {"url": "https://hf.space/plate.jpg"}}]
    return (ann_dict, details_df, stats, plates or gallery, zip_path, ocr)


# ── DetectionResult ──────────────────────────────────────────────────────────

class TestDetectionResult:
    def test_default_values(self):
        from hf_space_client import DetectionResult
        r = DetectionResult()
        assert r.mode == "unknown"
        assert r.has_violation is False
        assert r.detections == []
        assert r.ocr_text == ""
        assert r.error is None

    def test_accepts_all_fields(self):
        from hf_space_client import DetectionResult
        r = DetectionResult(mode="remote", has_violation=True, ocr_text="AB1234")
        assert r.mode == "remote"
        assert r.has_violation is True
        assert r.ocr_text == "AB1234"


# ── _parse_remote_result ─────────────────────────────────────────────────────

class TestParseRemoteResult:
    def test_parses_annotated_image_url(self):
        from hf_space_client import _parse_remote_result
        raw = _make_raw_remote_tuple(ann_url="https://example.com/out.jpg")
        assert _parse_remote_result(raw).annotated_image_url == "https://example.com/out.jpg"

    def test_parses_detections(self):
        from hf_space_client import _parse_remote_result
        raw = _make_raw_remote_tuple(
            details_headers=["Class", "Confidence"],
            details_data=[["Without Helmet", 0.91], ["License Plate", 0.78]],
        )
        result = _parse_remote_result(raw)
        assert len(result.detections) == 2
        assert result.detections[0]["Class"] == "Without Helmet"

    def test_sets_has_violation_true(self):
        from hf_space_client import _parse_remote_result
        raw = _make_raw_remote_tuple(details_headers=["Class","Confidence"], details_data=[["Without Helmet", 0.85]])
        assert _parse_remote_result(raw).has_violation is True

    def test_sets_has_violation_false_when_helmet_absent(self):
        from hf_space_client import _parse_remote_result
        raw = _make_raw_remote_tuple(details_headers=["Class","Confidence"], details_data=[["License Plate", 0.92]])
        assert _parse_remote_result(raw).has_violation is False

    def test_parses_ocr_text(self):
        from hf_space_client import _parse_remote_result
        assert _parse_remote_result(_make_raw_remote_tuple(ocr="KA05MN5678")).ocr_text == "KA05MN5678"

    def test_parses_plate_gallery(self):
        from hf_space_client import _parse_remote_result
        assert _parse_remote_result(_make_raw_remote_tuple()).plate_image_urls == ["https://hf.space/plate.jpg"]

    def test_returns_empty_result_on_short_tuple(self):
        from hf_space_client import _parse_remote_result
        r = _parse_remote_result(())
        assert r.mode == "remote"
        assert r.detections == []

    def test_stats_field(self):
        from hf_space_client import _parse_remote_result
        assert _parse_remote_result(_make_raw_remote_tuple(stats="3 objects")).stats == "3 objects"


# ── HFDetector — no token ────────────────────────────────────────────────────

class TestHFDetectorNoToken:
    def _make(self, tmp_path):
        from hf_space_client import DetectionResult, HFDetector
        det = HFDetector(hf_token="")
        det._local.predict = MagicMock(return_value=DetectionResult(mode="local", has_violation=True, stats="1×"))
        return det

    def test_uses_local_when_no_token(self, tmp_path):
        det = self._make(tmp_path)
        assert det.detect_from_url("https://example.com/img.jpg").mode == "local"

    def test_local_predict_called_with_correct_url(self, tmp_path):
        det = self._make(tmp_path)
        det.detect_from_url("https://example.com/frame.jpg", conf_threshold=0.5)
        kw = det._local.predict.call_args.kwargs
        assert kw["image_url"] == "https://example.com/frame.jpg"
        assert kw["conf_threshold"] == 0.5

    def test_detect_from_path_passes_path(self, tmp_path):
        det = self._make(tmp_path)
        det.detect_from_path(str(tmp_path / "snap.jpg"))
        assert det._local.predict.call_args.kwargs["image_path"] == str(tmp_path / "snap.jpg")

    def test_returns_error_result_when_local_also_fails(self, tmp_path):
        from hf_space_client import HFDetector
        det = HFDetector(hf_token="")
        det._local.predict = MagicMock(side_effect=RuntimeError("model file missing"))
        result = det.detect_from_url("https://example.com/img.jpg")
        assert result.error is not None
        assert "model file missing" in result.error


# ── HFDetector — remote success ──────────────────────────────────────────────

class TestHFDetectorRemoteSuccess:
    def _make(self):
        from hf_space_client import DetectionResult, HFDetector
        det = HFDetector(hf_token="hf_faketoken")
        det._remote_available = MagicMock(return_value=True)
        det._remote = MagicMock()
        det._remote.predict = MagicMock(return_value=DetectionResult(mode="remote", ocr_text="DL1CAB1234"))
        return det

    def test_uses_remote_when_token_present(self):
        assert self._make().detect_from_url("https://example.com/img.jpg").mode == "remote"

    def test_remote_predict_receives_correct_kwargs(self):
        det = self._make()
        det.detect_from_url("https://example.com/img.jpg", conf_threshold=0.6, extract_text=False)
        kw = det._remote.predict.call_args.kwargs
        assert kw["image_url"] == "https://example.com/img.jpg"
        assert kw["conf_threshold"] == 0.6
        assert kw["extract_text"] is False


# ── HFDetector — fallback ────────────────────────────────────────────────────

class TestHFDetectorFallback:
    def test_falls_back_to_local_on_network_error(self):
        from hf_space_client import DetectionResult, HFDetector
        det = HFDetector(hf_token="hf_faketoken")
        det._remote_available = MagicMock(return_value=True)
        det._remote = MagicMock()
        det._remote.predict = MagicMock(side_effect=ConnectionError("timeout"))
        local_res = DetectionResult(mode="local")
        det._local.predict = MagicMock(return_value=local_res)
        assert det.detect_from_url("https://example.com/img.jpg").mode == "local"
        det._local.predict.assert_called_once()

    def test_falls_back_to_local_on_import_error(self):
        from hf_space_client import DetectionResult, HFDetector
        det = HFDetector(hf_token="hf_faketoken")
        det._remote_available = MagicMock(return_value=True)
        det._remote = MagicMock()
        det._remote.predict = MagicMock(side_effect=ImportError("gradio_client missing"))
        det._local.predict = MagicMock(return_value=DetectionResult(mode="local"))
        assert det.detect_from_url("https://example.com/img.jpg").mode == "local"


# ── /detect Flask route (via app factory) ────────────────────────────────────

@pytest.fixture()
def app_client(tmp_path, monkeypatch):
    import app.data as data_mod
    from app import config as cfg_mod, create_app
    from app.blueprints import detection as det_mod

    data_file  = tmp_path / "offenders_data.json"
    static_dir = tmp_path / "static"
    (static_dir / "triples").mkdir(parents=True, exist_ok=True)
    (static_dir / "no_helmet").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(data_mod, "DATA_FILE",  data_file)
    monkeypatch.setattr(data_mod, "STATIC_DIR", static_dir)
    monkeypatch.setattr(cfg_mod,  "DATA_FILE",  data_file)
    monkeypatch.setattr(cfg_mod,  "STATIC_DIR", static_dir)

    flask_app = create_app()
    flask_app.config["TESTING"] = True

    with flask_app.test_client() as c:
        yield c, tmp_path, det_mod


class TestDetectRoute:
    def test_missing_body_returns_400(self, app_client):
        c, _, _ = app_client
        resp = c.post("/detect", json={})
        assert resp.status_code == 400
        msg = resp.get_json()["message"].lower()
        assert "image_url" in msg or "image_path" in msg

    def test_returns_503_when_detector_none(self, app_client, monkeypatch):
        c, _, det_mod = app_client
        # Force the lazy singleton to return None
        monkeypatch.setattr(det_mod, "_detector", False)
        resp = c.post("/detect", json={"image_url": "https://example.com/img.jpg"})
        assert resp.status_code == 503

    def test_path_traversal_blocked(self, app_client, monkeypatch):
        c, _, det_mod = app_client
        from hf_space_client import HFDetector
        monkeypatch.setattr(det_mod, "_detector", MagicMock(spec=HFDetector))
        resp = c.post("/detect", json={"image_path": "../../etc/passwd"})
        assert resp.status_code == 400
        assert "traversal" in resp.get_json()["message"].lower()

    def test_file_not_found_returns_404(self, app_client, monkeypatch):
        c, _, det_mod = app_client
        from hf_space_client import HFDetector
        monkeypatch.setattr(det_mod, "_detector", MagicMock(spec=HFDetector))
        resp = c.post("/detect", json={"image_path": "static/no_helmet/nonexistent.jpg"})
        assert resp.status_code == 404

    def test_successful_detect_with_url(self, app_client, monkeypatch):
        c, _, det_mod = app_client
        from hf_space_client import DetectionResult, HFDetector

        mock_result = DetectionResult(
            mode="local", has_violation=True,
            detections=[{"label": "Without Helmet", "confidence": 0.9, "x1": 0, "y1": 0, "x2": 50, "y2": 50}],
            ocr_text="MH12ZZ9999", stats="1 detection",
        )
        mock_det = MagicMock(spec=HFDetector)
        mock_det.detect.return_value = mock_result
        monkeypatch.setattr(det_mod, "_detector", mock_det)

        resp = c.post("/detect", json={"image_url": "https://example.com/img.jpg"})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["has_violation"] is True
        assert body["ocr_text"] == "MH12ZZ9999"
        assert body["mode"] == "local"
        assert len(body["detections"]) == 1

    def test_successful_detect_with_local_file(self, app_client, monkeypatch):
        c, tmp_path, det_mod = app_client
        from hf_space_client import DetectionResult, HFDetector
        from app import config as cfg_mod

        img_path = cfg_mod.BASE_DIR / "static" / "no_helmet" / "snap.jpg"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(b"fakejpeg")

        mock_det = MagicMock(spec=HFDetector)
        mock_det.detect.return_value = DetectionResult(mode="remote", has_violation=False, stats="0 detections")
        monkeypatch.setattr(det_mod, "_detector", mock_det)

        resp = c.post("/detect", json={"image_path": "static/no_helmet/snap.jpg"})
        assert resp.status_code == 200
        assert resp.get_json()["mode"] == "remote"
