"""
tests/test_speed_estimator.py
==============================
Unit tests for the optional Vehicle Speed Estimation feature.

All remote calls are mocked — no network or heavy ML deps required.
The Space being SLEEPING / unavailable is the *normal* path on CI.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# _parse_speed_stats
# ---------------------------------------------------------------------------

SAMPLE_STATS = """
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

✅ **Processing completed successfully!**
"""


class TestParseSpeedStats:
    def test_parses_total_count(self):
        from hf_space_client import _parse_speed_stats
        p = _parse_speed_stats(SAMPLE_STATS)
        assert p["total_count"] == 12

    def test_parses_in_out_counts(self):
        from hf_space_client import _parse_speed_stats
        p = _parse_speed_stats(SAMPLE_STATS)
        assert p["in_count"] == 8
        assert p["out_count"] == 4

    def test_parses_avg_speed(self):
        from hf_space_client import _parse_speed_stats
        p = _parse_speed_stats(SAMPLE_STATS)
        assert p["avg_speed_kmh"] == pytest.approx(45.3, rel=1e-3)

    def test_parses_max_min_speed(self):
        from hf_space_client import _parse_speed_stats
        p = _parse_speed_stats(SAMPLE_STATS)
        assert p["max_speed_kmh"] == pytest.approx(72.1, rel=1e-3)
        assert p["min_speed_kmh"] == pytest.approx(18.6, rel=1e-3)

    def test_parses_frames_and_time(self):
        from hf_space_client import _parse_speed_stats
        p = _parse_speed_stats(SAMPLE_STATS)
        assert p["frames_processed"] == 900
        assert p["processing_time_s"] == pytest.approx(12.34, rel=1e-3)

    def test_parses_model_used(self):
        from hf_space_client import _parse_speed_stats
        p = _parse_speed_stats(SAMPLE_STATS)
        assert p["model_used"] == "yolov8n"

    def test_empty_string_returns_zeros(self):
        from hf_space_client import _parse_speed_stats
        p = _parse_speed_stats("")
        assert p["total_count"] == 0
        assert p["avg_speed_kmh"] == 0.0
        assert p["model_used"] == ""


# ---------------------------------------------------------------------------
# SpeedEstimationResult
# ---------------------------------------------------------------------------

class TestSpeedEstimationResult:
    def test_default_not_available(self):
        from hf_space_client import SpeedEstimationResult
        r = SpeedEstimationResult()
        assert r.available is False
        assert r.mode == "unavailable"
        assert r.total_count == 0
        assert r.error is None

    def test_can_set_all_fields(self):
        from hf_space_client import SpeedEstimationResult
        r = SpeedEstimationResult(
            available=True,
            mode="remote",
            total_count=10,
            avg_speed_kmh=55.5,
            output_video_url="https://hf.space/out.mp4",
        )
        assert r.available is True
        assert r.total_count == 10
        assert r.avg_speed_kmh == pytest.approx(55.5)


# ---------------------------------------------------------------------------
# SpeedEstimator — no token (most common CI/CD scenario)
# ---------------------------------------------------------------------------

class TestSpeedEstimatorNoToken:
    def test_disabled_when_no_token(self):
        from hf_space_client import SpeedEstimator
        se = SpeedEstimator(hf_token="")
        assert se.enabled is False

    def test_estimate_returns_unavailable_when_no_token(self, tmp_path):
        from hf_space_client import SpeedEstimator
        se = SpeedEstimator(hf_token="")
        # create a dummy file so the file-existence check doesn't interfere
        fake_video = tmp_path / "traffic.mp4"
        fake_video.write_bytes(b"fake")
        result = se.estimate(str(fake_video))
        assert result.available is False
        assert result.error is not None

    def test_estimate_returns_unavailable_for_missing_file(self):
        from hf_space_client import SpeedEstimator
        se = SpeedEstimator(hf_token="")
        result = se.estimate("/nonexistent/video.mp4")
        assert result.available is False


# ---------------------------------------------------------------------------
# SpeedEstimator — token present, remote succeeds
# ---------------------------------------------------------------------------

class TestSpeedEstimatorRemoteSuccess:
    def test_returns_available_true_on_success(self, tmp_path):
        import sys
        from hf_space_client import SpeedEstimator

        fake_video = tmp_path / "traffic.mp4"
        fake_video.write_bytes(b"fakevideo")

        se = SpeedEstimator(hf_token="hf_faketoken")
        mock_client = MagicMock()
        mock_client.predict.return_value = (
            {"path": str(tmp_path / "out.mp4"), "url": "https://hf.space/out.mp4"},
            SAMPLE_STATS,
        )
        se._client = mock_client

        # Fake gradio_client presence so the enabled property returns True
        fake_gc = MagicMock()
        with patch.dict(sys.modules, {"gradio_client": fake_gc}):
            result = se.estimate(str(fake_video))

        assert result.available is True
        assert result.mode == "remote"
        assert result.total_count == 12
        assert result.avg_speed_kmh == pytest.approx(45.3, rel=1e-3)
        assert result.output_video_url == "https://hf.space/out.mp4"
        assert result.error is None

    def test_returns_unavailable_on_network_error(self, tmp_path):
        import sys
        from hf_space_client import SpeedEstimator

        fake_video = tmp_path / "traffic.mp4"
        fake_video.write_bytes(b"fakevideo")

        se = SpeedEstimator(hf_token="hf_faketoken")
        mock_client = MagicMock()
        mock_client.predict.side_effect = ConnectionError("Space still sleeping")
        se._client = mock_client

        fake_gc = MagicMock()
        with patch.dict(sys.modules, {"gradio_client": fake_gc}):
            result = se.estimate(str(fake_video))

        assert result.available is False
        assert "sleeping" in (result.error or "")


# ---------------------------------------------------------------------------
# /speed_estimate Flask route (via app factory)
# ---------------------------------------------------------------------------

@pytest.fixture()
def app_client_speed(tmp_path, monkeypatch):
    import app.data as data_mod
    from app import config as cfg_mod, create_app

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
        yield c


class TestSpeedEstimateRoute:
    def test_missing_video_path_returns_400(self, app_client_speed):
        resp = app_client_speed.post("/speed_estimate", json={})
        assert resp.status_code == 400
        assert resp.get_json()["available"] is False

    def test_returns_503_when_estimator_none(self, app_client_speed, monkeypatch):
        from app.blueprints import speed as speed_mod
        monkeypatch.setattr(speed_mod, "_estimator", False)
        resp = app_client_speed.post("/speed_estimate", json={"video_path": "v.mp4"})
        assert resp.status_code == 503

    def test_path_traversal_blocked(self, app_client_speed):
        resp = app_client_speed.post("/speed_estimate", json={"video_path": "../../etc/passwd"})
        assert resp.status_code == 400
        assert "traversal" in resp.get_json()["error"].lower()

    def test_unavailable_returns_200_with_available_false(self, app_client_speed, monkeypatch):
        from app.blueprints import speed as speed_mod
        from hf_space_client import SpeedEstimationResult, SpeedEstimator

        mock_est = MagicMock(spec=SpeedEstimator)
        mock_est.estimate.return_value = SpeedEstimationResult(
            available=False, error="Space is sleeping"
        )
        monkeypatch.setattr(speed_mod, "_estimator", mock_est)

        resp = app_client_speed.post("/speed_estimate", json={"video_path": "static/v.mp4"})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["available"] is False
        assert "sleeping" in body["error"]

    def test_successful_response_shape(self, app_client_speed, monkeypatch):
        from app.blueprints import speed as speed_mod
        from hf_space_client import SpeedEstimationResult, SpeedEstimator

        mock_result = SpeedEstimationResult(
            available=True, mode="remote",
            total_count=15, avg_speed_kmh=52.0, max_speed_kmh=80.0, min_speed_kmh=20.0,
            in_count=10, out_count=5, frames_processed=1800, processing_time_s=30.0,
            model_used="yolov8n", output_video_url="https://hf.space/out.mp4", stats_raw=SAMPLE_STATS,
        )
        mock_est = MagicMock(spec=SpeedEstimator)
        mock_est.estimate.return_value = mock_result
        monkeypatch.setattr(speed_mod, "_estimator", mock_est)

        resp = app_client_speed.post(
            "/speed_estimate",
            json={"video_path": "static/traffic.mp4", "model_choice": "yolov8s",
                  "line_position": 320, "confidence_threshold": 0.5},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["available"] is True
        assert body["total_count"] == 15
        assert body["avg_speed_kmh"] == pytest.approx(52.0)
        assert body["output_video_url"] == "https://hf.space/out.mp4"
        assert body["model_used"] == "yolov8n"

        kw = mock_est.estimate.call_args.kwargs
        assert kw["model_choice"] == "yolov8s"
        assert kw["line_position"] == 320
        assert kw["confidence_threshold"] == pytest.approx(0.5)

