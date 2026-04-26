"""tests/test_offender_routes.py — Dashboard route tests using the app factory."""
from pathlib import Path

import pytest

import app.data as data_mod
from app import create_app
from app import config as cfg_mod


@pytest.fixture()
def client(tmp_path, monkeypatch):
    data_file  = tmp_path / "offenders_data.json"
    static_dir = tmp_path / "static"
    (static_dir / "triples").mkdir(parents=True, exist_ok=True)
    (static_dir / "no_helmet").mkdir(parents=True, exist_ok=True)

    # Redirect data layer to tmp paths
    monkeypatch.setattr(data_mod, "DATA_FILE",  data_file)
    monkeypatch.setattr(data_mod, "STATIC_DIR", static_dir)
    monkeypatch.setattr(cfg_mod,  "DATA_FILE",  data_file)
    monkeypatch.setattr(cfg_mod,  "STATIC_DIR", static_dir)

    flask_app = create_app()
    flask_app.config["TESTING"] = True

    with flask_app.test_client() as test_client:
        yield test_client, data_file


def test_healthz_returns_ok(client):
    test_client, _ = client
    response = test_client.get("/healthz")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_index_renders_dashboard(client):
    test_client, _ = client
    response = test_client.get("/")
    assert response.status_code == 200
    assert b"Traffic Violation Dashboard" in response.data


def test_update_offender_roundtrip(client):
    test_client, data_file = client

    response = test_client.post(
        "/update_offender",
        json={"category": "triples", "folder": "CASE_001", "field": "fine_applied", "value": "true"},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "success"
    assert body["record"]["fine_applied"] is True

    assert data_file.exists()
    content = data_file.read_text(encoding="utf-8")
    assert "CASE_001" in content
    assert '"fine_applied": true' in content


def test_update_offender_rejects_negative_fine(client):
    test_client, _ = client
    response = test_client.post(
        "/update_offender",
        json={"category": "triples", "folder": "CASE_002", "field": "fine", "value": -5},
    )
    assert response.status_code == 400
    assert response.get_json()["status"] == "error"
