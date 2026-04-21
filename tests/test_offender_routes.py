from pathlib import Path

import offender
import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    data_file = tmp_path / "offenders_data.json"
    static_dir = tmp_path / "static"
    (static_dir / "triples").mkdir(parents=True, exist_ok=True)
    (static_dir / "no_helmet").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(offender, "DATA_FILE", data_file)
    monkeypatch.setattr(offender, "STATIC_DIR", static_dir)

    with offender.app.test_client() as test_client:
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

    update_payload = {
        "category": "triples",
        "folder": "CASE_001",
        "field": "fine_applied",
        "value": "true",
    }
    response = test_client.post("/update_offender", json=update_payload)
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
