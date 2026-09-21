"""Exercise the real web stack without depending on saved model locations."""

import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def web_app(monkeypatch):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    inference = ModuleType("src.serving.inference")
    inference.predict = lambda data: "Not likely to churn"
    monkeypatch.setitem(sys.modules, "src.serving.inference", inference)
    path = Path(__file__).resolve().parents[1] / "src/app/main.py"
    namespace = runpy.run_path(str(path))
    with TestClient(namespace["app"]) as client:
        yield client, namespace


def test_fastapi_and_gradio_routes(web_app):
    client, _ = web_app
    assert client.get("/").json() == {"status": "ok"}
    for path in ("/docs", "/openapi.json", "/ui/", "/ui/config", "/ui/gradio_api/info"):
        response = client.get(path)
        assert response.status_code == 200, (path, response.text)


def test_prediction_request_validation_and_form_callback(web_app):
    client, namespace = web_app
    sample = dict(zip(namespace["CustomerData"].model_fields, namespace["demo"].examples[0]))
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    assert response.json() == {"prediction": "Not likely to churn"}
    assert client.post("/predict", json={}).status_code == 422
    assert namespace["gradio_interface"](*sample.values()) == "Not likely to churn"
