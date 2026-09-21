"""Test API and UI against the packaged fitted model."""

import pandas as pd
from fastapi.testclient import TestClient

from src.app.main import app, gradio_interface
from src.features.pipeline import INPUT_COLUMNS
from src.serving.inference import load_bundle


def sample_customer() -> dict:
    return {
        "gender": "Female", "Partner": "No", "Dependents": "No",
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "SeniorCitizen": 0, "tenure": 1,
        "MonthlyCharges": 85.0, "TotalCharges": 85.0,
    }


def test_routes_and_real_prediction():
    with TestClient(app) as client:
        assert client.get("/").json() == {"status": "ok"}
        for path in ("/docs", "/openapi.json", "/ui/", "/ui/config"):
            assert client.get(path).status_code == 200
        response = client.post("/predict", json=sample_customer())
        assert response.status_code == 200, response.text
        assert response.json()["prediction"] in {"Likely to churn", "Not likely to churn"}


def test_invalid_request_is_rejected():
    with TestClient(app) as client:
        data = sample_customer()
        data["SeniorCitizen"] = 9
        assert client.post("/predict", json=data).status_code == 422
        data = sample_customer()
        data["Contract"] = "unknown"
        assert client.post("/predict", json=data).status_code == 422


def test_api_uses_saved_threshold_and_fitted_pipeline():
    data = sample_customer()
    bundle = load_bundle()
    probability = bundle["pipeline"].predict_proba(
        pd.DataFrame([data])
    )[0, 1]
    expected = "Likely to churn" if probability >= bundle["threshold"] else "Not likely to churn"
    with TestClient(app) as client:
        assert client.post("/predict", json=data).json() == {"prediction": expected}
    assert "SeniorCitizen" in bundle["input_columns"]


def test_gradio_and_api_use_the_same_bundle():
    data = sample_customer()
    ordered_values = [data[column] for column in INPUT_COLUMNS]
    with TestClient(app) as client:
        expected = client.post("/predict", json=data).json()["prediction"]
    assert gradio_interface(*ordered_values) == expected


def test_single_customer_keeps_categorical_signal():
    bundle = load_bundle()
    data = sample_customer()
    columns = bundle["feature_names"]
    vector = bundle["pipeline"][:-1].transform(pd.DataFrame([data]))[0]
    assert vector[columns.index("categories__InternetService_Fiber optic")] == 1
    assert vector[columns.index("categories__PaymentMethod_Electronic check")] == 1
    assert bundle["test_rows"] == 1409
    assert "roc_auc" in bundle["test_metrics"]
