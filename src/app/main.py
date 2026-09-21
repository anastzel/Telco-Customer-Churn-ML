"""FastAPI and Gradio interfaces for the versioned churn model."""

import logging
from contextlib import asynccontextmanager
from typing import Literal

import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.pipeline import CATEGORY_OPTIONS, INPUT_COLUMNS
from src.serving.inference import load_bundle, predict

logger = logging.getLogger(__name__)


class CustomerData(BaseModel):
    gender: Literal["Female", "Male"]
    Partner: Literal["No", "Yes"]
    Dependents: Literal["No", "Yes"]
    PhoneService: Literal["No", "Yes"]
    MultipleLines: Literal["No", "Yes", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No", "Yes", "No internet service"]
    OnlineBackup: Literal["No", "Yes", "No internet service"]
    DeviceProtection: Literal["No", "Yes", "No internet service"]
    TechSupport: Literal["No", "Yes", "No internet service"]
    StreamingTV: Literal["No", "Yes", "No internet service"]
    StreamingMovies: Literal["No", "Yes", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["No", "Yes"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    SeniorCitizen: int = Field(ge=0, le=1)
    tenure: int = Field(ge=0, le=120)
    MonthlyCharges: float = Field(ge=0, le=200)
    TotalCharges: float = Field(ge=0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_bundle()  # Fail startup if the selected model is missing or incomplete.
    yield


app = FastAPI(title="Telco Customer Churn API", version="2.0.0", lifespan=lifespan)


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def get_prediction(data: CustomerData) -> dict[str, str]:
    try:
        return {"prediction": predict(data.model_dump())}
    except Exception as error:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from error


def gradio_interface(*values):
    data = CustomerData.model_validate(dict(zip(INPUT_COLUMNS, values, strict=True)))
    return predict(data.model_dump())


inputs = [gr.Dropdown(options, label=name, value=options[0]) for name, options in CATEGORY_OPTIONS.items()]
inputs.extend([
    gr.Number(label="Senior citizen (0 or 1)", value=0, precision=0, minimum=0, maximum=1),
    gr.Number(label="Tenure (months)", value=1, precision=0, minimum=0, maximum=120),
    gr.Number(label="Monthly charges", value=85.0, minimum=0, maximum=200),
    gr.Number(label="Total charges", value=85.0, minimum=0),
])
demo = gr.Interface(
    fn=gradio_interface,
    inputs=inputs,
    outputs=gr.Textbox(label="Churn prediction"),
    title="Telco Customer Churn Predictor",
    description="Enter customer details to receive a churn classification.",
)
app = gr.mount_gradio_app(app, demo, path="/ui")
