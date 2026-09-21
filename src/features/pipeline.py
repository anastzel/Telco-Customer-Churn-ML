"""Fitted preprocessing shared by training and inference."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from xgboost import XGBClassifier

CATEGORY_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Electronic check", "Mailed check", "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}
CATEGORICAL_COLUMNS = list(CATEGORY_OPTIONS)
NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
INPUT_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def normalize_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert raw numeric fields without learning from the dataset."""
    frame = frame.loc[:, INPUT_COLUMNS].copy()
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def make_pipeline(scale_pos_weight: float) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            ("categories", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), CATEGORICAL_COLUMNS),
            ("numbers", SimpleImputer(strategy="median"), NUMERIC_COLUMNS),
        ],
        verbose_feature_names_out=True,
    )
    model = XGBClassifier(
        n_estimators=301,
        learning_rate=0.034,
        max_depth=7,
        subsample=0.95,
        colsample_bytree=0.98,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
        eval_metric="logloss",
    )
    return Pipeline([
        ("normalize", FunctionTransformer(normalize_inputs, validate=False)),
        ("preprocess", preprocessing),
        ("model", model),
    ])
