"""Load one versioned training bundle and serve its fitted pipeline."""

import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "model/current/churn_bundle.joblib"


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    path = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    if not path.is_file():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    bundle = joblib.load(path)
    if not {"pipeline", "threshold", "input_columns"} <= bundle.keys():
        raise ValueError(f"Incomplete model bundle: {path}")
    return bundle


def predict(input_dict: dict) -> str:
    bundle = load_bundle()
    data = pd.DataFrame([input_dict], columns=bundle["input_columns"])
    probability = float(bundle["pipeline"].predict_proba(data)[0, 1])
    return "Likely to churn" if probability >= bundle["threshold"] else "Not likely to churn"
