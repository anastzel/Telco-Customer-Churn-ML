"""Validate the raw data, fit a preprocessing/model pipeline, and log the run."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.pipeline import INPUT_COLUMNS, make_pipeline
from src.utils.validate_data import validate_telco_data

ROOT = Path(__file__).resolve().parents[1]


def train(input_path: Path, output_path: Path, threshold: float = 0.35,
          test_size: float = 0.2, experiment: str = "Telco Churn",
          mlflow_uri: str | None = None) -> dict:
    if not 0 < threshold < 1 or not 0 < test_size < 1:
        raise ValueError("threshold and test_size must be between 0 and 1")
    raw = pd.read_csv(input_path)
    if "Churn" not in raw:
        raise ValueError("Missing target column: Churn")
    mlflow.set_tracking_uri(mlflow_uri or (ROOT / "mlruns").as_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run() as run:
        valid, failed = validate_telco_data(raw)
        mlflow.log_metric("data_quality_pass", int(valid))
        if not valid:
            mlflow.log_text(json.dumps(failed, indent=2), "failed_expectations.json")
            raise ValueError(f"Data validation failed: {failed}")
        y = raw["Churn"].map({"No": 0, "Yes": 1})
        if y.isna().any():
            raise ValueError("Churn must contain only Yes or No")
        X = raw.loc[:, INPUT_COLUMNS]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        weight = float((y_train == 0).sum() / (y_train == 1).sum())
        pipeline = make_pipeline(weight)
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_time = time.perf_counter() - start
        start = time.perf_counter()
        probability = pipeline.predict_proba(X_test)[:, 1]
        prediction_time = time.perf_counter() - start
        prediction = (probability >= threshold).astype(int)
        metrics = {
            "precision": precision_score(y_test, prediction, zero_division=0),
            "recall": recall_score(y_test, prediction, zero_division=0),
            "f1": f1_score(y_test, prediction, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probability),
            "train_time": train_time,
            "pred_time": prediction_time,
        }
        mlflow.log_metrics(metrics)
        mlflow.log_params({
            "model": "xgboost", "threshold": threshold, "test_size": test_size,
            "scale_pos_weight": weight, "random_state": 42,
            **pipeline.named_steps["model"].get_params(),
        })
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
        bundle = {
            "pipeline": pipeline, "threshold": threshold,
            "input_columns": INPUT_COLUMNS, "feature_names": feature_names,
            "run_id": run.info.run_id,
            "data_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "test_metrics": {name: float(value) for name, value in metrics.items()},
            "test_rows": len(X_test),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, output_path)
        mlflow.log_artifact(str(output_path), artifact_path="bundle")
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        mlflow.log_text("\n".join(feature_names), "feature_columns.txt")
        print(f"Run: {run.info.run_id}; bundle: {output_path}")
        print(classification_report(y_test, prediction, digits=3))
        print(json.dumps(metrics, indent=2))
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target", default="Churn", choices=["Churn"])
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/churn_bundle.joblib")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--experiment", default="Telco Churn")
    parser.add_argument("--mlflow_uri")
    args = parser.parse_args()
    train(args.input, args.output, args.threshold, args.test_size, args.experiment, args.mlflow_uri)


if __name__ == "__main__":
    main()
