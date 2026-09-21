"""Optional Optuna experiment with a held-out final test set."""

import sys
from pathlib import Path

import optuna
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.pipeline import INPUT_COLUMNS, make_pipeline


def main() -> None:
    raw = pd.read_csv(Path("data/raw/Telco-Customer-Churn.csv"))
    X = raw.loc[:, INPUT_COLUMNS]
    y = raw["Churn"].map({"No": 0, "Yes": 1})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    weight = float((y_train == 0).sum() / (y_train == 1).sum())
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        pipeline = make_pipeline(weight)
        pipeline.named_steps["model"].set_params(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 8),
        )
        return float(cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1
        ).mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    best = make_pipeline(weight)
    best.named_steps["model"].set_params(**study.best_params)
    best.fit(X_train, y_train)
    probabilities = best.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.35).astype(int)
    print("Best cross-validation ROC AUC:", study.best_value)
    print("Untouched test ROC AUC:", roc_auc_score(y_test, probabilities))
    print("Untouched test recall:", recall_score(y_test, predictions))
    print("Untouched test precision:", precision_score(y_test, predictions))
    print("Best parameters:", study.best_params)


if __name__ == "__main__":
    main()
