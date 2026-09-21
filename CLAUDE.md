# Project notes

Run commands from the repository root with Python 3.11.

- Train and package a local model: `python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --output src/serving/model/current/churn_bundle.joblib`
- Run tests: `python -m pytest tests/ -q`
- Start the API and Gradio UI: `python -m uvicorn src.app.main:app --host 127.0.0.1 --port 8000`
- Build and run the image: `docker build -t telco-churn:local .` and `docker run --rm -p 8000:8000 telco-churn:local`

The training script validates the raw CSV, splits before fitting, fits the encoder and XGBoost together, and saves a bundle containing the fitted pipeline, threshold, input columns, and provenance. Serving loads one explicit bundle; set `MODEL_PATH` to serve another. The committed `src/serving/model/current/churn_bundle.joblib` is the bundled baseline for Docker and CI.

The Great Expectations tests use synthetic data. The API integration tests use the committed fitted model. The GitHub workflow tests Python dependencies and the API, builds a container, checks a real prediction, and then pushes `latest` and the commit SHA on `main`.

Historical models in `src/serving/model/` remain for comparison. Data files and local MLflow runs are ignored. Never tune on the final test split. The Optuna example uses cross-validation on training data before touching the test set.
