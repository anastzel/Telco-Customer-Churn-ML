FROM python:3.11-slim
WORKDIR /app

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY src/ ./src/
ENV PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/src/serving/model/current/churn_bundle.joblib

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
