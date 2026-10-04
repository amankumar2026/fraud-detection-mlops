FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY mlruns/ mlruns/
COPY mlflow.db mlflow.db
COPY scripts/rewrite_artifact_paths.py scripts/rewrite_artifact_paths.py

# mlflow.db stores artifact paths from the machine it was trained on
# (Windows absolute paths); point them at the in-image mlruns/ instead.
RUN python scripts/rewrite_artifact_paths.py /app/mlflow.db file:///app/mlruns

ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db
ENV MODEL_NAME=fraud-detector
ENV PREDICTION_THRESHOLD=0.5

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
