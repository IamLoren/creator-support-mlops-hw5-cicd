FROM python:3.14-slim

WORKDIR /app

COPY requirements-inference.txt .

RUN python -m pip install \
    --no-cache-dir \
    -r requirements-inference.txt

COPY inference ./inference

COPY artifacts/model.joblib ./artifacts/model.joblib
COPY artifacts/model_metadata.json ./artifacts/model_metadata.json

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "8000"]