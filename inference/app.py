from contextlib import asynccontextmanager
from pathlib import Path
import os

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


DEFAULT_MODEL_PATH = Path("artifacts/model.joblib")

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(DEFAULT_MODEL_PATH),
    )
)

model = None


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    intent: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found: {MODEL_PATH}. "
            "Run the training step first."
        )

    print(
        f"Loading model from: {MODEL_PATH}"
    )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Model loaded successfully"
    )

    yield


app = FastAPI(
    title="Customer Support Intent Classifier",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "model_loaded": model is not None,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    prediction = model.predict(
        [request.text]
    )

    return {
        "intent": str(
            prediction[0]
        ),
    }