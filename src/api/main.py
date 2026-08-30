"""FastAPI inference service for the Cats vs Dogs classifier."""
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.api.inference import load_model, predict

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/cnn_baseline.pt"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cats_dogs_api")

REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["method", "path", "status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "Request latency in seconds", ["path"]
)
PREDICTIONS = Counter("predictions_total", "Total predictions by label", ["label"])

model_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_state["model"] = load_model(MODEL_PATH)
    yield
    model_state.clear()


app = FastAPI(title="Cats vs Dogs Classifier", lifespan=lifespan)


@app.middleware("http")
async def log_and_measure_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start

    path = request.url.path
    REQUEST_COUNT.labels(method=request.method, path=path, status=response.status_code).inc()
    REQUEST_LATENCY.labels(path=path).observe(latency)

    # Logs request metadata only (method, path, status, latency) -- never request/response
    # bodies, which would include uploaded image bytes.
    logger.info(
        "%s %s status=%d latency_ms=%.1f",
        request.method,
        path,
        response.status_code,
        latency * 1000,
    )
    return response


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> dict:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read uploaded file as an image")

    result = predict(model_state["model"], image)
    PREDICTIONS.labels(label=result["label"]).inc()
    return result
