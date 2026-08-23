"""FastAPI inference service for the Cats vs Dogs classifier."""
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.api.inference import load_model, predict

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/cnn_baseline.pt"))

model_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_state["model"] = load_model(MODEL_PATH)
    yield
    model_state.clear()


app = FastAPI(title="Cats vs Dogs Classifier", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


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

    return predict(model_state["model"], image)
