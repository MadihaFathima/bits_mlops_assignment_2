"""Model loading and prediction logic for the inference API, kept separate from the
FastAPI layer so it can be unit tested without running a server.
"""
from pathlib import Path

import torch
from PIL import Image

from src.data.transforms import CLASS_NAMES, get_eval_transform
from src.models.model import SimpleCNN


def load_model(model_path: Path) -> SimpleCNN:
    """Load the trained SimpleCNN weights from `model_path` in eval mode, on CPU."""
    model = SimpleCNN()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(model: SimpleCNN, image: Image.Image) -> dict:
    """Run inference on a single PIL image. Returns label + class probabilities."""
    transform = get_eval_transform()
    tensor = transform(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        logit = model(tensor)
        dog_prob = torch.sigmoid(logit).item()

    cat_prob = 1.0 - dog_prob
    label = CLASS_NAMES[1] if dog_prob > 0.5 else CLASS_NAMES[0]

    return {
        "label": label,
        "probability": dog_prob if label == "dog" else cat_prob,
        "class_probabilities": {"cat": cat_prob, "dog": dog_prob},
    }
