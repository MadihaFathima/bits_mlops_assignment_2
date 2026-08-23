"""Unit tests for the model inference utility (src/api/inference.py)."""
from pathlib import Path

import pytest
from PIL import Image

from src.api.inference import load_model, predict

MODEL_PATH = Path("models/cnn_baseline.pt")


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip(f"Trained model not found at {MODEL_PATH}; run training first.")
    return load_model(MODEL_PATH)


def test_predict_returns_expected_structure(model):
    image = Image.new("RGB", (300, 300), color=(120, 80, 40))
    result = predict(model, image)

    assert set(result.keys()) == {"label", "probability", "class_probabilities"}
    assert result["label"] in {"cat", "dog"}
    assert 0.0 <= result["probability"] <= 1.0
    assert set(result["class_probabilities"].keys()) == {"cat", "dog"}


def test_predict_class_probabilities_sum_to_one(model):
    image = Image.new("RGB", (224, 224), color=(200, 200, 200))
    result = predict(model, image)
    total = result["class_probabilities"]["cat"] + result["class_probabilities"]["dog"]
    assert total == pytest.approx(1.0, abs=1e-5)


def test_predict_handles_non_square_and_non_224_images(model):
    image = Image.new("RGB", (500, 100), color=(50, 150, 250))
    result = predict(model, image)
    assert result["label"] in {"cat", "dog"}


def test_predict_handles_grayscale_input(model):
    image = Image.new("L", (224, 224), color=128)
    result = predict(model, image)
    assert result["label"] in {"cat", "dog"}
