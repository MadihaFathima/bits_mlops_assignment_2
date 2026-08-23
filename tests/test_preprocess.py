"""Unit tests for data preprocessing functions (src/data/preprocess.py)."""
from pathlib import Path

import pytest
from PIL import Image

from src.data.preprocess import is_valid_image, split_paths


@pytest.fixture
def valid_image_path(tmp_path: Path) -> Path:
    path = tmp_path / "valid.jpg"
    Image.new("RGB", (10, 10), color="red").save(path)
    return path


@pytest.fixture
def corrupt_image_path(tmp_path: Path) -> Path:
    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"not a real image")
    return path


@pytest.fixture
def empty_file_path(tmp_path: Path) -> Path:
    path = tmp_path / "empty.jpg"
    path.touch()
    return path


def test_is_valid_image_accepts_real_image(valid_image_path: Path):
    assert is_valid_image(valid_image_path) is True


def test_is_valid_image_rejects_corrupt_file(corrupt_image_path: Path):
    assert is_valid_image(corrupt_image_path) is False


def test_is_valid_image_rejects_empty_file(empty_file_path: Path):
    assert is_valid_image(empty_file_path) is False


def test_is_valid_image_rejects_missing_file(tmp_path: Path):
    assert is_valid_image(tmp_path / "does_not_exist.jpg") is False


def test_split_paths_respects_ratios():
    paths = [Path(f"img_{i}.jpg") for i in range(100)]
    train, val, test = split_paths(paths, ratios=(0.8, 0.1, 0.1), seed=42)
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10


def test_split_paths_no_overlap_and_covers_all():
    paths = [Path(f"img_{i}.jpg") for i in range(100)]
    train, val, test = split_paths(paths, ratios=(0.8, 0.1, 0.1), seed=42)
    train_set, val_set, test_set = set(train), set(val), set(test)
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert train_set | val_set | test_set == set(paths)


def test_split_paths_is_deterministic_given_seed():
    paths = [Path(f"img_{i}.jpg") for i in range(50)]
    result_a = split_paths(paths, seed=7)
    result_b = split_paths(paths, seed=7)
    assert result_a == result_b


def test_split_paths_different_seeds_can_differ():
    paths = [Path(f"img_{i}.jpg") for i in range(50)]
    result_a = split_paths(paths, seed=1)
    result_b = split_paths(paths, seed=2)
    assert result_a != result_b
