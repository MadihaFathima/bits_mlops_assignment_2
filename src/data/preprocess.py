"""Preprocess the raw Cats vs Dogs dataset into a resized, split, DVC-trackable form.

Filters corrupt images, subsamples a balanced set, splits into train/val/test,
and writes resized 224x224 RGB JPEGs to data/processed/{split}/{class}/.
"""
import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_SIZE = (224, 224)
CLASSES = {"Cat": "cat", "Dog": "dog"}


def is_valid_image(path: Path) -> bool:
    """Return True if `path` is a readable, non-empty image file."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def list_valid_images(class_dir: Path) -> list[Path]:
    """Return all valid image paths directly inside `class_dir`."""
    return [p for p in sorted(class_dir.iterdir()) if is_valid_image(p)]


def split_paths(
    paths: list[Path], ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42
) -> tuple[list[Path], list[Path], list[Path]]:
    """Shuffle and split `paths` into (train, val, test) lists per the given ratios."""
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def save_resized(src_path: Path, dest_path: Path, size: tuple[int, int] = IMAGE_SIZE) -> None:
    """Resize `src_path` to `size` in RGB and save it at `dest_path`."""
    with Image.open(src_path) as img:
        img = img.convert("RGB").resize(size)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest_path, format="JPEG")


def process_class(
    raw_dir: Path,
    out_dir: Path,
    kaggle_class: str,
    label_name: str,
    n_per_class: int,
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, int]:
    class_dir = raw_dir / kaggle_class
    valid_paths = list_valid_images(class_dir)
    rng = random.Random(seed)
    subsample = rng.sample(valid_paths, min(n_per_class, len(valid_paths)))

    train, val, test = split_paths(subsample, ratios=ratios, seed=seed)
    counts = {}
    for split_name, split_paths_list in (("train", train), ("val", val), ("test", test)):
        for i, src in enumerate(split_paths_list):
            dest = out_dir / split_name / label_name / f"{label_name}_{i:05d}.jpg"
            save_resized(src, dest)
        counts[split_name] = len(split_paths_list)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/PetImages"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--n-per-class", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)

    summary = {}
    for kaggle_class, label_name in CLASSES.items():
        counts = process_class(
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            kaggle_class=kaggle_class,
            label_name=label_name,
            n_per_class=args.n_per_class,
            ratios=(0.8, 0.1, 0.1),
            seed=args.seed,
        )
        summary[label_name] = counts
        print(f"{label_name}: {counts}")

    total = sum(c for counts in summary.values() for c in counts.values())
    print(f"Total processed images: {total}")


if __name__ == "__main__":
    main()
