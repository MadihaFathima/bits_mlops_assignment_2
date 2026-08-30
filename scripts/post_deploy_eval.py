"""Post-deployment model performance tracking (M5 Task 2).

Sends a small batch of real requests -- test images with known true labels -- to the
already-deployed inference service's /predict endpoint, and reports accuracy against
those ground-truth labels. This checks the model's real, currently-serving behavior
post-deployment, independent of the offline test-set metrics captured during training.
"""
import argparse
import json
import random
from pathlib import Path

import requests

CLASSES = ["cat", "dog"]


def collect_sample_paths(test_dir: Path, n_per_class: int, seed: int) -> list[tuple[Path, str]]:
    """Return up to n_per_class (path, true_label) pairs per class from test_dir."""
    samples = []
    rng = random.Random(seed)
    for label in CLASSES:
        class_dir = test_dir / label
        paths = sorted(class_dir.iterdir())
        chosen = rng.sample(paths, min(n_per_class, len(paths)))
        samples.extend((p, label) for p in chosen)
    rng.shuffle(samples)
    return samples


def call_predict(base_url: str, image_path: Path) -> dict:
    with open(image_path, "rb") as f:
        response = requests.post(
            f"{base_url}/predict",
            files={"file": (image_path.name, f, "image/jpeg")},
            timeout=30,
        )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--test-dir", type=Path, default=Path("data/processed/test"))
    parser.add_argument("--n-per-class", type=int, default=15)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=Path, default=Path("outputs/post_deploy_eval.json"))
    args = parser.parse_args()

    samples = collect_sample_paths(args.test_dir, args.n_per_class, args.seed)
    print(f"Sending {len(samples)} real requests to {args.base_url}/predict ...")

    results = []
    correct = 0
    for path, true_label in samples:
        result = call_predict(args.base_url, path)
        is_correct = result["label"] == true_label
        correct += is_correct
        results.append(
            {
                "file": path.name,
                "true_label": true_label,
                "predicted_label": result["label"],
                "probability": result["probability"],
                "correct": is_correct,
            }
        )
        print(f"  {path.name}: true={true_label} pred={result['label']} correct={is_correct}")

    accuracy = correct / len(samples)
    summary = {
        "base_url": args.base_url,
        "n_requests": len(samples),
        "accuracy": accuracy,
        "results": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    print(f"\nPost-deployment accuracy: {accuracy:.4f} ({correct}/{len(samples)})")
    print(f"Full report written to {args.out}")


if __name__ == "__main__":
    main()
