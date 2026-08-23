"""Train the baseline CNN on the preprocessed Cats vs Dogs dataset, tracked via MLflow."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.model import SimpleCNN

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def build_dataloaders(data_dir: Path, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_transform)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_ds.classes


def run_epoch(model, loader, criterion, optimizer=None, device="cpu") -> tuple[float, float]:
    """Run one pass over `loader`. Trains if `optimizer` is given, else evaluates.

    Returns (avg_loss, accuracy).
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            if is_train:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def evaluate_predictions(model, loader, device="cpu") -> tuple[list[int], list[int]]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).long().squeeze(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    return all_preds, all_labels


def plot_loss_curves(train_losses: list[float], val_losses: list[float], out_path: Path) -> None:
    plt.figure()
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(cm, class_names: list[str], out_path: Path) -> None:
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix - Test Set")
    plt.colorbar()
    plt.xticks([0, 1], class_names)
    plt.yticks([0, 1], class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-out", type=Path, default=Path("models/cnn_baseline.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--experiment-name", type=str, default="cats-vs-dogs-classification")
    args = parser.parse_args()

    device = torch.device("cpu")
    train_loader, val_loader, test_loader, class_names = build_dataloaders(args.data_dir, args.batch_size)

    model = SimpleCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "architecture": "SimpleCNN",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "optimizer": "Adam",
                "train_size": len(train_loader.dataset),
                "val_size": len(val_loader.dataset),
                "test_size": len(test_loader.dataset),
            }
        )

        train_losses, val_losses = [], []
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                step=epoch,
            )
            print(
                f"Epoch {epoch}/{args.epochs} - "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        preds, labels = evaluate_predictions(model, test_loader, device)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        test_acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
        cm = confusion_matrix(labels, preds)

        mlflow.log_metrics(
            {
                "test_accuracy": test_acc,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1,
            }
        )

        loss_curve_path = Path("outputs/loss_curve.png")
        cm_path = Path("outputs/confusion_matrix.png")
        plot_loss_curves(train_losses, val_losses, loss_curve_path)
        plot_confusion_matrix(cm, class_names, cm_path)
        mlflow.log_artifact(str(loss_curve_path))
        mlflow.log_artifact(str(cm_path))

        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.model_out)
        mlflow.log_artifact(str(args.model_out))
        example_input, _ = next(iter(test_loader))
        mlflow.pytorch.log_model(model, name="model", input_example=example_input[:1].numpy())

        metadata = {
            "run_id": mlflow.active_run().info.run_id,
            "class_to_idx": {name: idx for idx, name in enumerate(class_names)},
            "test_accuracy": test_acc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
        }
        metadata_path = Path("models/model_metadata.json")
        metadata_path.write_text(json.dumps(metadata, indent=2))
        mlflow.log_artifact(str(metadata_path))

        print(f"Test accuracy={test_acc:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
        print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
