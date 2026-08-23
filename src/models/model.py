"""Baseline CNN for Cats vs Dogs binary classification."""
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Small CNN: 3 conv+pool blocks, adaptive pooling, then a linear classifier head.

    Outputs a single logit per image (use with BCEWithLogitsLoss); sigmoid(logit) > 0.5
    means "dog" given the class-to-index mapping ImageFolder assigns (cat=0, dog=1).
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224 -> 112
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 -> 56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56 -> 28
            nn.AdaptiveAvgPool2d((7, 7)),  # 28 -> 7, keeps FC layer small
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
