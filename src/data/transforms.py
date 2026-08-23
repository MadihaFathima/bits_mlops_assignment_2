"""Image transforms shared between training and inference, to avoid train/serve skew."""
from torchvision import transforms

IMAGE_SIZE = (224, 224)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = ["cat", "dog"]  # index order matches ImageFolder's alphabetical mapping


def get_train_transform() -> transforms.Compose:
    """Transform for training: augmentation + normalization. Input must already be 224x224."""
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )


def get_eval_transform() -> transforms.Compose:
    """Transform for validation/test/inference: resize + normalize, no augmentation.

    Includes a Resize step (unlike training, which relies on already-224x224 processed
    files) since inference receives arbitrary-sized images from API callers.
    """
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )
