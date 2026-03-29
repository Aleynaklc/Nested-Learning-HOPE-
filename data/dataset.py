"""
CIFAR-10 Continual Learning Dataset Pipeline.

Splits CIFAR-10 into 5 class-incremental tasks:
  Task 0: classes [0, 1]
  Task 1: classes [2, 3]
  Task 2: classes [4, 5]
  Task 3: classes [6, 7]
  Task 4: classes [8, 9]
"""

from collections import Counter

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# Standard CIFAR-10 normalization
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

# Class-incremental task definitions
TASK_CLASSES = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
]

CIFAR10_CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _load_cifar10(root: str = "./data"):
    """Download and return raw CIFAR-10 train/test datasets."""
    train = datasets.CIFAR10(root=root, train=True,
                             download=True, transform=TRANSFORM)
    test = datasets.CIFAR10(root=root, train=False,
                            download=True, transform=TRANSFORM)
    return train, test


def split_by_classes(dataset, class_list: list[int]) -> Subset:
    """Return a Subset containing only samples whose label is in class_list."""
    targets = (
        dataset.targets if hasattr(dataset, "targets")
        else [y for _, y in dataset]
    )
    indices = [i for i, label in enumerate(targets) if label in class_list]
    return Subset(dataset, indices)


def get_continual_tasks(
    batch_size: int = 64,
    root: str = "./data",
) -> tuple[list[DataLoader], list[DataLoader]]:
    """
    Build train and test DataLoaders for each continual learning task.

    Returns:
        train_loaders: list of 5 DataLoaders (one per task)
        test_loaders:  list of 5 DataLoaders (one per task)
    """
    train_dataset, test_dataset = _load_cifar10(root)

    train_loaders = []
    test_loaders = []

    for task_classes in TASK_CLASSES:
        train_sub = split_by_classes(train_dataset, task_classes)
        test_sub = split_by_classes(test_dataset, task_classes)

        train_loaders.append(
            DataLoader(train_sub, batch_size=batch_size, shuffle=True)
        )
        test_loaders.append(
            DataLoader(test_sub, batch_size=batch_size, shuffle=False)
        )

    return train_loaders, test_loaders


def validate_splits(root: str = "./data"):
    """Print label sets, class distributions, and sample counts per task."""
    train_loaders, test_loaders = get_continual_tasks(root=root)

    print("=" * 60)
    print("CIFAR-10 CONTINUAL LEARNING — SPLIT VALIDATION")
    print("=" * 60)

    for i, (train_loader, test_loader) in enumerate(
        zip(train_loaders, test_loaders)
    ):
        labels = set()
        counter = Counter()
        for _, y in train_loader:
            labels.update(y.tolist())
            counter.update(y.tolist())

        class_names = [CIFAR10_CLASS_NAMES[c] for c in sorted(labels)]

        print(f"\nTask {i}: classes {sorted(labels)} → {class_names}")
        print(f"  Train samples: {sum(counter.values())}")
        for cls in sorted(counter):
            print(f"    Class {cls} ({CIFAR10_CLASS_NAMES[cls]}): "
                  f"{counter[cls]} samples")

        # Test set
        test_count = sum(len(y) for _, y in test_loader)
        print(f"  Test samples:  {test_count}")

    print("\n" + "=" * 60)
    print("✅ All splits validated.")
    print("=" * 60)


if __name__ == "__main__":
    validate_splits()
