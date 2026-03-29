"""
SimpleCNN Baseline — Encoder + Classifier for CIFAR-10.

Architecture:
    Conv(3→32, 3×3) → ReLU → Pool(2)
    Conv(32→64, 3×3) → ReLU → Pool(2)
    FC(4096 → 256) → ReLU
    FC(256 → num_classes)
"""

import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Baseline CNN for continual learning forgetting experiments."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # --- Encoder ---
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.flatten_dim = 64 * 8 * 8  # after 2× pool on 32×32

        # --- Classifier ---
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Encoder
        x = self.pool(F.relu(self.conv1(x)))   # → [B, 32, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))   # → [B, 64, 8, 8]

        x = x.view(x.size(0), -1)              # → [B, 4096]

        # Classifier
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
