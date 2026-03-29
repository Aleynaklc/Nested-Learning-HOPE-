"""
CNNWithMemory — SimpleCNN encoder + NestedMemory + classifier.

Inserts the multi-time-scale memory between the conv encoder
and the fully-connected classifier head.
"""

import torch.nn as nn
import torch.nn.functional as F

from models.nested_memory import NestedMemory


class CNNWithMemory(nn.Module):
    """CNN with HOPE-inspired Nested Memory for continual learning."""

    def __init__(
        self,
        num_classes: int = 10,
        mid_freq: int = 50,
        slow_freq: int = 500,
        use_soft_memory_schedule: bool = True,
    ):
        super().__init__()

        # --- Encoder (same as SimpleCNN) ---
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.flatten_dim = 64 * 8 * 8

        # --- Nested Memory ---
        self.memory = NestedMemory(
            dim=self.flatten_dim,
            mid_freq=mid_freq,
            slow_freq=slow_freq,
            use_soft_schedule=use_soft_memory_schedule,
        )

        # --- Classifier ---
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def encoder_classifier_params(self):
        """Parameters outside NestedMemory (shared backbone + head)."""
        return list(self.conv1.parameters()) + list(self.conv2.parameters()) + list(
            self.fc1.parameters()
        ) + list(self.fc2.parameters())

    def memory_param_groups(self, lr: float, weight_decay: float):
        """Different effective update rates per time scale (HOPE-style)."""
        return [
            {
                "params": list(self.memory.fast.parameters())
                + [self.memory.scale_fast],
                "lr": lr * 1.0,
                "weight_decay": weight_decay,
            },
            {
                "params": list(self.memory.mid.parameters())
                + [self.memory.scale_mid],
                "lr": lr * 0.4,
                "weight_decay": weight_decay,
            },
            {
                "params": list(self.memory.slow.parameters())
                + [self.memory.scale_slow],
                "lr": lr * 0.12,
                "weight_decay": weight_decay,
            },
            {
                "params": list(self.memory.norm.parameters()),
                "lr": lr * 0.5,
                "weight_decay": 0.0,
            },
        ]

    def forward(self, x):
        # Encoder
        x = self.pool(F.relu(self.conv1(x)))   # → [B, 32, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))   # → [B, 64, 8, 8]
        x = x.view(x.size(0), -1)              # → [B, 4096]

        x = self.memory(x)

        # Classifier
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
