"""
NestedMemory — HOPE-inspired multi-time-scale memory module.

Fast / mid / slow linear transforms act as residual paths with different
effective learning rates (set via optimizer param groups in training).
LayerNorm stabilizes features across tasks; learnable scales balance
contributions without hard gradient masking (which hurt continual fit).
"""

import torch
import torch.nn as nn


class NestedMemory(nn.Module):
    """Multi-time-scale associative memory."""

    def __init__(
        self,
        dim: int,
        mid_freq: int = 50,
        slow_freq: int = 500,
        use_soft_schedule: bool = True,
    ):
        super().__init__()

        self.mid_freq = mid_freq
        self.slow_freq = slow_freq
        self.use_soft_schedule = use_soft_schedule

        self.norm = nn.LayerNorm(dim)

        # --- Memory layers ---
        self.fast = nn.Linear(dim, dim)
        self.mid = nn.Linear(dim, dim)
        self.slow = nn.Linear(dim, dim)

        # Slightly smaller init on slower paths for stable long-term retention
        nn.init.xavier_uniform_(self.fast.weight)
        nn.init.zeros_(self.fast.bias)
        nn.init.xavier_uniform_(self.mid.weight, gain=0.8)
        nn.init.zeros_(self.mid.bias)
        nn.init.xavier_uniform_(self.slow.weight, gain=0.6)
        nn.init.zeros_(self.slow.bias)

        # Learnable residual scales (HOPE-style balance)
        self.scale_fast = nn.Parameter(torch.ones(1))
        self.scale_mid = nn.Parameter(torch.ones(1) * 0.85)
        self.scale_slow = nn.Parameter(torch.ones(1) * 0.65)

        # Step counter (DDP-safe via register_buffer)
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))

    def forward(self, x):
        """
        Args:
            x: [batch, dim]  —  flattened encoder features

        Returns:
            x + scaled fast/mid/slow residuals
        """
        h = self.norm(x)

        fast_out = self.fast(h)
        mid_out = self.mid(h)
        slow_out = self.slow(h)

        out = (
            x
            + self.scale_fast * fast_out
            + self.scale_mid * mid_out
            + self.scale_slow * slow_out
        )

        if self.training:
            self.step += 1
            if not self.use_soft_schedule:
                self._apply_update_mask(self.step.item())

        return out

    def _apply_update_mask(self, step: int):
        """Legacy: toggle requires_grad for mid/slow (sparse updates)."""
        for p in self.fast.parameters():
            p.requires_grad = True

        mid_update = (step % self.mid_freq == 0)
        for p in self.mid.parameters():
            p.requires_grad = mid_update

        slow_update = (step % self.slow_freq == 0)
        for p in self.slow.parameters():
            p.requires_grad = slow_update

    def reset_step_counter(self):
        """Reset step counter (e.g. between runs)."""
        self.step.zero_()
