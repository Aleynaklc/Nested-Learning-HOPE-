"""
Elastic Weight Consolidation (EWC) — mitigates catastrophic forgetting
by penalizing deviation from task-specific parameter snapshots weighted
by the Fisher information diagonal.
"""

import torch
import torch.nn as nn


def _fisher_diagonal(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
    max_batches: int | None = 200,
) -> dict[str, torch.Tensor]:
    """Monte Carlo estimate of E[∇log p(y|x,θ)²] over the given loader."""
    model.eval()
    fisher = {
        n: torch.zeros_like(p)
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    n_samples = 0

    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            fisher[name] += p.grad.detach().pow(2)
        n_samples += x.size(0)

    for name in fisher:
        fisher[name] /= max(n_samples, 1)
    return fisher


class EWCAggregate:
    """Stores per-task anchors and Fisher diagonals; builds the EWC penalty."""

    def __init__(self):
        self._anchors: list[dict[str, torch.Tensor]] = []
        self._fishers: list[dict[str, torch.Tensor]] = []

    def __len__(self) -> int:
        return len(self._anchors)

    def consolidate(self, model: nn.Module, fisher: dict[str, torch.Tensor]):
        """Record current parameters and Fisher after finishing a task."""
        anchor = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self._anchors.append(anchor)
        self._fishers.append({k: v.clone() for k, v in fisher.items()})

    def penalty(self, model: nn.Module, device: str) -> torch.Tensor:
        if not self._anchors:
            return torch.tensor(0.0, device=device)
        total = None
        for anchor, fisher in zip(self._anchors, self._fishers, strict=True):
            for n, p in model.named_parameters():
                if not p.requires_grad or n not in anchor or n not in fisher:
                    continue
                diff = p - anchor[n].to(device)
                term = (fisher[n].to(device) * diff.pow(2)).sum()
                total = term if total is None else total + term
        return total if total is not None else torch.tensor(0.0, device=device)


def compute_fisher_for_model(
    model: nn.Module,
    loader,
    device: str,
    criterion: nn.Module | None = None,
    max_batches: int | None = 200,
) -> dict[str, torch.Tensor]:
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    return _fisher_diagonal(model, loader, criterion, device, max_batches)


