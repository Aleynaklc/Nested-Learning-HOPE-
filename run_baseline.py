#!/usr/bin/env python3
"""
Phase 2 — Baseline CNN + Catastrophic Forgetting Proof.

Trains a SimpleCNN sequentially on 5 CIFAR-10 tasks and
demonstrates catastrophic forgetting through accuracy drift.
"""

import sys
import torch

from data.dataset import get_continual_tasks, validate_splits
from models.cnn import SimpleCNN
from training.trainer import continual_training
from utils.metrics import compute_forgetting, average_forgetting, average_accuracy
from utils.plots import plot_accuracy, plot_loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # --- Phase 1: Validate dataset ---
    print("📦 Phase 1 — Dataset Validation")
    validate_splits()

    # --- Phase 2: Baseline training ---
    print("\n🧠 Phase 2 — Baseline CNN Training")
    train_loaders, test_loaders = get_continual_tasks(batch_size=64)
    model = SimpleCNN(num_classes=10).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = continual_training(
        model,
        train_loaders,
        test_loaders,
        device,
        epochs=3,
        ewc_lambda=0.0,
        per_task_lr_decay=1.0,
    )

    # --- Results ---
    print("\n" + "=" * 50)
    print("  📊 RESULTS — Baseline CNN")
    print("=" * 50)

    forgetting = compute_forgetting(history["task_acc"])
    avg_fgt = average_forgetting(forgetting)
    avg_acc = average_accuracy(history["task_acc"])

    print(f"\n  Per-task forgetting: {forgetting}")
    print(f"  Average forgetting:  {avg_fgt:.4f}")
    print(f"  Average accuracy:    {avg_acc:.4f}")

    # --- Plots ---
    plot_accuracy(history,
                  title="Baseline CNN — Catastrophic Forgetting",
                  save_name="baseline_accuracy.png")
    plot_loss(history,
              title="Baseline CNN — Loss per Task",
              save_name="baseline_loss.png")

    print("\n✅ Phase 2 complete. Check outputs/ for graphs.")
    return history


if __name__ == "__main__":
    history = main()
