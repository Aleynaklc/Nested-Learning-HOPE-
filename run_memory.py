#!/usr/bin/env python3
"""
Phase 3 — Nested Memory Model.

Trains CNNWithMemory (HOPE-inspired Fast/Mid/Slow memory)
sequentially on 5 CIFAR-10 tasks and compares with baseline.
"""

import torch

from data.dataset import get_continual_tasks
from models.cnn import SimpleCNN
from models.cnn_with_memory import CNNWithMemory
from training.trainer import continual_training
from utils.metrics import compute_forgetting, average_forgetting, average_accuracy
from utils.plots import plot_accuracy, plot_loss, plot_comparison, plot_forgetting_comparison


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    train_loaders, test_loaders = get_continual_tasks(batch_size=64)

    # --- Run 1: Baseline ---
    print("🔴 Running Baseline CNN...")
    baseline_model = SimpleCNN(num_classes=10).to(device)
    baseline_history = continual_training(
        baseline_model,
        train_loaders,
        test_loaders,
        device,
        epochs=10,
        ewc_lambda=0.0,
        per_task_lr_decay=1.0,
    )

    # --- Run 2: Nested Memory ---
    print("\n🟢 Running CNN + Nested Memory...")
    memory_model = CNNWithMemory(num_classes=10).to(device)
    print(f"Memory model parameters: "
          f"{sum(p.numel() for p in memory_model.parameters()):,}")

    memory_history = continual_training(
        memory_model,
        train_loaders,
        test_loaders,
        device,
        epochs=12,
        per_task_lr_decay=0.96,
        ewc_lambda=45.0,
    )

    # --- Compare ---
    print("\n" + "=" * 60)
    print("  📊 COMPARISON — Baseline vs Nested Memory")
    print("=" * 60)

    bl_fgt = compute_forgetting(baseline_history["task_acc"])
    mm_fgt = compute_forgetting(memory_history["task_acc"])

    print(f"\n  Baseline  — Avg Forgetting: {average_forgetting(bl_fgt):.4f}"
          f"  |  Avg Accuracy: {average_accuracy(baseline_history['task_acc']):.4f}")
    print(f"  Memory    — Avg Forgetting: {average_forgetting(mm_fgt):.4f}"
          f"  |  Avg Accuracy: {average_accuracy(memory_history['task_acc']):.4f}")

    # --- Plots ---
    plot_accuracy(baseline_history,
                  title="Baseline CNN — Accuracy Drift",
                  save_name="phase3_baseline_accuracy.png")
    plot_accuracy(memory_history,
                  title="Nested Memory — Accuracy Drift",
                  save_name="phase3_memory_accuracy.png")

    plot_comparison(
        [baseline_history, memory_history],
        ["Baseline CNN", "Nested Memory"],
        task_id=0,
        title="Task 0 Accuracy — Baseline vs Memory",
        save_name="phase3_task0_comparison.png",
    )

    plot_forgetting_comparison(
        [bl_fgt, mm_fgt],
        ["Baseline", "Memory"],
        save_name="phase3_forgetting_comparison.png",
    )

    print("\n✅ Phase 3 complete. Check outputs/ for graphs.")
    return baseline_history, memory_history


if __name__ == "__main__":
    main()
