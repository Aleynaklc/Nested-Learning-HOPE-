#!/usr/bin/env python3
"""
Phase 4 — Full comparison: Baseline vs Nested Memory vs Memory + Replay.

Trains all three configurations and generates comprehensive comparison
graphs and forgetting metrics.
"""

import torch

from data.dataset import get_continual_tasks
from models.cnn import SimpleCNN
from models.cnn_with_memory import CNNWithMemory
from training.trainer import continual_training
from training.replay_trainer import continual_training_with_replay
from utils.metrics import compute_forgetting, average_forgetting, average_accuracy
from utils.plots import (
    plot_accuracy,
    plot_comparison,
    plot_forgetting_comparison,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    train_loaders, test_loaders = get_continual_tasks(batch_size=64)

    # ===== RUN 1: Baseline CNN =====
    print("🔴 [1/3] Baseline CNN...")
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

    # ===== RUN 2: CNN + Nested Memory (no replay) =====
    print("\n🟡 [2/3] CNN + Nested Memory...")
    memory_model = CNNWithMemory(num_classes=10).to(device)
    memory_history = continual_training(
        memory_model,
        train_loaders,
        test_loaders,
        device,
        epochs=12,
        per_task_lr_decay=0.96,
        ewc_lambda=45.0,
    )

    # ===== RUN 3: CNN + Memory + Replay (full rehearsal → forgetting ≈ 0) =====
    print("\n🟢 [3/3] CNN + Nested Memory + Full Replay + Consolidation...")
    replay_model = CNNWithMemory(num_classes=10).to(device)
    replay_history = continual_training_with_replay(
        replay_model,
        train_loaders,
        test_loaders,
        device,
        epochs=12,
        buffer_capacity=50000,
        samples_per_class=5000,
        replay_ratio=3.0,
        consolidation_epochs=1,
        ewc_lambda=25.0,
    )

    # ===== RESULTS =====
    print("\n" + "=" * 70)
    print("  📊 FINAL COMPARISON — All Methods")
    print("=" * 70)

    bl_fgt = compute_forgetting(baseline_history["task_acc"])
    mm_fgt = compute_forgetting(memory_history["task_acc"])
    rp_fgt = compute_forgetting(replay_history["task_acc"])

    methods = [
        ("Baseline CNN", baseline_history, bl_fgt),
        ("Nested Memory", memory_history, mm_fgt),
        ("Memory + Replay", replay_history, rp_fgt),
    ]

    for name, hist, fgt in methods:
        print(f"\n  {name}:")
        print(f"    Avg Forgetting: {average_forgetting(fgt):.4f}")
        print(f"    Avg Accuracy:   {average_accuracy(hist['task_acc']):.4f}")
        print(f"    Per-task fgt:   {fgt}")

    # ===== PLOTS =====
    # Individual accuracy
    for hist, name, fname in [
        (baseline_history, "Baseline CNN", "phase4_baseline_acc.png"),
        (memory_history, "Nested Memory", "phase4_memory_acc.png"),
        (replay_history, "Memory + Replay", "phase4_replay_acc.png"),
    ]:
        plot_accuracy(hist, title=f"{name} — Accuracy Drift",
                      save_name=fname)

    # Task 0 comparison (most telling for forgetting)
    plot_comparison(
        [baseline_history, memory_history, replay_history],
        ["Baseline CNN", "Nested Memory", "Memory + Replay"],
        task_id=0,
        title="Task 0 Accuracy — All Methods",
        save_name="phase4_task0_comparison.png",
    )

    # Forgetting bar chart
    plot_forgetting_comparison(
        [bl_fgt, mm_fgt, rp_fgt],
        ["Baseline", "Memory", "Memory + Replay"],
        save_name="phase4_forgetting_comparison.png",
    )

    print("\n✅ Phase 4 complete. Check outputs/ for graphs.")
    print("   Full replay (5000/class) + stratified rehearsal targets ~0 forgetting.")


if __name__ == "__main__":
    main()
