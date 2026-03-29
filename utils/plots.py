"""
Visualization utilities for continual learning experiments.

All plots save to the outputs/ directory as PNG files.
"""

import os
import matplotlib
matplotlib.use("Agg")  # headless backend for saving
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs"


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_accuracy(history: dict, title: str = "Accuracy Drift",
                  save_name: str = "accuracy_drift.png"):
    """
    Plot per-task accuracy across training steps.
    Shows catastrophic forgetting as downward drift.
    """
    _ensure_output_dir()
    task_acc = history["task_acc"]
    num_tasks = len(task_acc)

    plt.figure(figsize=(10, 6))

    for task_id in range(num_tasks):
        accs = []
        for t in range(num_tasks):
            if task_id < len(task_acc[t]):
                accs.append(task_acc[t][task_id])
            else:
                accs.append(None)

        # Filter None for plotting
        xs = [i for i, a in enumerate(accs) if a is not None]
        ys = [a for a in accs if a is not None]

        plt.plot(xs, ys, marker="o", linewidth=2,
                 label=f"Task {task_id} (cls {task_id*2}-{task_id*2+1})")

    plt.xlabel("After Training on Task", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.xticks(range(num_tasks))
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📈 Saved: {path}")


def plot_loss(history: dict, title: str = "Training Loss per Task",
              save_name: str = "loss.png"):
    """Plot training loss for each task."""
    _ensure_output_dir()

    plt.figure(figsize=(8, 5))
    plt.plot(history["loss"], marker="s", linewidth=2, color="crimson")
    plt.xlabel("Task", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xticks(range(len(history["loss"])))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📈 Saved: {path}")


def plot_comparison(
    histories: list[dict],
    labels: list[str],
    task_id: int = 0,
    title: str = "Task 0 Accuracy — Method Comparison",
    save_name: str = "comparison.png",
):
    """
    Compare accuracy of a single task across different methods.

    Args:
        histories: list of history dicts from different runs
        labels:    method names
        task_id:   which task's accuracy to track
    """
    _ensure_output_dir()

    plt.figure(figsize=(10, 6))

    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12"]

    for idx, (hist, label) in enumerate(zip(histories, labels)):
        task_acc = hist["task_acc"]
        accs = []
        for t in range(len(task_acc)):
            if task_id < len(task_acc[t]):
                accs.append(task_acc[t][task_id])
            else:
                accs.append(None)

        xs = [i for i, a in enumerate(accs) if a is not None]
        ys = [a for a in accs if a is not None]

        color = colors[idx % len(colors)]
        plt.plot(xs, ys, marker="o", linewidth=2.5,
                 color=color, label=label)

    plt.xlabel("After Training on Task", fontsize=12)
    plt.ylabel(f"Task {task_id} Accuracy", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.xticks(range(max(len(h["task_acc"]) for h in histories)))
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📈 Saved: {path}")


def plot_forgetting_comparison(
    forgetting_scores: list[dict],
    labels: list[str],
    save_name: str = "forgetting_comparison.png",
):
    """Bar chart comparing forgetting scores across methods."""
    _ensure_output_dir()

    import numpy as np

    num_methods = len(labels)
    task_ids = sorted(forgetting_scores[0].keys())
    x = np.arange(len(task_ids))
    width = 0.8 / num_methods

    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12"]

    plt.figure(figsize=(10, 6))

    for i, (fgt, label) in enumerate(zip(forgetting_scores, labels)):
        vals = [fgt.get(tid, 0) for tid in task_ids]
        color = colors[i % len(colors)]
        plt.bar(x + i * width, vals, width, label=label, color=color,
                alpha=0.85)

    plt.xlabel("Task", fontsize=12)
    plt.ylabel("Forgetting Score", fontsize=12)
    plt.title("Forgetting Score Comparison", fontsize=14, fontweight="bold")
    plt.xticks(x + width * (num_methods - 1) / 2,
               [f"Task {t}" for t in task_ids])
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📈 Saved: {path}")
