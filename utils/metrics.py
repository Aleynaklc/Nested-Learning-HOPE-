"""
Forgetting metrics for continual learning evaluation.
"""

import numpy as np


def compute_forgetting(task_acc_history: list[list[float]]) -> dict:
    """
    Compute per-task forgetting from accuracy history.

    Args:
        task_acc_history: list of lists, where task_acc_history[t]
            contains accuracies on tasks 0..t after training on task t.
            Example:
                [[0.90],
                 [0.55, 0.85],
                 [0.40, 0.60, 0.88]]

    Returns:
        dict mapping task_id → forgetting score
        (max_accuracy - final_accuracy for that task)
    """
    max_acc = {}
    forgetting = {}

    for t, accs in enumerate(task_acc_history):
        for task_id, acc in enumerate(accs):
            if task_id not in max_acc:
                max_acc[task_id] = acc
            else:
                max_acc[task_id] = max(max_acc[task_id], acc)

            forgetting[task_id] = max_acc[task_id] - acc

    return forgetting


def average_forgetting(forgetting: dict) -> float:
    """Compute mean forgetting across all tasks (excluding the last)."""
    if len(forgetting) <= 1:
        return 0.0
    # Exclude the last task (it was never "forgotten")
    values = [v for k, v in forgetting.items() if k < max(forgetting.keys())]
    return np.mean(values) if values else 0.0


def average_accuracy(task_acc_history: list[list[float]]) -> float:
    """Average accuracy across all tasks after final training step."""
    if not task_acc_history:
        return 0.0
    final_accs = task_acc_history[-1]
    return np.mean(final_accs)
