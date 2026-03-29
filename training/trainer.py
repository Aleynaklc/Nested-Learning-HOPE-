"""
Standard continual training loop — trains sequentially on tasks
and evaluates all seen tasks after each one.

Supports AdamW, cosine LR per task, optional EWC against catastrophic
forgetting, and HOPE-style param groups for CNNWithMemory.
"""

import torch
import torch.optim as optim

from models.cnn_with_memory import CNNWithMemory
from training.ewc import EWCAggregate, compute_fisher_for_model


def build_optimizer(model, lr: float, weight_decay: float):
    if isinstance(model, CNNWithMemory):
        groups = [
            {
                "params": model.encoder_classifier_params(),
                "lr": lr,
                "weight_decay": weight_decay,
            }
        ]
        groups.extend(model.memory_param_groups(lr, weight_decay))
        return optim.AdamW(groups)
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_one_task(
    model,
    loader,
    optimizer,
    criterion,
    device,
    ewc_agg: EWCAggregate | None = None,
    ewc_lambda: float = 0.0,
    max_grad_norm: float | None = 1.0,
):
    """Train for one epoch on a single task. Returns avg loss."""
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        if ewc_agg is not None and ewc_lambda > 0 and len(ewc_agg) > 0:
            loss = loss + ewc_lambda * ewc_agg.penalty(model, device)
        loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def evaluate(model, loader, device):
    """Evaluate accuracy on a single task. Returns accuracy [0, 1]."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total if total > 0 else 0.0


def continual_training(
    model,
    train_loaders,
    test_loaders,
    device,
    epochs: int = 8,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    ewc_lambda: float = 22.0,
    fisher_max_batches: int = 320,
    per_task_lr_decay: float = 1.0,
):
    """
    Sequential continual training across all tasks.

    Args:
        ewc_lambda: weight for EWC penalty (0 disables consolidation term).
        fisher_max_batches: minibatches used to estimate Fisher after each task.
        per_task_lr_decay: multiply base lr by decay**task_id each task (e.g. 0.94
            stabilizes updates and reduces catastrophic forgetting vs decay=1.0).

    Returns:
        history dict with task_acc and loss.
    """
    criterion = torch.nn.CrossEntropyLoss()
    ewc_agg = EWCAggregate()

    history = {"task_acc": [], "loss": []}

    for task_id, train_loader in enumerate(train_loaders):
        print(f"\n{'='*50}")
        print(f"  TRAINING — Task {task_id}  (classes {task_id*2}, {task_id*2+1})")
        print(f"{'='*50}")

        task_lr = lr * (per_task_lr_decay ** task_id)
        optimizer = build_optimizer(model, task_lr, weight_decay)
        print(f"  Task LR (base): {task_lr:.6f}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=task_lr * 0.02
        )

        for epoch in range(epochs):
            loss = train_one_task(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                ewc_agg=ewc_agg,
                ewc_lambda=ewc_lambda,
            )
            scheduler.step()
            print(f"  Epoch {epoch+1}/{epochs} — Loss: {loss:.4f}")

        history["loss"].append(loss)

        if ewc_lambda > 0:
            fisher = compute_fisher_for_model(
                model,
                train_loader,
                device,
                criterion,
                max_batches=fisher_max_batches,
            )
            ewc_agg.consolidate(model, fisher)

        print(f"\n  📊 Evaluation after Task {task_id}:")
        task_accs = []
        for eval_id in range(task_id + 1):
            acc = evaluate(model, test_loaders[eval_id], device)
            task_accs.append(acc)
            marker = " ← current" if eval_id == task_id else ""
            print(f"    Task {eval_id}: {acc:.4f}{marker}")

        history["task_acc"].append(task_accs)

    return history
