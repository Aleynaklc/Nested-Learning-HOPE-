"""
Replay-integrated continual training loop.

Only original batch samples are added to the buffer,
not replayed samples — this prevents distribution collapse.
"""

import torch

from training.trainer import build_optimizer, evaluate
from training.ewc import EWCAggregate, compute_fisher_for_model
from utils.replay_buffer import ReplayBuffer


def train_with_replay(
    model,
    loader,
    optimizer,
    criterion,
    buffer,
    device,
    ewc_agg: EWCAggregate | None = None,
    ewc_lambda: float = 0.0,
    replay_ratio: float = 3.0,
    max_grad_norm: float | None = 1.0,
    stratified_replay: bool = True,
):
    """
    Train one epoch on a task, mixing each batch with replay samples.

    stratified_replay: equal samples per class in the replay portion when possible
    (best for minimizing forgetting).
    """
    model.train()
    total_loss = 0.0

    for x, y in loader:
        buffer.add_batch(x.clone(), y.clone())

        x, y = x.to(device), y.to(device)

        bs = x.size(0)
        replay_n = max(1, int(bs * replay_ratio))
        if stratified_replay:
            replay_x, replay_y = buffer.sample_stratified(replay_n)
        else:
            replay_x, replay_y = buffer.sample(replay_n)

        if replay_x is not None:
            replay_x = replay_x.to(device)
            replay_y = replay_y.to(device)
            x_combined = torch.cat([x, replay_x], dim=0)
            y_combined = torch.cat([y, replay_y], dim=0)
        else:
            x_combined = x
            y_combined = y

        optimizer.zero_grad()
        out = model(x_combined)
        loss = criterion(out, y_combined)
        if ewc_agg is not None and ewc_lambda > 0 and len(ewc_agg) > 0:
            loss = loss + ewc_lambda * ewc_agg.penalty(model, device)
        loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def run_buffer_consolidation(
    model,
    buffer: ReplayBuffer,
    optimizer,
    criterion,
    device,
    batch_size: int,
    ewc_agg: EWCAggregate | None = None,
    ewc_lambda: float = 0.0,
    max_grad_norm: float | None = 1.0,
    steps: int = 200,
):
    """Extra gradient steps on stratified replay only — locks past tasks."""
    if len(buffer) == 0:
        return 0.0

    model.train()
    total_loss = 0.0
    for _ in range(steps):
        replay_x, replay_y = buffer.sample_stratified(batch_size)
        if replay_x is None:
            break
        replay_x = replay_x.to(device)
        replay_y = replay_y.to(device)

        optimizer.zero_grad()
        out = model(replay_x)
        loss = criterion(out, replay_y)
        if ewc_agg is not None and ewc_lambda > 0 and len(ewc_agg) > 0:
            loss = loss + ewc_lambda * ewc_agg.penalty(model, device)
        loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(steps, 1)


def continual_training_with_replay(
    model,
    train_loaders,
    test_loaders,
    device,
    epochs: int = 12,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    buffer_capacity: int = 50000,
    num_classes: int = 10,
    samples_per_class: int = 5000,
    ewc_lambda: float = 25.0,
    fisher_max_batches: int = 400,
    replay_ratio: float = 3.0,
    per_task_lr_decay: float = 0.96,
    stratified_replay: bool = True,
    consolidation_epochs: int = 1,
):
    """
    Near–minimum forgetting: full exemplar storage per class (CIFAR: 5000),
    strong stratified replay, optional post-task consolidation on buffer only.
    """
    criterion = torch.nn.CrossEntropyLoss()
    buffer = ReplayBuffer(
        capacity=buffer_capacity,
        num_classes=num_classes,
        samples_per_class=samples_per_class,
    )
    ewc_agg = EWCAggregate()

    history = {"task_acc": [], "loss": []}

    for task_id, train_loader in enumerate(train_loaders):
        print(f"\n{'='*50}")
        print(f"  TRAINING + REPLAY — Task {task_id}"
              f"  (classes {task_id*2}, {task_id*2+1})")
        print(f"  Buffer size: {len(buffer)}  |  replay_ratio: {replay_ratio}")
        print(f"{'='*50}")

        task_lr = lr * (per_task_lr_decay ** task_id)
        optimizer = build_optimizer(model, task_lr, weight_decay)
        print(f"  Task LR (base): {task_lr:.6f}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=task_lr * 0.02
        )

        bs = train_loader.batch_size or 64

        for epoch in range(epochs):
            loss = train_with_replay(
                model,
                train_loader,
                optimizer,
                criterion,
                buffer,
                device,
                ewc_agg=ewc_agg,
                ewc_lambda=ewc_lambda,
                replay_ratio=replay_ratio,
                stratified_replay=stratified_replay,
            )
            scheduler.step()
            print(f"  Epoch {epoch+1}/{epochs} — Loss: {loss:.4f}")

        if consolidation_epochs > 0 and len(buffer) > 0:
            steps_per_epoch = max(
                1,
                min(800, (len(buffer) + bs - 1) // bs),
            )
            total_steps = steps_per_epoch * consolidation_epochs
            print(f"  Buffer consolidation: {total_steps} steps × stratified")
            cons_loss = run_buffer_consolidation(
                model,
                buffer,
                optimizer,
                criterion,
                device,
                batch_size=bs,
                ewc_agg=ewc_agg,
                ewc_lambda=ewc_lambda,
                steps=total_steps,
            )
            print(f"  Consolidation avg loss: {cons_loss:.4f}")

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
