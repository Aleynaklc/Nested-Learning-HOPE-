#!/usr/bin/env python3
"""2 görev × 2 epoch hızlı demo — tam karşılaştırma için: python run_replay.py"""

import torch

from data.dataset import get_continual_tasks
from models.cnn import SimpleCNN
from models.cnn_with_memory import CNNWithMemory
from training.trainer import continual_training
from training.replay_trainer import continual_training_with_replay
from utils.metrics import compute_forgetting, average_forgetting, average_accuracy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    train_loaders, test_loaders = get_continual_tasks(batch_size=128)
    train_loaders = train_loaders[:2]
    test_loaders = test_loaders[:2]

    epochs = 2

    print("=" * 60)
    print("HIZLI DEMO — 2 görev, 2 epoch")
    print("Tam 5 görev için: python run_replay.py")
    print("=" * 60)

    print("\n[1/3] Baseline CNN...")
    m = SimpleCNN(10).to(device)
    h1 = continual_training(
        m, train_loaders, test_loaders, device,
        epochs=epochs, ewc_lambda=0.0, per_task_lr_decay=1.0,
    )

    print("\n[2/3] CNN + Nested Memory + EWC...")
    m2 = CNNWithMemory(10).to(device)
    h2 = continual_training(
        m2, train_loaders, test_loaders, device,
        epochs=epochs, ewc_lambda=45.0, per_task_lr_decay=0.96,
    )

    print("\n[3/3] CNN + Memory + Full Replay...")
    m3 = CNNWithMemory(10).to(device)
    h3 = continual_training_with_replay(
        m3, train_loaders, test_loaders, device,
        epochs=epochs,
        buffer_capacity=50000,
        samples_per_class=5000,
        replay_ratio=3.0,
        consolidation_epochs=1,
        ewc_lambda=25.0,
    )

    print("\n" + "=" * 60)
    print("ÖZET")
    print("=" * 60)
    for name, hist in [
        ("Baseline", h1),
        ("Memory + EWC", h2),
        ("Memory + Replay", h3),
    ]:
        fgt = compute_forgetting(hist["task_acc"])
        print(f"\n{name}:")
        print(f"  Ortalama unutma: {average_forgetting(fgt):.4f}")
        print(f"  Ortalama doğruluk (final): {average_accuracy(hist['task_acc']):.4f}")
        print(f"  Son görev doğrulukları: {[round(x, 4) for x in hist['task_acc'][-1]]}")


if __name__ == "__main__":
    main()
