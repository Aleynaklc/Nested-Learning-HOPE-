"""
ReplayBuffer — Class-balanced experience replay for continual learning.

Optional full per-class storage (e.g. 5000/class on CIFAR-10 train) enables
near–zero catastrophic forgetting when combined with strong rehearsal.
"""

import random
import torch


class ReplayBuffer:
    """Per-class reservoir / exemplar storage with balanced sampling."""

    def __init__(
        self,
        capacity: int = 50000,
        num_classes: int = 10,
        samples_per_class: int | None = None,
    ):
        """
        Args:
            capacity: used only if samples_per_class is None (per_class_cap = capacity // K).
            samples_per_class: hard cap per label (e.g. 5000 for full CIFAR-10 train/class).
        """
        self.num_classes = num_classes
        if samples_per_class is not None:
            self.per_class_cap = samples_per_class
        else:
            self.per_class_cap = max(1, capacity // num_classes)

        self.buffer: dict[int, list[tuple[torch.Tensor, int]]] = {
            c: [] for c in range(num_classes)
        }
        self.seen_per_class = {c: 0 for c in range(num_classes)}

    def __len__(self):
        return sum(len(v) for v in self.buffer.values())

    def add_batch(self, x: torch.Tensor, y: torch.Tensor):
        """Reservoir sampling independently within each class bucket."""
        for i in range(x.size(0)):
            sample_x = x[i].detach().cpu()
            c = int(y[i].item())
            if c < 0 or c >= self.num_classes:
                continue

            self.seen_per_class[c] += 1
            bucket = self.buffer[c]

            if len(bucket) < self.per_class_cap:
                bucket.append((sample_x, c))
            else:
                j = random.randint(0, self.seen_per_class[c] - 1)
                if j < self.per_class_cap:
                    bucket[j] = (sample_x, c)

    def sample(self, batch_size: int):
        """
        Sample by choosing a class uniformly among classes present,
        then a random exemplar — balances replay in expectation.
        """
        nonempty = [c for c, b in self.buffer.items() if len(b) > 0]
        if not nonempty:
            return None, None

        picks: list[tuple[torch.Tensor, int]] = []
        for _ in range(batch_size):
            c = random.choice(nonempty)
            b = self.buffer[c]
            picks.append(b[random.randint(0, len(b) - 1)])

        xs, ys = zip(*picks)
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    def sample_stratified(self, batch_size: int):
        """
        As equal as possible samples from each class that currently has data —
        best for rehearsal when minimizing forgetting across tasks.
        """
        nonempty = [c for c, b in self.buffer.items() if len(b) > 0]
        if not nonempty:
            return None, None

        if batch_size < len(nonempty):
            return self.sample(batch_size)

        n_cls = len(nonempty)
        base = batch_size // n_cls
        rem = batch_size % n_cls
        random.shuffle(nonempty)

        picks: list[tuple[torch.Tensor, int]] = []
        for i, c in enumerate(nonempty):
            n_take = base + (1 if i < rem else 0)
            b = self.buffer[c]
            for _ in range(n_take):
                picks.append(b[random.randint(0, len(b) - 1)])

        random.shuffle(picks)
        xs, ys = zip(*picks)
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)
