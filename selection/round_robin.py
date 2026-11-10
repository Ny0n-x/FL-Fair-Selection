import numpy as np
from .base import Selector


class RoundRobinSelector(Selector):
    """Shuffle clients once (seeded), then cycle through them K at a time.
    The pointer advances in update(), so select() is idempotent within a round.
    Needs no extra client queries."""

    def __init__(self, n_clients, k, data_sizes, cfg):
        super().__init__(n_clients, k, data_sizes, cfg)
        rng = np.random.default_rng(cfg.get("seed", 0))
        self.order = rng.permutation(n_clients)
        self.ptr = 0

    def select(self, round_idx, ctx):
        idx = (self.ptr + np.arange(self.K)) % self.N
        return [int(i) for i in self.order[idx]]

    def update(self, round_idx, selected, ctx):
        self.ptr = (self.ptr + self.K) % self.N
