import numpy as np
from .base import Selector
from .tracker import Tracker


class GuaranteeSelector(Selector):
    """Rule 2 (M=1): every client is picked at least once in any window of T rounds."""

    def __init__(self, n_clients, k, data_sizes, cfg):
        super().__init__(n_clients, k, data_sizes, cfg)
        self.T = cfg.get("T", 10)
        assert n_clients <= k * self.T, "infeasible: need N <= K*T (M=1)"
        self.tracker = Tracker(n_clients, k, data_sizes, self.T)
        self.rng = np.random.default_rng(cfg.get("seed", 0))
        self.filler = cfg.get("filler", "random")   # "random" or "loss"

    def _feasible_after(self, S):
        t = self.tracker.R + 1                      # round being chosen now
        d = self.tracker.deadlines().copy()
        d[list(S)] = t + self.T                     # selected clients get new deadlines
        if (d < t + 1).any():                       # someone missed their deadline
            return False
        for j in range(self.T):
            if (d <= t + 1 + j).sum() > self.K * (j + 1):
                return False
        return True

    def select(self, round_idx, client_losses):
        d = self.tracker.deadlines()
        if self.filler == "loss" and client_losses:
            pri = np.array([client_losses.get(i, -np.inf) for i in range(self.N)])
        else:
            pri = self.rng.random(self.N)
        # EDF base set: earliest deadlines first, ties broken by priority
        S = [int(i) for i in np.lexsort((-pri, d))[:self.K]]
        # try to swap in higher-priority clients while the guarantee stays safe
        for c in np.argsort(-pri):
            c = int(c)
            if c in S:
                continue
            worst = max(S, key=lambda i: d[i])      # least urgent current member
            trial = [x for x in S if x != worst] + [c]
            if self._feasible_after(trial):
                S = trial
        return S

    def update(self, round_idx, selected, client_losses):
        self.tracker.record(selected)