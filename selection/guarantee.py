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

    def select(self, round_idx, ctx):
        d = self.tracker.deadlines()
        if self.filler == "loss":
            pri = np.asarray(ctx.loss_fn(range(self.N)), dtype=float)
            self.extra_queries += self.N  # loss filler queries every client
        else:
            pri = self.rng.random(self.N)
        S = [int(i) for i in np.lexsort((-pri, d))[:self.K]]
        for c in np.argsort(-pri):
            c = int(c)
            if c in S:
                continue
            for w in sorted(S, key=lambda i: pri[i]):  # lowest priority first
                if pri[w] >= pri[c]:
                    break  # nobody left that c beats
                trial = [x for x in S if x != w] + [c]
                if self._feasible_after(trial):
                    S = trial
                    break
        return S

    def update(self, round_idx, selected, ctx):
        self.tracker.record(selected)