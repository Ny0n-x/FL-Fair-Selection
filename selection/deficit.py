import numpy as np
from .base import Selector
from .tracker import Tracker


class DeficitSelector(Selector):
    """Rule 1, deficit-weighted sampling (heuristic; not PropFair, not Prop1).

    Def_i(R) = e_i(R) - k_i(R), with e_i = K*R/N (good_def="count") or
    w_i*K*R (good_def="weighted"). Sample K clients without replacement with
    probability proportional to max(0, Def_i) + eps. Stochastic for any eps > 0;
    no count-gap bound is proven. Needs no extra client queries.
    """

    def __init__(self, n_clients, k, data_sizes, cfg):
        super().__init__(n_clients, k, data_sizes, cfg)
        self.mode = cfg.get("good_def", "count")
        assert self.mode in ("count", "weighted"), self.mode
        self.eps = max(float(cfg.get("eps", 0.1)), 1e-12)
        self.tracker = Tracker(n_clients, k, data_sizes, cfg.get("T", 10))
        self.rng = np.random.default_rng(cfg.get("seed", 0))
        self.max_deficit = 0.0   # largest deficit seen at selection time (report empirically)

    def select(self, round_idx, ctx):
        deficit = self.tracker.deficit(self.mode)
        self.max_deficit = max(self.max_deficit, float(deficit.max()))
        w = np.maximum(0.0, deficit) + self.eps
        chosen = self.rng.choice(self.N, size=self.K, replace=False, p=w / w.sum())
        return [int(i) for i in chosen]

    def update(self, round_idx, selected, ctx):
        self.tracker.record(selected)
