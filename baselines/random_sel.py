# import numpy as np
# from selection.base import Selector   # adjust to whatever base.py actually exports


# class RandomSelector(Selector):
#     """FedAvg-style uniform sampling of K clients without replacement."""

#     def __init__(self, n_clients, k, data_sizes, cfg):
#         self.n = n_clients
#         self.k = k
#         seed = cfg.get("seed", 0) if isinstance(cfg, dict) else getattr(cfg, "seed", 0)
#         self.rng = np.random.default_rng(seed)   # own RNG, so runs reproduce per seed

#     def select(self, round_idx, client_losses):
#         return self.rng.choice(self.n, size=self.k, replace=False).tolist()

#     def update(self, round_idx, selected, client_losses):
#         pass   # Random keeps no state

import numpy as np


class RandomSelector:
    """FedAvg-style uniform sampling of K clients without replacement."""

    def __init__(self, n_clients, k, data_sizes, cfg):
        self.n, self.k = n_clients, k
        self.rng = np.random.default_rng(cfg.get("seed", 0))
        self.extra_queries = 0          # Random needs no extra client queries

    def select(self, round_idx, ctx):
        return [int(i) for i in self.rng.choice(self.n, size=self.k, replace=False)]

    def update(self, round_idx, selected, ctx):
        pass