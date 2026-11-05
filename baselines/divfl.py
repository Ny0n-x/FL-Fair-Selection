# import numpy as np
# from selection.base import Selector   # match base.py


# class DivFLSelector(Selector):
#     """Greedy facility-location selection over client update vectors (Balakrishnan et al., ICLR 2022)."""

#     def __init__(self, n_clients, k, data_sizes, cfg):
#         self.n, self.k = n_clients, k
#         self.G = None                  # (N, P) last known update per client
#         self.seen = np.zeros(n_clients, dtype=bool)
#         self.query_count = 0           # extra client computations, for the communication claim

#     def select(self, round_idx, update_fn):
#         # warm-up: get an update from every client that has none yet (all N in round 0)
#         missing = np.where(~self.seen)[0]
#         if len(missing) > 0:
#             U = np.asarray(update_fn(missing))
#             if self.G is None:
#                 self.G = np.zeros((self.n, U.shape[1]), dtype=np.float32)
#             self.G[missing] = U
#             self.seen[missing] = True
#             self.query_count += len(missing)
#         return self._greedy()

#     def _greedy(self):
#         G = self.G
#         sq = (G ** 2).sum(1)
#         D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * G @ G.T, 0))   # pairwise distances
#         sim = D.max() - D                                                     # similarity >= 0
#         best = np.zeros(self.n)        # best similarity of each client to the chosen set so far
#         chosen = []
#         for _ in range(self.k):
#             gain = np.maximum(sim, best[:, None]).sum(0) - best.sum()
#             gain[chosen] = -np.inf
#             j = int(np.argmax(gain))
#             chosen.append(j)
#             best = np.maximum(best, sim[:, j])
#         return chosen

#     def update(self, round_idx, selected, updates):
#         # updates: dict {client_id: flat update vector} for the clients just trained
#         for i in selected:
#             self.G[i] = updates[i]

import numpy as np


class DivFLSelector:
    """Greedy facility-location selection over client update vectors
    (Balakrishnan et al., ICLR 2022).

    Simplification: round 0 gets an update from every client; afterwards each client's
    stored update is refreshed only when it is selected (stale otherwise).
    VERIFY against the paper: gradient vs update, staleness handling, and aggregation
    weights (paper may weight by cluster size; we keep data-size FedAvg on purpose).
    """

    def __init__(self, n_clients, k, data_sizes, cfg):
        self.n, self.k = n_clients, k
        self.G = None                               # (N, P) last known update per client
        self.seen = np.zeros(n_clients, dtype=bool)
        self.extra_queries = 0

    def select(self, round_idx, ctx):
        missing = np.where(~self.seen)[0]
        if len(missing) > 0:
            U = np.asarray(ctx.update_fn(missing), dtype=np.float32)
            if self.G is None:
                self.G = np.zeros((self.n, U.shape[1]), dtype=np.float32)
            self.G[missing] = U
            self.seen[missing] = True
            self.extra_queries += len(missing)
        return self._greedy()

    def _greedy(self):
        G = self.G
        sq = (G ** 2).sum(1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * G @ G.T, 0))
        sim = D.max() - D                           # similarity >= 0
        best = np.zeros(self.n)                     # best similarity to chosen set so far
        chosen = []
        for _ in range(self.k):
            gain = np.maximum(sim, best[:, None]).sum(0) - best.sum()
            gain[chosen] = -np.inf
            j = int(np.argmax(gain))
            chosen.append(j)
            best = np.maximum(best, sim[:, j])
        return chosen

    def update(self, round_idx, selected, ctx):
        for i in selected:
            self.G[i] = ctx.updates[i]