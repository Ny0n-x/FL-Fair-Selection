import numpy as np


class PowerOfChoiceSelector:
    """Sample d candidates (prob. proportional to data size), evaluate the current
    global model on each, keep the top-K by loss.  (Cho, Wang, Joshi 2020)

    VERIFY against the paper: candidate sampling, loss basis (full local data here),
    and the value of d (default 2*K is a guess).
    """

    def __init__(self, n_clients, k, data_sizes, cfg):
        self.n, self.k = n_clients, k
        self.d = int(cfg.get("d") or 2 * k)
        assert self.k <= self.d <= self.n, "need K <= d <= N"
        sizes = np.asarray(data_sizes, dtype=float)
        self.p = sizes / sizes.sum()
        self.rng = np.random.default_rng(cfg.get("seed", 0))
        self.extra_queries = 0          # loss evaluations on non-selected clients too

    def select(self, round_idx, ctx):
        cand = self.rng.choice(self.n, size=self.d, replace=False, p=self.p)
        losses = np.asarray(ctx.loss_fn(cand))      # loss of CURRENT global model
        self.extra_queries += self.d
        top = np.argsort(-losses)[: self.k]         # highest loss first
        return [int(c) for c in cand[top]]

    def update(self, round_idx, selected, ctx):
        pass