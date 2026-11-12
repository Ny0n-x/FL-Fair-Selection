import numpy as np


class SubTruncSelector:
    """SUBTRUNC (Castillo J., Kaya, Ye, Hashemi, CDC 2024, arXiv 2408.13683). Standalone.

    Implements eq. (10)-(14) of the paper:
        W(S) = G(S) + H(S),   |S| <= K
        G(S) : DivFL facility location over client gradient/update vectors
               G(S) = sum_i max_{j in S} (c - ||g_i - g_j||),  c = max pairwise distance
        H(S) = lam * min(b, F(S)),   F(S) = sum_{i in S} phi(loss_i),  phi(x) = ln(1+x)
    Selection = stochastic greedy: at each of the K steps, sample r candidates from the
    clients not yet chosen and add the one with the best marginal gain of W.

    Config keys: subtrunc_b (default 1.10, paper), subtrunc_lambda (default 0.5; the
    paper sweeps 0.01-0.95), subtrunc_r (default 10, paper uses r=10 with N=100).
    lam = 0 recovers DivFL.

    Things the paper text does not pin down (our choices, state them in the report):
      - candidates are RESAMPLED at every greedy step (otherwise r=K would just be
        random selection)
      - g_i = stored local-update vector (refreshed when client i is selected, all
        clients at round 0), since fresh gradients from all N clients each round are
        costly; the paper writes grad f_i(w)
      - "most recent loss": the client's last reported local training loss (global-model
        loss on its data at round 0)
      - G is not normalized, so lam's effect depends on the gradient scale; if SUBTRUNC
        behaves exactly like DivFL, try subtrunc_norm: true (divides G by its maximum)
    """

    def __init__(self, n_clients, k, data_sizes, cfg):
        self.n, self.k = n_clients, k
        b = cfg.get("subtrunc_b")
        lam = cfg.get("subtrunc_lambda")
        r = cfg.get("subtrunc_r")
        self.b = 1.10 if b is None else float(b)
        self.lam = 0.5 if lam is None else float(lam)
        self.r = 10 if r is None else int(r)
        self.norm = bool(cfg.get("subtrunc_norm", False))
        self.rng = np.random.default_rng(cfg.get("seed", 0))
        self.G = None                                   # (N, P) stored update per client
        self.loss = np.zeros(n_clients)                 # most recent loss per client
        self.seen = np.zeros(n_clients, dtype=bool)
        self.extra_queries = 0

    def select(self, round_idx, ctx):
        missing = np.where(~self.seen)[0]
        if len(missing) > 0:                            # warm-up: update + loss from everyone
            U = np.asarray(ctx.update_fn(missing), dtype=np.float32)
            if self.G is None:
                self.G = np.zeros((self.n, U.shape[1]), dtype=np.float32)
            self.G[missing] = U
            self.loss[missing] = np.asarray(ctx.loss_fn(missing), dtype=float)
            self.seen[missing] = True
            self.extra_queries += len(missing)
        return self._greedy()

    def _greedy(self):
        G = self.G
        sq = (G ** 2).sum(1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * G @ G.T, 0))
        sim = D.max() - D                               # G(S) = sum_i max_{j in S} sim[i, j]
        if self.norm and sim.max() > 0:
            sim = sim / sim.max() / self.n
        phi = np.log1p(np.maximum(self.loss, 0.0))      # phi(x) = ln(1 + x)
        best = np.zeros(self.n)                         # current facility-location coverage
        F = 0.0                                         # F(S) of the set built so far
        chosen = []
        for _ in range(self.k):
            remaining = np.setdiff1d(np.arange(self.n), chosen)
            m = min(self.r, len(remaining))
            cand = self.rng.choice(remaining, size=m, replace=False)
            gain_G = np.maximum(sim[:, cand], best[:, None]).sum(0) - best.sum()
            gain_H = self.lam * (np.minimum(self.b, F + phi[cand]) - min(self.b, F))
            j = int(cand[np.argmax(gain_G + gain_H)])
            chosen.append(j)
            best = np.maximum(best, sim[:, j])
            F += phi[j]
        return chosen

    def update(self, round_idx, selected, ctx):
        losses = getattr(ctx, "losses", {})
        for i in selected:
            self.G[i] = ctx.updates[i]
            if i in losses:
                self.loss[i] = losses[i]