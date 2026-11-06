import numpy as np
from selection.guarantee import GuaranteeSelector


class FakeCtx:
    def __init__(self, losses):
        self.losses = losses
    def loss_fn(self, ids):
        return [self.losses[i] for i in ids]


def max_gap(N, K, T, rounds, losses):
    sel = GuaranteeSelector(N, K, [1] * N, {"T": T, "seed": 0, "filler": "loss"})
    ctx = FakeCtx(losses)
    last = np.zeros(N, dtype=int)
    worst = 0
    for r in range(1, rounds + 1):
        S = sel.select(r, ctx)
        assert len(set(S)) == K
        sel.update(r, S, ctx)
        for i in S:
            worst = max(worst, r - last[i])
            last[i] = r
    return max(worst, rounds - last.min())


def test_adversarial_filler():
    losses = {i: (100.0 if i < 10 else 0.0) for i in range(50)}
    assert max_gap(50, 10, 10, 1000, losses) <= 10


def test_tight_small_case():
    losses = {i: float(i == 0) for i in range(6)}
    assert max_gap(6, 2, 3, 500, losses) <= 3

def test_loss_filler_picks_top_k_in_round_one():
    n, k = 50, 10
    losses = np.random.default_rng(0).random(n)

    class Ctx:
        loss_fn = staticmethod(lambda ids: losses[list(ids)])

    sel = GuaranteeSelector(n, k, [1] * n, {"T": 10, "filler": "loss", "seed": 0})
    assert set(sel.select(0, Ctx())) == set(np.argsort(-losses)[:k].tolist())