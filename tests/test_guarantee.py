import numpy as np
from selection.guarantee import GuaranteeSelector


def max_gap(N, K, T, rounds, losses):
    sel = GuaranteeSelector(N, K, [1] * N, {"T": T, "seed": 0, "filler": "loss"})
    last = np.zeros(N, dtype=int)
    worst = 0
    for r in range(1, rounds + 1):
        S = sel.select(r, losses)
        assert len(set(S)) == K
        sel.update(r, S, losses)
        for i in S:
            worst = max(worst, r - last[i])
            last[i] = r
    worst = max(worst, rounds - last.min())
    return worst


def test_adversarial_filler():
    # filler always prefers clients 0-9; naive forcing breaks here
    losses = {i: (100.0 if i < 10 else 0.0) for i in range(50)}
    assert max_gap(50, 10, 10, 1000, losses) <= 10


def test_tight_small_case():
    # N == K*T, zero slack
    losses = {i: float(i == 0) for i in range(6)}
    assert max_gap(6, 2, 3, 500, losses) <= 3