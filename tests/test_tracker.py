import numpy as np
from selection.tracker import Tracker


def test_counts_and_deficit():
    N, K, T = 50, 10, 10
    tr = Tracker(N, K, [1] * N, T)
    rng = np.random.default_rng(0)
    history = []
    for _ in range(1000):
        sel = rng.choice(N, K, replace=False)
        tr.record(sel)
        history.append(set(int(i) for i in sel))
    assert tr.counts.sum() == 1000 * K
    assert abs(tr.deficit("count").sum()) < 1e-6
    brute = np.zeros(N, dtype=int)
    for s in history[-T:]:
        for i in s:
            brute[i] += 1
    assert (tr.window_count() == brute).all()