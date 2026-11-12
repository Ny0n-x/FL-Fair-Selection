import numpy as np
from selection import REGISTRY
from selection.round_robin import RoundRobinSelector
from selection.deficit import DeficitSelector
from baselines.random_sel import RandomSelector


def run(sel, rounds):
    counts = np.zeros(sel.N if hasattr(sel, "N") else sel.n, dtype=int)
    for r in range(rounds):
        S = sel.select(r, None)
        assert len(set(S)) == len(S)
        sel.update(r, S, None)
        counts[S] += 1
    return counts


def test_registered():
    assert REGISTRY["round_robin"] is RoundRobinSelector
    assert REGISTRY["deficit"] is DeficitSelector


def test_round_robin_cycles_and_is_balanced():
    N, K = 50, 10
    sel = RoundRobinSelector(N, K, [1] * N, {"seed": 3})
    first = []
    for r in range(N // K):                       # one full cycle covers everyone once
        S = sel.select(r, None)
        assert len(set(S)) == K
        first += S
        sel.update(r, S, None)
    assert sorted(first) == list(range(N))
    assert (run(RoundRobinSelector(N, K, [1] * N, {"seed": 3}), 100) == 20).all()


def test_round_robin_wraps_when_k_does_not_divide_n():
    N, K = 7, 3
    counts = run(RoundRobinSelector(N, K, [1] * N, {"seed": 0}), 7 * 10)
    assert counts.max() - counts.min() <= 1


def test_round_robin_select_is_idempotent_before_update():
    sel = RoundRobinSelector(20, 5, [1] * 20, {"seed": 0})
    assert sel.select(0, None) == sel.select(0, None)


def test_deficit_count_variance_below_half_of_random():
    N, K, R = 50, 10, 1000
    sizes = [1] * N
    v_def, v_rand = [], []
    for seed in range(5):
        cfg = {"seed": seed, "good_def": "count", "eps": 0.1, "T": 10}
        v_def.append(run(DeficitSelector(N, K, sizes, cfg), R).var())
        v_rand.append(run(RandomSelector(N, K, sizes, cfg), R).var())
    assert np.mean(v_def) < 0.5 * np.mean(v_rand)


def test_deficit_weighted_follows_size_ratio():
    N, K, R = 20, 5, 1000
    sizes = [100] * 10 + [300] * 10               # 1:3 split
    counts = run(DeficitSelector(N, K, sizes, {"seed": 0, "good_def": "weighted", "eps": 0.1}), R)
    ratio = counts[10:].mean() / counts[:10].mean()
    assert 2.0 <= ratio <= 4.0


def test_deficit_stays_stochastic_and_valid_with_tiny_eps():
    N, K = 50, 10
    sel = DeficitSelector(N, K, [1] * N, {"seed": 0, "eps": 0.0, "good_def": "count"})
    run(sel, 200)                                  # eps=0 must not crash (fewer than K positive weights)
    assert sel.max_deficit >= 0
