import math

import numpy as np
import pytest

from metrics.metrics import (
    coeff_variation, entropy_normalized, jain_index, min_acc, outcome_metrics,
    selection_freq_variance, variance, worst_frac_mean,
)


def test_jain_equal_is_one():
    assert jain_index([0.7] * 50) == pytest.approx(1.0)


def test_jain_one_nonzero_is_one_over_n():
    assert jain_index([1, 0, 0, 0]) == pytest.approx(0.25)


def test_jain_known():
    # (1+2+3)^2 / (3 * (1+4+9)) = 36/42
    assert jain_index([1, 2, 3]) == pytest.approx(36 / 42)


def test_jain_scale_invariant():
    x = np.array([0.2, 0.5, 0.9])
    assert jain_index(x) == pytest.approx(jain_index(100 * x))


def test_cv_known():
    # mean 2, population std sqrt(2/3)
    assert coeff_variation([1, 2, 3]) == pytest.approx(math.sqrt(2 / 3) / 2)


def test_variance_population():
    assert variance([1, 2, 3, 4]) == pytest.approx(1.25)


def test_min():
    assert min_acc([0.4, 0.1, 0.9]) == pytest.approx(0.1)


def test_worst10_n50_takes_5():
    x = list(range(50))  # lowest 5 = 0..4, mean 2
    assert worst_frac_mean(x) == pytest.approx(2.0)


def test_worst10_small_n_takes_at_least_one():
    assert worst_frac_mean([5, 1, 3]) == pytest.approx(1.0)


def test_worst10_n15_rounds_up():
    x = list(range(15))  # ceil(1.5) = 2 -> mean(0, 1)
    assert worst_frac_mean(x) == pytest.approx(0.5)


def test_entropy_uniform_is_log_n():
    assert entropy_normalized([0.3] * 50) == pytest.approx(math.log(50))


def test_entropy_point_mass_is_zero():
    assert entropy_normalized([1, 0, 0]) == pytest.approx(0.0)


def test_selection_freq_variance_round_robin_is_zero():
    assert selection_freq_variance([20] * 50) == 0.0


def test_rejects_empty_and_nan():
    with pytest.raises(ValueError):
        jain_index([])
    with pytest.raises(ValueError):
        variance([1.0, float("nan")])


def test_outcome_metrics_keys():
    m = outcome_metrics(np.linspace(0.2, 0.8, 50))
    assert set(m) == {"mean_client_acc", "variance", "std", "min", "worst10_mean",
                      "jain", "cv", "entropy"}


from metrics.metrics import rounds_to_target, size_buckets  # noqa: E402


def test_rounds_to_target_no_smoothing():
    assert rounds_to_target([0.1, 0.3, 0.5, 0.7], 0.5, smooth=1) == 3


def test_rounds_to_target_smoothing_ignores_spike():
    acc = [0.1, 0.9, 0.1, 0.1, 0.6, 0.6, 0.6]
    assert rounds_to_target(acc, 0.5, smooth=1) == 2
    # trailing 3-round means end at rounds 3..7: .367 .367 .267 .433 .6
    assert rounds_to_target(acc, 0.5, smooth=3) == 7


def test_rounds_to_target_never():
    assert rounds_to_target([0.1, 0.2], 0.9) is None


def test_size_buckets_quartiles():
    b = size_buckets([50, 10, 40, 20, 30, 60, 80, 70])
    assert b.tolist() == [2, 0, 1, 0, 1, 2, 3, 3]
