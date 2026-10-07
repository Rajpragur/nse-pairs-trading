import numpy as np
import pandas as pd
import pytest

from pairs_trading.robustness import cost_sweep, randomized_controls


def _fixture():
    index = pd.bdate_range("2024-01-01", periods=30)
    z = pd.Series([0.0, 2.2, 2.1, 0.0, -2.3, -2.1, 0.0] * 4 + [0.0, 0.0], index=index)
    a = pd.Series(100 * (1.01 ** np.arange(30)), index=index)
    b = pd.Series(100 * (1.002 ** np.arange(30)), index=index)
    beta = pd.Series(1.0, index=index)
    return z, a, b, beta


def test_cost_sweep_is_monotone_when_gross_path_unchanged():
    result = cost_sweep(*_fixture(), cost_bps=[0, 2, 10])
    assert list(result.cost_bps) == [0, 2, 10]
    assert result.loc[0, "total_return"] >= result.loc[1, "total_return"] >= result.loc[2, "total_return"]
    assert (result.total_cost >= 0).all()


def test_randomized_controls_are_deterministic_and_labelled_null():
    args = _fixture()
    left = randomized_controls(*args, cost_bps=2, permutations=20, seed=7)
    right = randomized_controls(*args, cost_bps=2, permutations=20, seed=7)
    pd.testing.assert_frame_equal(left, right)
    assert len(left) == 20
    assert left.control_type.eq("circular_block_permutation").all()
    assert left.null_description.str.len().gt(0).all()


def test_robustness_validates_costs_and_block_parameters():
    args = _fixture()
    with pytest.raises(ValueError, match="cost_bps"):
        cost_sweep(*args, cost_bps=[-1])
    with pytest.raises(ValueError, match="block_length"):
        randomized_controls(*args, cost_bps=1, permutations=2, block_length=0)
