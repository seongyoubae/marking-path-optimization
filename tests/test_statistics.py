import pandas as pd
import pytest
from scipy.stats import ranksums

from analysis.statistical_test import compare_groups


def samples():
    return pd.DataFrame(
        {
            "method": ["SAC-GWO"] * 3 + ["GA"] * 3,
            "seed": [0, 1, 2] * 2,
            "travel_cost_normalized": [1, 2, 3, 4, 5, 6],
            "instance": ["synthetic_test"] * 6,
        }
    )


def test_uses_source_rank_sum_function():
    result = compare_groups(samples()).iloc[0]
    expected = ranksums([1, 2, 3], [4, 5, 6])
    assert result.statistic == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue)
    assert result.p_value_adjustment == "none"


def test_cannot_pool_distinct_inputs():
    frame = samples()
    frame.loc[0, "instance"] = "another_input"
    with pytest.raises(ValueError, match="instance"):
        compare_groups(frame)


def test_duplicate_seed_is_rejected():
    frame = samples()
    frame.loc[0, "seed"] = 1
    with pytest.raises(ValueError, match="Duplicate"):
        compare_groups(frame)


def test_missing_costs_removed_with_counts():
    frame = samples()
    frame.loc[0, "travel_cost_normalized"] = float("nan")
    assert compare_groups(frame).iloc[0].n_reference == 2
