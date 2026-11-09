import numpy as np
import pandas as pd
import pytest

from src.evaluation.stats import (
    ci95,
    compare,
    holm_correction,
    seed_means,
    summarize,
    to_latex_table,
)


def make_frame():
    """Learned method 'learned' (5 train seeds x 2 test episodes) and
    rule-based 'rule' (train_seed 0, 5 test episodes)."""
    rows = []
    for seed in range(5):
        for test_seed, offset in ((10, -0.5), (11, 0.5)):
            rows.append({"method": "learned", "scenario": "single", "demand": "default",
                         "train_seed": seed, "test_seed": test_seed,
                         "avg_waiting_time": 10.0 + seed + offset, "throughput": 900.0 + seed})
    for i, test_seed in enumerate(range(10, 15)):
        rows.append({"method": "rule", "scenario": "single", "demand": "default",
                     "train_seed": 0, "test_seed": test_seed,
                     "avg_waiting_time": 30.0 + i, "throughput": 850.0 + i})
    return pd.DataFrame(rows)


def test_seed_means_average_test_episodes():
    means = seed_means(make_frame(), "avg_waiting_time")
    learned = means[means["method"] == "learned"].sort_values("train_seed")
    np.testing.assert_allclose(learned["avg_waiting_time"], [10, 11, 12, 13, 14])


def test_summarize_units_and_ci():
    summary = summarize(make_frame(), "avg_waiting_time").set_index("method")
    assert summary.loc["learned", "ci_unit"] == "train_seed"
    assert summary.loc["rule", "ci_unit"] == "test_episode"
    assert summary.loc["learned", "n"] == 5
    assert summary.loc["learned", "mean"] == pytest.approx(12.0)
    expected = 2.7764451 * np.std([10, 11, 12, 13, 14], ddof=1) / np.sqrt(5)
    assert summary.loc["learned", "ci95"] == pytest.approx(expected, rel=1e-5)


def test_ci_needs_two_values():
    assert np.isnan(ci95([3.0]))


def test_holm_known_example():
    adjusted = holm_correction([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.02])


def test_holm_caps_at_one():
    assert holm_correction([0.6, 0.7]) == [1.0, 1.0]


def test_compare_detects_clear_difference():
    result = compare(make_frame(), "learned", "rule", "avg_waiting_time", "single")
    assert result["test"] in {"welch_t", "mann_whitney_u"}
    assert result["p_value"] < 0.05
    assert result["effect_size"] < 0  # learned waits less
    assert result["n_a"] == 5 and result["n_b"] == 5


def test_compare_missing_method():
    with pytest.raises(ValueError):
        compare(make_frame(), "learned", "nope", "avg_waiting_time", "single")


def test_latex_table_bolds_best_per_direction():
    df = make_frame()
    summary = pd.concat([summarize(df, "avg_waiting_time"), summarize(df, "throughput")])
    table = to_latex_table(summary, ["avg_waiting_time", "throughput"], "Caption", "tab:test")
    assert "\\toprule" in table and "$\\pm$" in table
    learned_row = next(line for line in table.splitlines() if line.startswith("learned"))
    rule_row = next(line for line in table.splitlines() if line.startswith("rule"))
    assert learned_row.count("\\textbf") == 2  # lower wait and higher throughput
    assert "\\textbf" not in rule_row
