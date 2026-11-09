"""Statistics over evaluation CSVs: seed means, 95% CIs, tests, LaTeX tables.

The unit of analysis for a learned method is the training seed: test episodes
are averaged within each (method, scenario, demand, train_seed) first, and the
CI is taken across those seed means.

Rule-based methods (fixed_time, fixed_cycle, max_pressure) have a single
"train seed" (0). For them n=1 seed mean would give no CI, so the CI is taken
across their test episodes instead. ``summarize`` reports which unit was used
in the ``ci_unit`` column.

The 95% CI half-width uses the t-distribution:

.. math:: h = t_{0.975,\,n-1} \\frac{s}{\\sqrt{n}}
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

GROUP_COLUMNS = ("method", "scenario", "demand")

# True means lower values are better for that metric.
LOWER_IS_BETTER = {
    "avg_waiting_time": True,
    "avg_queue": True,
    "avg_trip_duration": True,
    "avg_time_loss": True,
    "teleports": True,
    "throughput": False,
    "avg_speed": False,
}

METRIC_LABELS = {
    "avg_waiting_time": "Wait (s)",
    "avg_queue": "Queue (veh)",
    "avg_trip_duration": "Trip (s)",
    "avg_time_loss": "Time loss (s)",
    "throughput": "Throughput (veh)",
    "avg_speed": "Speed (m/s)",
    "teleports": "Teleports",
}


def seed_means(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Average the test episodes within each (method, scenario, demand, train_seed)."""
    keys = [*GROUP_COLUMNS, "train_seed"]
    return df.groupby(keys, as_index=False)[metric].mean()


def ci95(values: Sequence[float]) -> float:
    """Half-width of the 95% t-interval. NaN when fewer than two values."""
    x = _clean(values)
    if x.size < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size))


def analysis_units(group: pd.DataFrame, metric: str) -> tuple[np.ndarray, str]:
    """Seed means if the group has several train seeds, else its test episodes."""
    if group["train_seed"].nunique() > 1:
        return _clean(group.groupby("train_seed")[metric].mean()), "train_seed"
    return _clean(group[metric]), "test_episode"


def summarize(df: pd.DataFrame, metric: str, by: Iterable[str] = GROUP_COLUMNS) -> pd.DataFrame:
    """Mean, std, n and 95% CI half-width of ``metric`` per group."""
    by = list(by)
    rows = []
    for keys, group in df.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        units, unit_name = analysis_units(group, metric)
        rows.append(
            {
                **dict(zip(by, keys)),
                "metric": metric,
                "mean": float(units.mean()) if units.size else float("nan"),
                "std": float(units.std(ddof=1)) if units.size > 1 else float("nan"),
                "n": int(units.size),
                "ci95": ci95(units),
                "ci_unit": unit_name,
            }
        )
    return pd.DataFrame(rows)


def compare(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    metric: str,
    scenario: str,
    demand: str = "default",
) -> dict:
    """Two-sided test of method_a vs method_b on the analysis units of each.

    Shapiro-Wilk is run on each group. If both look normal (p > 0.05), Welch's
    t-test is used with Cohen's d; otherwise Mann-Whitney U with the
    rank-biserial correlation. Shapiro-Wilk needs at least 3 values, so a group
    with fewer is treated as normal. Negative effect sizes mean method_a has
    the smaller values.
    """
    xa = _units_for(df, method_a, metric, scenario, demand)
    xb = _units_for(df, method_b, metric, scenario, demand)
    if xa.size < 2 or xb.size < 2:
        raise ValueError(f"Need at least 2 values per method, got {xa.size} and {xb.size}.")

    normal = all(_looks_normal(x) for x in (xa, xb))
    if normal:
        result = stats.ttest_ind(xa, xb, equal_var=False)
        test, effect_name, effect = "welch_t", "cohens_d", cohens_d(xa, xb)
    else:
        result = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        test, effect_name = "mann_whitney_u", "rank_biserial"
        effect = 2.0 * float(result.statistic) / (xa.size * xb.size) - 1.0
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "test": test,
        "effect_size": float(effect),
        "effect_name": effect_name,
        "n_a": int(xa.size),
        "n_b": int(xb.size),
        "mean_a": float(xa.mean()),
        "mean_b": float(xb.mean()),
    }


def cohens_d(xa: np.ndarray, xb: np.ndarray) -> float:
    """(mean_a - mean_b) / pooled std, with the average of the two variances."""
    pooled = np.sqrt((np.var(xa, ddof=1) + np.var(xb, ddof=1)) / 2.0)
    if pooled == 0:
        return 0.0
    return float((np.mean(xa) - np.mean(xb)) / pooled)


def holm_correction(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in the input order."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def to_latex_table(
    summary: pd.DataFrame,
    metrics: Sequence[str],
    caption: str,
    label: str,
    lower_is_better: dict[str, bool] | None = None,
) -> str:
    """Booktabs table: one row per method, "mean $\\pm$ ci" per metric.

    ``summary`` is the concatenation of ``summarize`` outputs for one scenario
    and demand (it has a ``metric`` column). The best mean per column is bold.
    """
    direction = {**LOWER_IS_BETTER, **(lower_is_better or {})}
    methods = sorted(summary["method"].unique())
    cells: dict[tuple[str, str], str] = {}
    for metric in metrics:
        block = summary[summary["metric"] == metric].set_index("method")
        if block.empty:
            continue
        means = block["mean"].dropna()
        best = (means.idxmin() if direction.get(metric, True) else means.idxmax()) if not means.empty else None
        for method in methods:
            if method not in block.index or np.isnan(block.loc[method, "mean"]):
                cells[(method, metric)] = "--"
                continue
            mean, half = block.loc[method, "mean"], block.loc[method, "ci95"]
            text = f"{mean:.2f}" if np.isnan(half) else f"{mean:.2f} $\\pm$ {half:.2f}"
            cells[(method, metric)] = f"\\textbf{{{text}}}" if method == best else text

    header = " & ".join(["Method", *[METRIC_LABELS.get(m, m.replace("_", "\\_")) for m in metrics]])
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\columnwidth}{!}{%",
        f"\\begin{{tabular}}{{l{'c' * len(metrics)}}}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
    ]
    for method in methods:
        row = [method.replace("_", "\\_"), *[cells.get((method, m), "--") for m in metrics]]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def _clean(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return x[~np.isnan(x)]


def _units_for(df: pd.DataFrame, method: str, metric: str, scenario: str, demand: str) -> np.ndarray:
    group = df[(df["method"] == method) & (df["scenario"] == scenario) & (df["demand"] == demand)]
    if group.empty:
        raise ValueError(f"No rows for method={method!r}, scenario={scenario!r}, demand={demand!r}.")
    return analysis_units(group, metric)[0]


def _looks_normal(x: np.ndarray) -> bool:
    if x.size < 3 or np.ptp(x) == 0:
        return True
    return bool(stats.shapiro(x).pvalue > 0.05)
