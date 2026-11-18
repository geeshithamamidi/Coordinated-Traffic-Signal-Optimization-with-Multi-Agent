#!/usr/bin/env python3
"""Ablation table for shared PPO vs neighbor-obs and spatial-reward variants.

    python scripts/ablation_table.py

Reads results/raw/grid4x4/default/{shared_ppo,obs_only,reward_only,coordinated_ppo}.csv.
A missing file is printed and skipped. Writes results/tables/ablation_grid4x4.tex
and results/figures/ablation_grid4x4.png from the files that are present.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.coordinated_ppo import FROZEN_PATH, load_frozen  # noqa: E402
from src.evaluation.stats import (  # noqa: E402
    LOWER_IS_BETTER,
    METRIC_LABELS,
    compare,
    holm_correction,
    summarize,
)

RAW = REPO_ROOT / "results" / "raw" / "grid4x4" / "default"
TABLES = REPO_ROOT / "results" / "tables"
FIGURES = REPO_ROOT / "results" / "figures"
SCENARIO = "grid4x4"
DEMAND = "default"
BASELINE = "shared_ppo"
P_METRIC = "avg_waiting_time"
METRICS = ["avg_waiting_time", "avg_queue", "avg_trip_duration", "throughput"]

# method, neighbor observations, whether alpha is frozen (else 0).
VARIANTS = (
    ("shared_ppo", "off", False),
    ("obs_only", "on", False),
    ("reward_only", "off", True),
    ("coordinated_ppo", "on", True),
)
LABELS = {
    "shared_ppo": "Shared PPO",
    "obs_only": "Obs only",
    "reward_only": "Reward only",
    "coordinated_ppo": "Coordinated PPO",
}
COLORS = {
    "shared_ppo": "#0072B2",
    "obs_only": "#E69F00",
    "reward_only": "#009E73",
    "coordinated_ppo": "#D55E00",
}
REQUIRED = {"train_seed", *METRICS}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the grid4x4 coordinated-PPO ablation table.")
    parser.add_argument("--raw", type=Path, default=RAW, help="Directory of evaluate() CSVs.")
    parser.add_argument("--table", type=Path, default=TABLES / "ablation_grid4x4.tex")
    parser.add_argument("--figure", type=Path, default=FIGURES / "ablation_grid4x4.png")
    parser.add_argument("--frozen", type=Path, default=FROZEN_PATH)
    return parser.parse_args(argv)


def load_frames(raw_dir: Path) -> pd.DataFrame | None:
    """Load the ablation CSVs. Missing paths are printed and left out."""
    frames = []
    for method, _obs, _frozen in VARIANTS:
        path = raw_dir / f"{method}.csv"
        if not path.is_file():
            print(f"missing {_display(path)}")
            continue
        frame = pd.read_csv(path)
        missing_cols = REQUIRED - set(frame.columns)
        if missing_cols:
            print(f"{_display(path)} is missing columns: {', '.join(sorted(missing_cols))}")
            continue
        frame = frame.copy()
        frame["method"] = method
        frame["scenario"] = SCENARIO
        frame["demand"] = DEMAND
        frames.append(frame)
    if not frames:
        print("no ablation CSVs found, so no table or figure was written.")
        return None
    return pd.concat(frames, ignore_index=True)


def variant_alphas(frozen_path: Path) -> dict[str, float | None]:
    """Shared PPO and obs-only use alpha 0. The other two use configs/frozen.yaml."""
    alphas: dict[str, float | None] = {method: 0.0 for method, _obs, frozen in VARIANTS if not frozen}
    for method, _obs, frozen in VARIANTS:
        if frozen:
            alphas[method] = None
    try:
        payload = load_frozen(frozen_path)
    except (OSError, ValueError) as exc:
        print(f"could not read {_display(frozen_path)}: {exc}")
        return alphas
    alphas["reward_only"] = float(payload["reward_alpha"])
    alphas["coordinated_ppo"] = float(payload["alpha"])
    return alphas


def waiting_pvalues(df: pd.DataFrame) -> dict[str, float]:
    """Holm-adjusted p-values of each variant against shared PPO on waiting time."""
    methods = set(df["method"])
    if BASELINE not in methods:
        print("shared_ppo is missing, so p-values against it were not computed.")
        return {}
    raw: dict[str, float] = {}
    for method, _obs, _frozen in VARIANTS:
        if method == BASELINE or method not in methods:
            continue
        try:
            result = compare(df, method, BASELINE, P_METRIC, SCENARIO, DEMAND)
        except ValueError as exc:
            print(f"could not compare {method} with shared_ppo: {exc}")
            continue
        raw[method] = result["p_value"]
        print(
            f"{method} vs shared_ppo on {P_METRIC}: p={result['p_value']:.4g} "
            f"({result['test']}, n={result['n_a']} vs {result['n_b']})"
        )
    if not raw:
        return {}
    order = list(raw)
    adjusted = holm_correction([raw[method] for method in order])
    return dict(zip(order, adjusted))


def render_table(
    summary: pd.DataFrame,
    alphas: dict[str, float | None],
    pvalues: dict[str, float],
) -> str:
    present = [method for method, _obs, _frozen in VARIANTS if method in set(summary["method"])]
    obs_of = {method: obs for method, obs, _frozen in VARIANTS}
    blocks = {metric: summary[summary["metric"] == metric].set_index("method") for metric in METRICS}
    best = {metric: _best_method(blocks[metric], metric) for metric in METRICS}
    header = " & ".join(
        [
            "Variant",
            "Neighbor obs",
            "Alpha",
            *[METRIC_LABELS[metric] for metric in METRICS],
            "p vs shared PPO",
        ]
    )
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        (
            "\\caption{Ablation on grid4x4 (default demand). Neighbor observations and the "
            "spatial reward are added to shared PPO. Cells are mean $\\pm$ 95\\% CI. "
            "$p$ is the Holm-adjusted two-sided test of average waiting time against shared PPO.}"
        ),
        "\\label{tab:ablation_grid4x4}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
    ]
    for method in present:
        cells = [
            LABELS[method],
            obs_of[method],
            _format_alpha(alphas.get(method)),
        ]
        for metric in METRICS:
            block = blocks[metric]
            text = "--" if method not in block.index else _format_mean_ci(block.loc[method, "mean"], block.loc[method, "ci95"])
            if method == best.get(metric) and text != "--":
                text = f"\\textbf{{{text}}}"
            cells.append(text)
        cells.append("---" if method == BASELINE else _format_p(pvalues.get(method)))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    return "\n".join(lines)


def save_figure(summary: pd.DataFrame, path: Path) -> None:
    present = [method for method, _obs, _frozen in VARIANTS if method in set(summary["method"])]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    panels = (
        ("avg_waiting_time", "Avg waiting time (s)"),
        ("avg_queue", "Avg queue (veh)"),
    )
    for ax, (metric, ylabel) in zip(axes, panels):
        block = summary[summary["metric"] == metric].set_index("method")
        methods = [method for method in present if method in block.index]
        means = [float(block.loc[method, "mean"]) for method in methods]
        errors = [_error_bar(block.loc[method, "ci95"]) for method in methods]
        ax.bar(
            [LABELS[method] for method in methods],
            means,
            yerr=errors,
            capsize=4,
            color=[COLORS[method] for method in methods],
        )
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("grid4x4 ablation, mean ± 95% CI")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    df = load_frames(args.raw)
    if df is None:
        return
    summary = pd.concat([summarize(df, metric) for metric in METRICS], ignore_index=True)
    alphas = variant_alphas(args.frozen)
    pvalues = waiting_pvalues(df)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(render_table(summary, alphas, pvalues), encoding="utf-8")
    save_figure(summary, args.figure)
    print(f"wrote {_display(args.table)} and {_display(args.figure)}")


def _best_method(block: pd.DataFrame, metric: str) -> str | None:
    if block.empty:
        return None
    means = block["mean"].dropna()
    if means.empty:
        return None
    return str(means.idxmin() if LOWER_IS_BETTER.get(metric, True) else means.idxmax())


def _format_mean_ci(mean: float, half: float) -> str:
    if pd.isna(mean):
        return "--"
    if pd.isna(half):
        return f"{mean:.2f}"
    return f"{mean:.2f} $\\pm$ {half:.2f}"


def _format_alpha(alpha: float | None) -> str:
    if alpha is None or (isinstance(alpha, float) and np.isnan(alpha)):
        return "--"
    return f"{alpha:g}"


def _format_p(p: float | None) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    if p < 0.001:
        return "$<0.001$"
    return f"{p:.3f}"


def _error_bar(half: float) -> float:
    return 0.0 if pd.isna(half) else float(half)


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
