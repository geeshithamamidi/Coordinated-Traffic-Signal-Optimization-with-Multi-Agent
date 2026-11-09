#!/usr/bin/env python3
"""Turn results/raw CSVs into the report's main tables and bar charts.

    python scripts/aggregate_results.py                       # single + grid4x4
    python scripts/aggregate_results.py --scenario grid4x4 --demand default

Writes results/tables/main_<scenario>.tex and results/figures/bars_<scenario>.png
(non-default demand adds a _<demand> suffix).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import SUMMARY_COLUMNS  # noqa: E402
from src.evaluation.stats import summarize, to_latex_table  # noqa: E402

RAW = REPO_ROOT / "results" / "raw"
TABLES = REPO_ROOT / "results" / "tables"
FIGURES = REPO_ROOT / "results" / "figures"
METRICS = ["avg_waiting_time", "avg_queue", "avg_trip_duration", "throughput"]

# Okabe-Ito colorblind-safe palette, used when plot_style.py is not there yet.
FALLBACK = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate evaluation CSVs into tables and figures.")
    parser.add_argument("--scenario", nargs="+", default=["single", "grid4x4"])
    parser.add_argument("--demand", default="default")
    return parser.parse_args(argv)


def load_results(scenario: str, demand: str) -> pd.DataFrame:
    """Concatenate every evaluate() CSV in results/raw/<scenario>/<demand>/."""
    folder = RAW / scenario / demand
    frames = []
    for path in sorted(folder.glob("*.csv")):
        frame = pd.read_csv(path)
        if set(SUMMARY_COLUMNS) <= set(frame.columns):  # skips e.g. fixed_cycle_tuning.csv
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SUMMARY_COLUMNS)


def method_colors(methods: list[str]) -> dict[str, str]:
    try:
        from src.evaluation import plot_style  # type: ignore

        palette = getattr(plot_style, "METHOD_COLORS", {})
    except ImportError:
        palette = {}
    return {m: palette.get(m, FALLBACK[i % len(FALLBACK)]) for i, m in enumerate(methods)}


def bar_chart(summary: pd.DataFrame, scenario: str, demand: str, out: Path) -> None:
    block = summary[summary["metric"] == "avg_waiting_time"].sort_values("mean")
    colors = method_colors(sorted(block["method"]))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(
        block["method"],
        block["mean"],
        yerr=block["ci95"].fillna(0),
        capsize=4,
        color=[colors[m] for m in block["method"]],
    )
    ax.set_xlabel("Method")
    ax.set_ylabel("Avg waiting time per vehicle (s)")
    ax.set_title(f"{scenario}, {demand} demand (mean ± 95% CI)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for scenario in args.scenario:
        df = load_results(scenario, args.demand)
        if df.empty:
            print(f"[{scenario}] no result CSVs in {RAW / scenario / args.demand}, skipping.")
            continue
        summary = pd.concat([summarize(df, metric) for metric in METRICS], ignore_index=True)
        suffix = "" if args.demand == "default" else f"_{args.demand}"
        caption = (
            f"Results on {scenario} ({args.demand} demand), mean $\\pm$ 95\\% CI over training seeds "
            "(over test episodes for rule-based methods)."
        )
        table = TABLES / f"main_{scenario}{suffix}.tex"
        table.write_text(to_latex_table(summary, METRICS, caption, f"tab:main_{scenario}{suffix}"), encoding="utf-8")
        figure = FIGURES / f"bars_{scenario}{suffix}.png"
        bar_chart(summary, scenario, args.demand, figure)

        wide = summary.pivot_table(index="method", columns="metric", values="mean")[METRICS]
        print(f"\n[{scenario}] methods found: {', '.join(sorted(df['method'].unique()))}")
        print(wide.round(2).to_string())
        print(f"wrote {table.relative_to(REPO_ROOT)} and {figure.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
