#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.fixed_time import (  # noqa: E402
    FIXED_TIME_ENV_KWARGS,
    GREEN_CANDIDATES,
    fixed_cycle_factory,
    fixed_time_factory,
    tune_fixed_cycle,
)
from src.baselines.max_pressure import max_pressure_factory  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402
from src.utils.config import load_config  # noqa: E402

METHODS = ("fixed_time", "fixed_cycle", "max_pressure")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rule-based baselines through evaluate().")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--demand", default="default")
    parser.add_argument("--test-seeds", type=int, nargs="+")
    parser.add_argument("--validation-seeds", type=int, nargs="+")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--green-s", type=int, help="Skip tuning and use this fixed-cycle green time")
    parser.add_argument("--config")
    return parser.parse_args(argv)


def run_fixed_time(args, test_seeds):
    return evaluate("fixed_time", args.scenario, args.demand, fixed_time_factory, 0, test_seeds, FIXED_TIME_ENV_KWARGS)


def run_fixed_cycle(args, test_seeds, validation_seeds):
    if args.green_s is not None:
        green_s = args.green_s
    else:
        green_s, scores = tune_fixed_cycle(args.scenario, args.demand, validation_seeds, GREEN_CANDIDATES)
        out = REPO_ROOT / "results" / "raw" / args.scenario / args.demand / "fixed_cycle_tuning.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"green_s": list(scores), "val_avg_waiting_time": list(scores.values())}
        ).to_csv(out, index=False)
        print(f"best green_s={green_s}")
    return evaluate("fixed_cycle", args.scenario, args.demand, fixed_cycle_factory(green_s), 0, test_seeds)


def run_max_pressure(args, test_seeds):
    return evaluate("max_pressure", args.scenario, args.demand, max_pressure_factory, 0, test_seeds)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    test_seeds = args.test_seeds or [int(s) for s in config["seeds"]["test"]]
    validation_seeds = args.validation_seeds or [int(s) for s in config["seeds"]["validation"]]
    frames = []
    if "fixed_time" in args.methods:
        frames.append(run_fixed_time(args, test_seeds))
    if "fixed_cycle" in args.methods:
        frames.append(run_fixed_cycle(args, test_seeds, validation_seeds))
    if "max_pressure" in args.methods:
        frames.append(run_max_pressure(args, test_seeds))
    if frames:
        print(pd.concat(frames).to_string(index=False))


if __name__ == "__main__":
    main()
