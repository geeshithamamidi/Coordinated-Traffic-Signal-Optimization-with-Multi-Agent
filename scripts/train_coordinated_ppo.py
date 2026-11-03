#!/usr/bin/env python3
"""Train coordinated PPO, or freeze alpha on the validation route seeds.

One script covers the ablations:

    shared PPO    --neighbor-obs off --alpha 0
    obs only      --neighbor-obs on  --alpha 0
    reward only   --neighbor-obs off --alpha 0.25   # any alpha in {0.25, 0.5, 0.75}
    coordinated   --neighbor-obs on  --alpha <value from configs/frozen.yaml>

Examples:
    python scripts/train_coordinated_ppo.py --scenario grid4x4 --seed 0 --episodes 300 --neighbor-obs on --alpha 0.5
    python scripts/train_coordinated_ppo.py --select-alpha --scenario grid4x4 --seed 0 --episodes 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.coordinated_ppo import ALLOWED_ALPHAS, choose_alpha, train  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train coordinated PPO or select its spatial-reward alpha.")
    parser.add_argument("--scenario", required=True, help="single, grid4x4, arterial4x4, cologne8, or ingolstadt7")
    parser.add_argument("--seed", required=True, type=int, help="SUMO and PPO train seed")
    parser.add_argument("--episodes", required=True, type=int, help="Training episodes of 3600 s each")
    parser.add_argument("--demand", default="default")
    parser.add_argument("--config", help="YAML merged over configs/base.yaml")
    parser.add_argument(
        "--neighbor-obs",
        choices=["on", "off"],
        help="Append the fixed neighbor block. Required unless --select-alpha is set.",
    )
    parser.add_argument(
        "--alpha",
        choices=[f"{alpha:g}" for alpha in ALLOWED_ALPHAS],
        help="Spatial reward weight. Required unless --select-alpha is set.",
    )
    parser.add_argument(
        "--select-alpha",
        action="store_true",
        help="Train each alpha with neighbor observations on, score validation seeds, and write configs/frozen.yaml.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.select_alpha:
        best = choose_alpha(
            args.scenario,
            args.seed,
            args.episodes,
            config,
            demand=args.demand,
        )
        print(f"alpha: {best:g}")
        return
    if args.neighbor_obs is None or args.alpha is None:
        raise SystemExit("--neighbor-obs and --alpha are required unless --select-alpha is set.")
    model_file, curve_file = train(
        args.scenario,
        args.seed,
        args.episodes,
        config,
        neighbor_obs=args.neighbor_obs,
        alpha=float(args.alpha),
        demand=args.demand,
    )
    print(f"model: {model_file}")
    print(f"curve: {curve_file}")


if __name__ == "__main__":
    main()
