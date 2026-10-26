#!/usr/bin/env python3
"""Train one shared PPO policy on a scenario.

Example:
    python scripts/train_shared_ppo.py --scenario single --seed 0 --episodes 100
    python scripts/train_shared_ppo.py --scenario grid4x4 --seed 1 --episodes 300 --config configs/override.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.shared_ppo import train  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a shared PPO policy with Stable-Baselines3.")
    parser.add_argument("--scenario", required=True, help="single, grid4x4, arterial4x4, cologne8, or ingolstadt7")
    parser.add_argument("--seed", required=True, type=int, help="SUMO and PPO seed")
    parser.add_argument("--episodes", required=True, type=int, help="Training episodes of 3600 s each")
    parser.add_argument("--config", help="YAML merged over configs/base.yaml")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    model_file, curve_file = train(args.scenario, args.seed, args.episodes, config)
    print(f"model: {model_file}")
    print(f"curve: {curve_file}")


if __name__ == "__main__":
    main()
