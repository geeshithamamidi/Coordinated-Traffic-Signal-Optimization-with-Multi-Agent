"""One environment factory so every method trains and evaluates on the same settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Relative network and route files, and the simulation begin time, match
# sumo_rl.environment.resco_envs. Cologne and Ingolstadt departures start at
# those begin times; num_seconds stays 3600 so every scenario is one hour.
_SCENARIOS: dict[str, dict[str, Any]] = {
    "single": {
        "net": "nets/single-intersection/single-intersection.net.xml",
        "route": "nets/single-intersection/single-intersection.rou.xml",
        "begin_time": 0,
    },
    "grid4x4": {
        "net": "nets/RESCO/grid4x4/grid4x4.net.xml",
        "route": "nets/RESCO/grid4x4/grid4x4_1.rou.xml",
        "begin_time": 0,
    },
    "arterial4x4": {
        "net": "nets/RESCO/arterial4x4/arterial4x4.net.xml",
        "route": "nets/RESCO/arterial4x4/arterial4x4_1.rou.xml",
        "begin_time": 0,
    },
    "cologne8": {
        "net": "nets/RESCO/cologne8/cologne8.net.xml",
        "route": "nets/RESCO/cologne8/cologne8.rou.xml",
        "begin_time": 25200,
    },
    "ingolstadt7": {
        "net": "nets/RESCO/ingolstadt7/ingolstadt7.net.xml",
        "route": "nets/RESCO/ingolstadt7/ingolstadt7.rou.xml",
        "begin_time": 57600,
    },
}

_NUM_SECONDS = 3600
_DELTA_TIME = 5
_YELLOW_TIME = 2
_MIN_GREEN = 5

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_sumo_home() -> None:
    if os.environ.get("SUMO_HOME"):
        return
    import sumo

    os.environ["SUMO_HOME"] = sumo.SUMO_HOME


def _use_pettingzoo_agent_selector_class() -> None:
    """Make sumo-rl's ``agent_selector(...)`` call work on PettingZoo 1.25+.

    sumo-rl 1.4.5 imports ``pettingzoo.utils.agent_selector`` and calls it.
    PettingZoo 1.25+ binds that name to the module. Bind the class first.
    """
    import pettingzoo.utils as pettingzoo_utils
    from pettingzoo.utils.agent_selector import AgentSelector

    if not callable(getattr(pettingzoo_utils, "agent_selector", None)):
        pettingzoo_utils.agent_selector = AgentSelector
    # A previous import may already have copied the module into sumo-rl.
    import sys

    env_module = sys.modules.get("sumo_rl.environment.env")
    if env_module is not None and not callable(getattr(env_module, "agent_selector", None)):
        env_module.agent_selector = AgentSelector


def _sumo_rl_api():
    _ensure_sumo_home()
    _use_pettingzoo_agent_selector_class()
    import sumo_rl
    from sumo_rl import env as aec_env
    from sumo_rl import parallel_env

    return sumo_rl, aec_env, parallel_env


def make_env(
    scenario: str,
    demand: str = "default",
    split: str = "train",
    seed: int = 0,
    parallel: bool = True,
    out_csv: str | None = None,
    reward_fn: str = "diff-waiting-time",
    observation_class=None,
    fixed_ts: bool = False,
    use_gui: bool = False,
    additional_sumo_cmd: str | None = None,
):
    """Build the shared PettingZoo environment for ``scenario``.

    ``parallel=True`` returns the PettingZoo parallel API. ``parallel=False``
    returns the AEC API. Timing, horizon, and ``sumo_seed`` are fixed so
    methods stay comparable.
    """
    if scenario not in _SCENARIOS:
        known = ", ".join(_SCENARIOS)
        raise ValueError(f"Unknown scenario {scenario!r}. Expected one of: {known}.")

    sumo_rl, aec_env, parallel_env = _sumo_rl_api()
    spec = _SCENARIOS[scenario]
    install_dir = Path(sumo_rl.__file__).resolve().parent
    net_file = install_dir / spec["net"]
    if demand == "default":
        route_file = install_dir / spec["route"]
    else:
        route_file = _REPO_ROOT / "scenarios" / scenario / f"{demand}_{split}.rou.xml"

    if not net_file.is_file():
        raise FileNotFoundError(f"SUMO network not found: {net_file}")
    if not route_file.is_file():
        raise FileNotFoundError(f"SUMO route file not found: {route_file}")

    kwargs: dict[str, Any] = {
        "net_file": str(net_file),
        "route_file": str(route_file),
        "begin_time": spec["begin_time"],
        "num_seconds": _NUM_SECONDS,
        "delta_time": _DELTA_TIME,
        "yellow_time": _YELLOW_TIME,
        "min_green": _MIN_GREEN,
        "sumo_seed": seed,
        "out_csv_name": out_csv,
        "reward_fn": reward_fn,
        "fixed_ts": fixed_ts,
        "use_gui": use_gui,
        "additional_sumo_cmd": additional_sumo_cmd,
    }
    if observation_class is not None:
        kwargs["observation_class"] = observation_class

    builder = parallel_env if parallel else aec_env
    return builder(**kwargs)
