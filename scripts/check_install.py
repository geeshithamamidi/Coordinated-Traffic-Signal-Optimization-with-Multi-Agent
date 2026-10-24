#!/usr/bin/env python3
"""Print SUMO and sumo-rl versions, then roll out the packaged single intersection."""

import os
import subprocess
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, NoReturn

EXPECTED_SUMO_VERSION = "1.27.1"


def fail(message: str) -> NoReturn:
    print(message, file=sys.stderr, flush=True)
    raise SystemExit(1)


def resolve_sumo_home() -> str:
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        fail(
            "SUMO_HOME is not set. From the traffic environment, run "
            'conda env config vars set SUMO_HOME="$(python -c \'import sumo; print(sumo.SUMO_HOME)\')", '
            "then deactivate and activate the environment again."
        )
    if not Path(sumo_home).is_dir():
        fail(f"SUMO_HOME is not a directory: {sumo_home}")
    return sumo_home


def sumo_version(sumo_home: str) -> str:
    binary = Path(sumo_home) / "bin" / "sumo"
    if not binary.is_file():
        fail(f"SUMO binary not found at {binary}. Check that SUMO_HOME points at the 1.27.1 install.")
    completed = subprocess.run(
        [str(binary), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    text = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0 or not text:
        detail = completed.stderr.strip() or completed.stdout.strip()
        fail(f"Could not read SUMO version from {binary}. {detail}")
    line = text.splitlines()[0]
    version = line.split()[-1]
    if version != EXPECTED_SUMO_VERSION:
        fail(
            f"Wrong SUMO version: {line}. Expected {EXPECTED_SUMO_VERSION}. "
            "SUMO 1.28 needs macOS 15, so this project pins 1.27.1."
        )
    return line


def use_pettingzoo_agent_selector_class() -> None:
    """Point pettingzoo.utils.agent_selector at the class before sumo-rl imports it.

    sumo-rl 1.4.5 does ``from pettingzoo.utils import agent_selector`` and calls
    that object. PettingZoo 1.24 binds the name to the class. PettingZoo 1.25+
    binds it to the agent_selector module, so the call raises
    ``TypeError: 'module' object is not callable``.
    """
    import pettingzoo.utils as pettingzoo_utils
    from pettingzoo.utils.agent_selector import AgentSelector

    if not callable(getattr(pettingzoo_utils, "agent_selector", None)):
        pettingzoo_utils.agent_selector = AgentSelector


def single_intersection_files(sumo_rl_module) -> tuple[str, str]:
    net_dir = os.path.join(os.path.dirname(sumo_rl_module.__file__), "nets", "single-intersection")
    net_file = os.path.join(net_dir, "single-intersection.net.xml")
    route_file = os.path.join(net_dir, "single-intersection.rou.xml")
    assert os.path.isfile(net_file), f"sumo-rl net file not found: {net_file}"
    assert os.path.isfile(route_file), f"sumo-rl route file not found: {route_file}"
    return net_file, route_file


def has_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        if key in value:
            return True
        return any(has_key(item, key) for item in value.values())
    return False


def main() -> None:
    sumo_home = resolve_sumo_home()
    print(sumo_version(sumo_home), flush=True)

    # Bind the class before sumo-rl's import captures the name.
    use_pettingzoo_agent_selector_class()
    import sumo_rl
    from sumo_rl import parallel_env

    print(f"sumo-rl {sumo_rl.__version__}", flush=True)

    net_file, route_file = single_intersection_files(sumo_rl)
    warnings.filterwarnings(
        "ignore",
        message=r"The base environment `.*` does not have a `render_mode` defined\.",
    )
    # sumo_warnings=False quiets SUMO warning spam. TraCI still prints "Step #"
    # and "Retrying in 1 seconds" while the SUMO socket starts. Those lines are
    # normal startup noise and do not mean the install failed.
    # Default delta_time is 5s, so 100 decisions stay inside this horizon.
    env = parallel_env(
        net_file=net_file,
        route_file=route_file,
        num_seconds=1000,
        use_gui=False,
        sumo_warnings=False,
    )
    info: dict = {}
    steps = 0
    try:
        reset_out = env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            _, info = reset_out
        for _ in range(100):
            if not env.agents:
                break
            actions = {a: env.action_space(a).sample() for a in env.agents}
            obs, rew, term, trunc, info = env.step(actions)
            steps += 1
    finally:
        env.close()

    print(info, flush=True)
    if steps != 100:
        fail(f"Rollout stopped after {steps} steps; expected 100.")
    if not has_key(info, "system_mean_waiting_time"):
        fail("Final info dict has no system_mean_waiting_time.")
    print("INSTALL OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        print("Install check failed", file=sys.stderr, flush=True)
        sys.exit(1)
