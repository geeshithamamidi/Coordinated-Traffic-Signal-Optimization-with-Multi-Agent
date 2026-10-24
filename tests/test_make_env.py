"""Smoke tests for the shared environment factory."""

from src.envs.make_env import make_env


def _rollout(scenario: str, expected_agents: int | None = None) -> None:
    env = make_env(scenario, seed=0, additional_sumo_cmd="--no-step-log")
    try:
        env.reset()
        if expected_agents is not None:
            assert len(env.agents) == expected_agents
        for _ in range(10):
            assert env.agents, f"{scenario} ended before 10 steps"
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            env.step(actions)
    finally:
        env.close()


def test_single_steps():
    _rollout("single")


def test_grid4x4_has_16_agents_and_steps():
    _rollout("grid4x4", expected_agents=16)
