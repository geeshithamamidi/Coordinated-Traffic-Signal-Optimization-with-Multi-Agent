"""Neighbor observations keep a fixed size and mark real neighbors with a presence flag."""

from pathlib import Path

import numpy as np
import sumo_rl

from src.envs.coordination import NeighborObsWrapper, SpatialRewardWrapper
from src.envs.make_env import make_env
from src.envs.neighbors import build_neighbor_map

_GRID_NET = Path(sumo_rl.__file__).resolve().parent / "nets" / "RESCO" / "grid4x4" / "grid4x4.net.xml"
_CORNERS = ("A0", "A3", "D0", "D3")


def _presence_flags(observation: np.ndarray, own_dim: int, k_max: int, max_phases: int) -> np.ndarray:
    slot_size = 2 + max_phases + 1
    presence_index = slot_size - 1
    return np.array(
        [observation[own_dim + slot * slot_size + presence_index] for slot in range(k_max)],
        dtype=np.float32,
    )


def test_grid4x4_neighbor_obs_dimension_and_corner_flags():
    env = make_env("grid4x4", seed=0, additional_sumo_cmd="--no-step-log --no-warnings")
    try:
        neighbor_map = build_neighbor_map(str(_GRID_NET), list(env.possible_agents))
        wrapped = NeighborObsWrapper(env, neighbor_map, k_max=4)
        observations, _infos = wrapped.reset(seed=0)

        for agent, observation in observations.items():
            own_dim = int(np.prod(env.observation_space(agent).shape))
            expected = own_dim + wrapped.k_max * (2 + wrapped.max_phases + 1)
            assert observation.shape == (expected,)
            assert wrapped.observation_space(agent).shape == (expected,)
            assert observation.dtype == np.float32

        for corner in _CORNERS:
            own_dim = int(np.prod(env.observation_space(corner).shape))
            flags = _presence_flags(observations[corner], own_dim, wrapped.k_max, wrapped.max_phases)
            assert int(np.sum(flags == 1.0)) == 2
    finally:
        env.close()


class _FixedRewardEnv:
    """Parallel env with a fixed local reward dict."""

    def __init__(self, rewards: dict[str, float]):
        self.rewards = rewards
        self.agents = list(rewards)
        self.possible_agents = list(rewards)

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs = {agent: np.zeros(1, dtype=np.float32) for agent in self.agents}
        return obs, {}

    def step(self, actions):
        obs = {agent: np.zeros(1, dtype=np.float32) for agent in self.agents}
        flags = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return obs, dict(self.rewards), flags, flags, infos

    def close(self):
        pass


_LINE = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}
_LOCAL = {"A": 1.25, "B": -3.0, "C": 0.5}


def _step_rewards(alpha: float, **kwargs) -> dict:
    env = _FixedRewardEnv(_LOCAL)
    wrapped = SpatialRewardWrapper(env, _LINE, alpha, **kwargs)
    wrapped.reset()
    _obs, rewards, _terms, _truncs, _infos = wrapped.step({})
    return rewards


def test_spatial_reward_alpha_zero_returns_original_rewards():
    for kwargs in ({}, {"hop_decay": True}, {"hop_decay": True, "D": 2}):
        rewards = _step_rewards(0, **kwargs)
        assert rewards == _LOCAL
        for agent, reward in _LOCAL.items():
            assert rewards[agent] is _LOCAL[agent]


def test_spatial_reward_one_hop_and_hop_decay():
    one_hop = _step_rewards(0.5)
    assert one_hop == {"A": 1.25 + 0.5 * -3.0, "B": -3.0 + 0.5 * (1.25 + 0.5), "C": 0.5 + 0.5 * -3.0}

    decay = _step_rewards(0.5, hop_decay=True, D=2)
    assert decay["A"] == 1.25 + 0.5 * -3.0 + 0.25 * 0.5
    assert decay["B"] == -3.0 + 0.5 * 1.25 + 0.5 * 0.5
    assert decay["C"] == 0.5 + 0.5 * -3.0 + 0.25 * 1.25
