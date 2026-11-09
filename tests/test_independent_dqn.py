from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.agents.independent_dqn import (  # noqa: E402
    DQNAgent,
    ReplayBuffer,
    epsilon_at,
    policy_factory,
    save_model,
)

CFG = {"lr": 1e-3, "gamma": 0.99, "buffer": 100, "batch": 8, "target_update": 500, "final_eps": 0.01}


def test_buffer_sample_shapes_and_wrap():
    buf = ReplayBuffer(capacity=100, obs_dim=4)
    for i in range(150):
        buf.push(np.full(4, i), i % 2, 1.0, np.full(4, i + 1), 0.0)
    assert len(buf) == 100
    batch = buf.sample(8, np.random.default_rng(0))
    assert batch["obs"].shape == (8, 4)
    assert batch["next_obs"].shape == (8, 4)
    assert batch["actions"].shape == (8,)
    assert batch["rewards"].shape == (8,)
    assert batch["dones"].shape == (8,)


def test_epsilon_schedule():
    assert epsilon_at(0, 1000, 0.01) == pytest.approx(1.0)
    assert epsilon_at(300, 1000, 0.01) == pytest.approx(0.01)
    assert epsilon_at(900, 1000, 0.01) == pytest.approx(0.01)
    assert 0.01 < epsilon_at(150, 1000, 0.01) < 1.0


def test_update_reduces_loss_on_toy_batch():
    torch.manual_seed(0)
    agent = DQNAgent(3, 2, CFG, np.random.default_rng(0))
    rng = np.random.default_rng(1)
    batch = {
        "obs": rng.normal(size=(16, 3)).astype(np.float32),
        "actions": rng.integers(0, 2, size=16),
        "rewards": rng.normal(size=16).astype(np.float32),
        "next_obs": rng.normal(size=(16, 3)).astype(np.float32),
        "dones": np.ones(16, dtype=np.float32),
    }
    first = agent.learn(batch)
    for _ in range(200):
        last = agent.learn(batch)
    assert last < first


def test_saved_model_gives_greedy_actions(tmp_path):
    agents = {"A": DQNAgent(3, 2, CFG, np.random.default_rng(0)), "B": DQNAgent(3, 4, CFG, np.random.default_rng(1))}
    path = tmp_path / "model.pt"
    save_model(agents, path)
    policy = policy_factory(path)
    obs = {"A": np.zeros(3, dtype=np.float32), "B": np.ones(3, dtype=np.float32)}
    actions = policy(obs, SimpleNamespace(agents=["A", "B"]))
    assert actions["A"] in (0, 1)
    assert actions["B"] in (0, 1, 2, 3)


@pytest.mark.sumo
def test_one_episode_on_single():
    pytest.importorskip("sumo_rl")
    from src.agents.independent_dqn import train
    from src.utils.config import load_config

    model_file, curve_file = train("single", 0, 1, load_config())
    assert model_file.is_file() and curve_file.is_file()
