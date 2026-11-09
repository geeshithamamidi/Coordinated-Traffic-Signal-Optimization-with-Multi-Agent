from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.agents.independent_ppo import PPOAgent, compute_gae, policy_factory, save_model  # noqa: E402

CFG = {"lr": 3e-4, "gamma": 0.99, "gae_lambda": 0.95, "clip": 0.2, "n_steps": 16, "n_epochs": 2, "batch": 8}


def test_gae_hand_computed():
    # gamma*lambda = 0.72, every delta = 1 + 0.9*0.5 - 0.5 = 0.95
    adv, ret = compute_gae([1, 1, 1], [0.5, 0.5, 0.5], [0, 0, 0], 0.5, gamma=0.9, lam=0.8)
    np.testing.assert_allclose(adv, [2.12648, 1.634, 0.95], rtol=1e-5)
    np.testing.assert_allclose(ret, adv + 0.5, rtol=1e-6)


def test_gae_stops_at_episode_end():
    adv, _ = compute_gae([1, 1, 1], [0, 0, 0], [0, 1, 0], 10.0, gamma=1.0, lam=1.0)
    np.testing.assert_allclose(adv, [2.0, 1.0, 11.0])


def test_update_runs_and_clears_buffer():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    agent = PPOAgent(3, 2, CFG, rng)
    for i in range(16):
        obs = rng.normal(size=3).astype(np.float32)
        action, logp, value = agent.act(obs)
        agent.buffer.add(obs, action, logp, 1.0, float(i == 15), value)
    assert agent.buffer.full
    agent.update(np.zeros(3, dtype=np.float32))
    assert agent.buffer.size == 0


def test_saved_model_gives_argmax_actions(tmp_path):
    rng = np.random.default_rng(0)
    agents = {"A": PPOAgent(3, 2, CFG, rng), "B": PPOAgent(3, 4, CFG, rng)}
    path = tmp_path / "ippo.pt"
    save_model(agents, path)
    policy = policy_factory(path)
    obs = {"A": np.zeros(3, dtype=np.float32), "B": np.ones(3, dtype=np.float32)}
    env = SimpleNamespace(agents=["A", "B"])
    first, second = policy(obs, env), policy(obs, env)
    assert first == second
    assert first["A"] in (0, 1) and first["B"] in (0, 1, 2, 3)


@pytest.mark.sumo
def test_one_episode_on_single():
    pytest.importorskip("sumo_rl")
    from src.agents.independent_ppo import train
    from src.utils.config import load_config

    model_file, curve_file = train("single", 0, 1, load_config())
    assert model_file.is_file() and curve_file.is_file()
