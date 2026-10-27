import numpy as np

from src.baselines.independent_ql import encode_obs


def test_encode_obs_keeps_phase_and_bins_values():
    obs = np.array([0, 1, 1, 0.0, 0.05, 0.55, 1.0], dtype=np.float32)
    assert encode_obs(obs, n_phases=2) == (0, 1, 1, 0, 0, 5, 9)


def test_encode_obs_is_hashable():
    state = encode_obs(np.zeros(6), n_phases=2)
    assert hash(state) == hash(state)
