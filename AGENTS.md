# Coordinated Traffic Signal Optimization with Multi-Agent

## Environment

Use the conda env `traffic` with Python 3.11. SUMO is pinned at 1.27.1 (`eclipse-sumo`, `sumo-data`, `traci`, `sumolib`). SUMO 1.28 needs macOS 15. Set `SUMO_HOME` with `conda env config vars set SUMO_HOME="$(python -c 'import sumo; print(sumo.SUMO_HOME)')"` and reactivate the env. `hpc/train_array.slurm` sets `LIBSUMO_AS_TRACI=1`. Import `sumo-rl`; do not copy its source into this repo. `requirements.txt` pins `sumo-rl==1.4.5` and `pettingzoo==1.24.3`.

## Code map

- `src/envs/make_env.py` builds every scenario with the shared 3600 s protocol.
- `src/envs/neighbors.py` builds the signal neighbor graph. `src/envs/coordination.py` adds `NeighborObsWrapper` and `SpatialRewardWrapper`.
- `src/agents/shared_ppo.py` trains one shared PPO policy. `src/agents/coordinated_ppo.py` applies the neighbor and spatial-reward wrappers before the SuperSuit stack. `src/agents/independent_dqn.py` and `src/agents/independent_ppo.py` train one network per intersection.
- `src/baselines/` holds fixed-time, max-pressure, and independent Q-learning.
- `src/evaluation/metrics.py` scores a policy. `src/evaluation/stats.py` summarizes results.
- `src/utils/config.py` loads `configs/base.yaml`. `configs/frozen.yaml` stores the spatial-reward alpha chosen on validation seeds. `configs/sweeps/` holds HPC sweep files.
- `scripts/check_install.py`, `scripts/run_experiment.py`, `scripts/run_baselines.py`, `scripts/make_jobs.py`, `scripts/train_shared_ppo.py`, `scripts/train_coordinated_ppo.py`, and `scripts/aggregate_results.py` are the entry points.
- `hpc/` has the SLURM array, env setup, and `jobs.txt`. `notebooks/` explores the env, baselines, and PPO. `results/` and `report/` hold figures, tables, and the write-up.

## Rules

`make_env` uses `num_seconds=3600`, `delta_time=5`, `yellow_time=2`, and `min_green=5`. Route seeds are train 0–4, validation 5–9, and test 10–14. Choose the spatial-reward alpha on validation seeds only and record it in `configs/frozen.yaml`. Coordinated PPO ablations are shared PPO (neighbor obs off, alpha 0), obs only (obs on, alpha 0), reward only (obs off, alpha > 0), and coordinated (obs on, frozen alpha).

## How to run

```bash
conda activate traffic
pip install -r requirements.txt
python scripts/check_install.py
pytest -q
```

`python scripts/check_install.py` should print SUMO 1.27.1, the sumo-rl version, and a final info dict that includes `system_mean_waiting_time`, then `INSTALL OK`. TraCI may print `Step #` and `Retrying in 1 seconds` while SUMO starts. Those lines are normal.
