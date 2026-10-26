# SJSU CoE HPC

Train on the College of Engineering cluster. Do the heavy runs through SLURM. The login node is only for editing, installing, and `sbatch`.

Account access is requested by the course instructor. The cluster is reachable on SJSU_premier, a campus Ethernet port, or the [SJSU VPN](https://www.sjsu.edu/it/services/network/vpn/index.php). The department's access notes are at [College of Engineering HPC](https://www.sjsu.edu/cmpe/resources/hpc.php).

## Log in

```bash
ssh SJSU_ID@coe-hpc.sjsu.edu
```

If that times out over VPN, use `ssh SJSU_ID@coe-hpc1.sjsu.edu`. The password is the SJSUOne password. Accept the host key the first time.

## Environment

On the login node, load Anaconda and create `traffic`:

```bash
module avail anaconda
module load anaconda/3.9
git clone https://github.com/geeshithamamidi/Coordinated-Traffic-Signal-Optimization-with-Multi-Agent.git
cd Coordinated-Traffic-Signal-Optimization-with-Multi-Agent
bash hpc/setup_env.sh
conda deactivate
conda activate traffic
```

`hpc/setup_env.sh` creates a Python 3.11 env, installs `requirements.txt` (SUMO 1.27.1 and the rest of the pinned stack), and installs `libsumo==1.27.1`. It then stores two conda environment variables:

- `SUMO_HOME` from `import sumo; print(sumo.SUMO_HOME)`
- `LIBSUMO_AS_TRACI=1`

They are exported the next time you `conda activate traffic`. If `module avail` does not list `anaconda/3.9`, load the Anaconda module name it does list, then run the setup script.

Check the install on the login node with a short run, not a training job:

```bash
python scripts/check_install.py
```

## Jobs

`hpc/experiments.yaml` lists methods, scenarios, demands, seeds, and the training episode budget. Regenerate the array input after editing it:

```bash
python scripts/make_jobs.py
mkdir -p hpc/logs
```

That writes `hpc/jobs.txt`, one line per run: `method scenario demand seed`.

Before submitting, set the two TODO fields in `hpc/train_array.slurm`:

- `#SBATCH --partition=TODO_PARTITION` — partition name for this course. `sinfo` prints the queues.
- `#SBATCH --account=TODO_ACCOUNT` — SLURM account for this course allocation.

Each array task is 1 CPU, 4 GB RAM, and 8 hours. Logs go to `hpc/logs/<jobid>_<task>.out`. Submit from the repository root so those relative paths resolve:

```bash
sbatch --array=1-$(wc -l < hpc/jobs.txt) hpc/train_array.slurm
```

Task `k` reads line `k` of `hpc/jobs.txt` and runs `scripts/run_experiment.py` with that method, scenario, demand, and seed. The script trains, then scores the saved policy on the test seeds in `configs/base.yaml`.
