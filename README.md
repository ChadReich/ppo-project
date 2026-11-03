# PPO from scratch: comparison with Stable-Baselines3

Code and results for my MSc research report *Advanced Topics in Reinforcement Learning: Proximal Policy Optimisation* (Stellenbosch University, 2026), supervised by Prof Hugo Touchette.

I implemented proximal policy optimisation (PPO) from scratch in PyTorch and compared it with Stable-Baselines3 (SB3) on `HalfCheetah-v5` (primary benchmark) and `Pendulum-v1` (supplementary benchmark), using five training seeds per implementation and matched high-level hyperparameters.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The exact versions used for the report are listed in `versions.txt`. All runs used the CPU.

## Files

### Implementation

| File | Purpose |
|---|---|
| `ppo.py` | Scratch PPO: discrete and continuous (diagonal Gaussian) actors, critic, rollout storage, GAE and the PPO update. |

### Development and checks

| File | Purpose |
|---|---|
| `check_envs.ipynb`, `check_halfcheetah.py` | Inspect environment observation and action spaces. |
| `check_actor_critic.ipynb`, `check_actor_critic_updated.ipynb` | Check the actor and critic networks and their outputs. |
| `check_continuous_ppo.ipynb` | Check the continuous policy, log-probabilities and PPO ratio. |
| `train_cartpole.py` | Discrete development run on `CartPole-v1` (Table 5.1). |
| `train_pendulum.py` | Initial continuous pilot on `Pendulum-v1` (Table 5.1). |
| `train_halfcheetah.py`, `train_halfcheetah_sb3.py` | Original fixed-protocol `HalfCheetah-v5` runs for scratch PPO and SB3. |

### Fixed-protocol comparison

| File | Purpose |
|---|---|
| `run_halfcheetah_frozen.py` | Runs all five seeds of scratch PPO and SB3 on `HalfCheetah-v5`, alternating by seed, and records training time. |
| `train_halfcheetah_frozen_scratch_seed.py`, `train_halfcheetah_frozen_sb3_seed.py` | One `HalfCheetah-v5` training seed for each implementation (called by the runner). |
| `run_pendulum_frozen.py` | The same for `Pendulum-v1`. |
| `train_pendulum_frozen_scratch_seed.py`, `train_pendulum_frozen_sb3_seed.py` | One `Pendulum-v1` training seed for each implementation. |
| `analyse_halfcheetah.py` | Final-window summaries, late-training diagnostic summaries and the two `HalfCheetah` figures. |

Run a comparison from the repository root, for example:

```bash
python run_halfcheetah_frozen.py            # resumes from existing results
python run_halfcheetah_frozen.py --restart  # deletes existing results and starts from seed 0
python analyse_halfcheetah.py
```

**Naming:** files marked *frozen* are the final runs with the fixed settings described in Section 5.6 of the report.

## Results and where they appear in the report

| File(s) in `results/` | Report |
|---|---|
| `halfcheetah_scratch.csv`, `halfcheetah_sb3.csv` | Figure 5.1, Table 5.5, normalised AUC in Table 5.6 |
| `halfcheetah_scratch_diagnostics.csv`, `halfcheetah_sb3_diagnostics.csv` | Figure 5.2, Section 5.8 |
| `halfcheetah_*_frozen_runtime.csv` | `HalfCheetah` training time and throughput in Table 5.6 |
| `pendulum_*_frozen.csv` | `Pendulum` normalised AUC in Table 5.6 |
| `pendulum_*_frozen_diagnostics.csv` | `Pendulum` diagnostics in Section 5.8 |
| `pendulum_*_frozen_runtime.csv` | `Pendulum` training time and throughput in Table 5.6 |

The timed `HalfCheetah` rerun reproduced the evaluation trajectories and diagnostics of the original runs exactly for all five seeds, so the evaluation and diagnostic files above are the ones used in the report. The rerun's runtime files supply the timing.

Figures are saved in `figures/`.
