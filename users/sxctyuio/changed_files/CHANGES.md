# Changed repository files

The exact files this project added to or changed in the repository, at their
repository paths. They are what the training runs in RESULTS.md used.

`neoracer_rl/` one level up holds the same code as a self-contained package
that leaves the repository untouched (only import paths differ). Use that to
run the project; this folder shows what it changes in place.

Built on an older version of the repository (Python 3.14, MuJoCo 3.16.0).
Copying these over current `main` will not work as is: `main` needs Python
3.15 and has a different `contract.py` and `sensors.py`.

## Modified

| File | Change |
|---|---|
| `src/neoracer_mujoco/envs/neoracer_env.py` | `TypedDict` from `typing_extensions` (Python 3.14); `RewardConfig` gains a corner bonus, `wall_penalty` and a wall-clearance cost; `TerminationConfig.terminate_on_wall`; actions go through `drive_model` and `car_model` (GitHub car support, rear-blind LiDAR, floor grip); car-specific physics step and wheel radius |
| `src/neoracer_mujoco/envs/__init__.py` | exports `ProceduralRacerEnv` and `ProceduralEnvConfig` |
| `src/neoracer_mujoco/sensors.py` | `read_layout` accepts layouts with extra sensors (the GitHub car adds some, e.g. `drive_pos`) |

## Added

| File | What it is |
|---|---|
| `src/neoracer_mujoco/envs/procedural_env.py` | single car on generated tracks: centerline progress, LiDAR corner sensing, track pool, optional saved track bank |
| `src/neoracer_mujoco/envs/multi_racer_env.py` | 2-4 car racing as an SB3 VecEnv (one slot per car): grid and opposite starts, respawn, fault-based collisions, LiDAR place counting, lap-time winners |
| `src/neoracer_mujoco/envs/sim2real.py` | randomization (via physics.py), sensor noise, LiDAR dropout, action delay, heading-free observation |
| `src/neoracer_mujoco/envs/physics.py` | port of upstream `physics.py` domain randomization (wheel friction, chassis mass, motor torque, steering stiffness) to this car |
| `src/neoracer_mujoco/envs/car_model.py` | what differs between the GitHub car and the older car (actuators, physics step, wheel radius) |
| `src/neoracer_mujoco/envs/drive_model.py` | action -> actuators for both cars, rear-blind LiDAR, floor grip, wall clearance |
| `src/neoracer_mujoco/envs/history.py` | memory wrappers: last N observations and actions |
| `src/neoracer_mujoco/track_generation/` | the upstream track generator, ported to Python 3.14 (`MappingProxyType` instead of `frozendict` in `config.py`) |
| `examples/train_ppo.py` | PPO training for every mode; all options are flags (`--help`) |
| `examples/render_policy.py` | watch, record or test (`--headless`) a trained run |

## Not included

`assets/neoracer_github.xml`: the upstream car, unchanged, saved under that
name so the older car keeps its file:

```bash
git show origin/main:assets/neoracer.xml > assets/neoracer_github.xml
```
