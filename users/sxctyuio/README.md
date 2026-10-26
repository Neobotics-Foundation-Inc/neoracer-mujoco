# NeoRacer reinforcement learning

PPO driving policies for the NeoRacer, trained in MuJoCo with sim-to-real in
mind: a single car on the loop track, a single car on procedurally generated
tracks it has never seen, and 2-4 car racing with one shared policy.

Everything here is a self-contained package, `neoracer_rl`, plus two scripts.
It uses the core `neoracer_mujoco` modules (`assets`, `collision`, `contract`,
`sim`) unmodified.

## Results (GitHub car, sim-to-real randomization on)

| Policy | Test | Randomized cars | Clean cars |
|---|---|---|---|
| Racing, 2 cars, opposite starts | 100 races | 0 crashes (~600 car-laps), best lap 7.5 s | 0 crashes, 7.0 s |
| Procedural tracks, difficulty 1-2 | 100 never-seen tracks | 96/100 laps | 99/100 laps |
| Single car, loop | 50 laps | 50/50, 9.5 s from a standing start | |

Every number above is from `scripts/render_policy.py --headless` with a fixed
`--seed`, so the same tracks and cars for every model compared. How these were
reached, including what did not work, is in [RESULTS.md](RESULTS.md).

## Setup

Built and tested against an older version of this repository (Python 3.14,
MuJoCo 3.16.0), whose core modules this package uses. Current `main` requires
Python 3.15 and has a different `contract.py`, so it may need small changes
there.

```bash
pip install -e .                                  # the core package, from the repo root
pip install -r users/sxctyuio/requirements.txt
```

The default car is the GitHub car (upstream `assets/neoracer.xml`), expected
under its own name so the older car keeps its file:

```bash
git show origin/main:assets/neoracer.xml > assets/neoracer_github.xml
```

## Quick start

```bash
cd users/sxctyuio
# train (one run uses every core; --smoke for a one-minute check)
python3 scripts/train_ppo.py --sim2real --out runs/loop
python3 scripts/train_ppo.py --track procedural --difficulty 1-2 --sim2real --history 6 --out runs/proc
python3 scripts/train_ppo.py --cars 2 --start opposite --sim2real --out runs/race

# test: results only, no window
python3 scripts/render_policy.py runs/proc/final --headless --seed 3 --episodes 100 | grep -c lap_complete

# watch (macOS: mjpython for a window) or record
mjpython scripts/render_policy.py runs/race/final
python3 scripts/render_policy.py runs/race/final --record race.mp4
```

`render_policy.py` reads the run folder's `env.json` / `race.json`, so it
replays every run on the car, grip, track type, memory and race format it
trained with. `--init-from DIR` warm-starts a run from another model, also
across observation sizes (see Training).

The settings behind the results table:

```bash
python3 scripts/train_ppo.py --cars 2 --start opposite --sim2real --wall-penalty 50 --stuck-penalty 50 --rollover-penalty 50 --wall-clearance-weight 0 --timesteps 10000000 --init-from <previous> --out runs/race_gh
python3 scripts/train_ppo.py --track procedural --difficulty 1-2 --sim2real --history 6 --wall-penalty 20 --stuck-penalty 20 --rollover-penalty 20 --time-penalty 0.005 --corner-weight 0.2 --wall-clearance-weight 0 --timesteps 10000000 --init-from <previous> --out runs/proc_gh
```

## Files

```
neoracer_rl/
  envs/
    neoracer_env.py      single car on loop_corridor (Gymnasium env)
    procedural_env.py    single car on generated tracks; LiDAR corner sensing; track pool / bank
    multi_racer_env.py   2-4 car racing, one shared policy (SB3 VecEnv, one slot per car)
    sim2real.py          randomization, sensor noise, action delay; heading-free observation
    physics.py           port of upstream physics.py domain randomization to this car
    car_model.py         what differs between the GitHub car and the older car
    drive_model.py       action -> actuators; rear-blind LiDAR; floor grip; wall clearance
    history.py           memory: last N observations and actions in the observation
  sensors.py             core sensors.py, tolerant of the GitHub car's extra sensors
  track_generation/      upstream procedural track generator (Python 3.14 port: no frozendict)
scripts/
  train_ppo.py           PPO training (Stable-Baselines3); every option is a flag, see --help
  render_policy.py       watch, record or test (--headless) a trained run
RESULTS.md               every evaluation, in order, with what each experiment changed
changed_files/           the same work as in-place repository changes, at repository paths;
                         CHANGES.md lists each added and modified file
```

## How it works

**Car.** The GitHub car by default: the RS-545 DC motor and SPT5435 servo
modeled natively, 3.0 kg, 42.5 mm wheels, a 1/360 s physics step. The throttle
action is ESC voltage (+-11.1 V) and steering is the servo's target angle.
Full throttle settles at about 3.5 m/s on flat ground. On top of the car:

- Floor grip 0.8 on every surface (typical rubber on an indoor floor; the
  repository's 1.5 is untested). `--grip` sets a measured value.
- LiDAR blind straight back, matching the driver's 45-315 deg scan window.
- `--vehicle realistic` / `legacy` select the older car, with or without this
  project's approximate drive model.

**Observation.** 8 LiDAR ranges, IMU accelerometer and gyro, steering
position, wheel speed, and orientation: 20 values, or 19 with sim-to-real,
which replaces the world-frame quaternion with gravity in the body frame so a
policy cannot steer by absolute heading. Racing adds the car's place (21/20).
`--history N` appends the last N observations and actions.

**Reward.** Progress around the track (per meter of centerline on generated
tracks), a corner bonus for tight lines, a per-step time penalty, and
penalties for wall contact, rollover, getting stuck and hitting walls.
Options: `--wall-clearance-weight` (cost for running within 8 cm of a wall),
`--smoothness-weight` (cost for changing the commands step to step),
`--time-penalty-end` (ramps the speed incentive up during training).

**Sim-to-real.** Per episode (per car in races): the upstream randomization of
wheel friction, chassis mass, motor torque and steering stiffness, plus
steering trim and 0-1 steps of action delay; per step, sensor noise, LiDAR
dropout and action noise. Ranges are in `physics.GENTLE_RANGES` and
`sim2real.Sim2RealConfig`; upstream ships no recommended ranges.

**Procedural tracks.** Each worker keeps a pool of generated tracks and swaps
one out every 20 episodes, about 1000 distinct tracks over a 10M-step run
(`--track-bank N` trains on a saved, pre-generated set instead). Tests use a
fresh track every episode. The corner bonus is computed from LiDAR alone: the
circle through each side's three wall-facing beams gives that wall's
curvature (turn direction right in 98% of corners, closeness correlates 0.86
with the true apex distance over 3000 poses).

**Racing.** One policy drives every car. `--start grid` (2-4 cars side by
side) or `--start opposite` (2 cars half a lap apart, RoboRacer format).
Winning is by time over the same distance. A car that crashes alone respawns
at the start while the race goes on; a car-to-car crash costs the car at
fault more than the car it hit, and does not end the race. Each car tracks
its place from its own LiDAR (a rival seen front -> side -> rear is a pass).

**Training.** Stable-Baselines3 PPO with observation and reward normalization,
linear learning-rate decay and a KL limit. `--init-from` copies a policy,
its value function and both normalizers; when the new observation is longer
(memory, place), the extra inputs start at zero weight, so the warm-started
policy first drives exactly like its source.

## Open questions for the real car

- **Top speed.** The website says 25 km/h (6.9 m/s); the documented motor
  gives about 4.3 m/s free speed, and the sim car tops out near 3.5 m/s. If
  the real car is much faster, the motor model is too weak. Measure it on a
  straight (`/odom`).
- **Floor grip.** Pull the car sideways with a spring scale: force / weight.
  Training covers 0.6-1.0.
- **Control delay.** Bag `/drive` and `/imu`, time a steering step. Training
  covers 0-1 control steps (0-33 ms).
