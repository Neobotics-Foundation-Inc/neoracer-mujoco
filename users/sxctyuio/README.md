# NeoRacer reinforcement learning

PPO driving policies for the NeoRacer, trained in MuJoCo with sim-to-real in
mind: a single car on the loop track, a single car on generated tracks it has
never seen, and 2-4 car racing with one shared policy.

The simulator is neoracer-rl's race env, `NeoRacerRace`, on the current
neoracer-mujoco car: the cars, their sensors, the firmware's speed loop, the
physics randomization, the generated tracks and the per-car progress all come
from there. What is this project's is the task, the policy's side of
sim-to-real, and the training:

```
racing/
  task.py        the observation, the ends and the reward, as functions of a Step
  setup.py       a run's settings (Setup) and the RaceConfig they make: the grid, the physics ranges
  vec_env.py     NeoRacerRace as a Stable-Baselines3 VecEnv, one env per car
  sim2real.py    sensor noise and biases, LiDAR dropout, action noise, trim and delay
  history.py     memory: the last N observations and actions in the observation
scripts/
  train_ppo.py      PPO training (Stable-Baselines3); a flag for each field of Setup
  render_policy.py  score (--headless), watch or record a trained run
RESULTS.md       every evaluation of the earlier implementation, in order
```

## Setup

Python 3.14. neoracer-rl brings the neoracer-mujoco wheel it pins, so the
repository's own package need not be installed:

```bash
pip install -r users/sxctyuio/requirements.txt
```

## Quick start

```bash
cd users/sxctyuio
python3 scripts/train_ppo.py --sim2real --out runs/loop
python3 scripts/train_ppo.py --track procedural --difficulty 2 --sim2real --history 6 --out runs/proc
python3 scripts/train_ppo.py --cars 2 --sim2real --out runs/race

python3 scripts/render_policy.py runs/race --headless --races 100   # how each race ended
python3 scripts/render_policy.py runs/proc --headless --track-shift 1000  # on unseen tracks
mjpython scripts/render_policy.py runs/race                          # watch (mjpython on macOS)
python3 scripts/render_policy.py runs/race --record race.mp4
```

A run saves its `Setup` as `setup.json` beside the policy, so
`render_policy.py` replays it on the car, track, race and memory it trained
with.

## How it works

**Car.** neoracer-mujoco's NeoRacer, whose program asks its firmware's speed
loop for a speed. Its LiDAR sees the other cars, whose outline is their
collision mesh, and it collides with them.

**Observation** (`task.HeadingFree`): the 8 LiDAR ranges, the IMU's
acceleration and turn rate, gravity in the body frame in place of the
world-frame quaternion, so that a policy cannot steer by its heading, the
steering angle and the encoder's speed: 19 values. A race adds the car's
place, 0 when it leads and 1 when it is last (20). `--history N` appends the
last N observations and actions.

**Reward** (`task.RaceReward`): progress around the track, the one-time lap
bonus and the crash, stuck and rollover penalties, a per-step time penalty,
`--overtake-reward` for each car passed (less as much for each pass suffered),
and at a car-to-car collision `--collision-penalty` for the car behind and
`--victim-share` of it for the car ahead.

**Sim-to-real.** `--sim2real` draws each car's physics at each spawn from
`setup.GENTLE`: wheel friction, chassis mass, motor torque and steering
stiffness. `sim2real.py` adds, per episode and car, a steering trim, IMU
biases, a wheel-speed scale error and 0-1 steps of action delay, and per
step, sensor noise, LiDAR dropout and action noise. The ranges are first
guesses.

**Procedural tracks.** `--track procedural` trains on a pool of `--pool`
generated tracks of `--difficulty`, from seed `--first-seed` on, and each
race draws its track at each spawn. `render_policy.py --track-shift N`
races a run on the tracks of the seeds N on from its own.

**Racing.** One policy drives every car (`Each` agents): each car is an env
of the VecEnv. 2-4 cars start on a grid at the track's start, in rows of two,
`--grid-lane` apart across the track and `--grid-row` between rows. A race
ends for every car when a car crashes, rolls over, gets stuck, collides with
another car or finishes `--laps` laps; the cars whose episode it did not end
are truncated, so PPO bootstraps their value, and the race starts over. Every
1024 steps each race draws a new spawn, track and physics.

## What changed from the earlier implementation

The earlier implementation kept its own copies of the env, the sensors, the
physics randomization, the track generator and the car models: about 5,000
lines with its scripts, and 5,100 more in changed_files/. This one is about
700 lines on neoracer-rl, and some rules differ:

- A car that crashes ends the race for every car, instead of respawning while
  the race goes on.
- The place is the true place, from neoracer-rl's `Step.rivals`, not the place
  that a car counts from its own LiDAR beams.
- The car behind is at fault in a collision, rather than the car with the
  higher speed toward the other.
- Not ported yet, each a function of a `Step` or a VecEnv wrapper: opposite
  starts (they need a heading for a car's entry in `assets.Entry`), the corner
  bonus, the near-miss and wall-clearance costs, the smoothness cost and its
  time-penalty ramp, `--init-from`, track banks, and the older car models.

The numbers in RESULTS.md came from the earlier implementation, on the earlier
car, so they need to be measured again here.

## Open questions for the real car

- **Top speed.** The website says 25 km/h (6.9 m/s); the documented motor
  gives about 4.3 m/s free speed, and the sim car tops out near 3.5 m/s. If
  the real car is much faster, the motor model is too weak. Measure it on a
  straight (`/odom`).
- **Floor grip.** Pull the car sideways with a spring scale: force / weight.
  Training covers a quarter either side of the model's.
- **Control delay.** Bag `/drive` and `/imu`, time a steering step. Training
  covers 0-1 control steps (0-33 ms).
