# NeoRacer MuJoCo Simulator

Clean, team-facing MuJoCo baseline for the NeoRacer 1/12-scale RC car.

## Setup

The mesh STLs in `assets/meshes/` are stored with [Git LFS](https://git-lfs.com). Run `git lfs install` once before cloning, or you'll get 132-byte pointer files instead of the real meshes.

Dependencies are pinned in `requirements.txt` (mujoco, numpy, pytest). Pick one:

### conda

```sh
conda env create -f environment.yml   # creates the "neoracer-mujoco" env
conda activate neoracer-mujoco
```

### uv

```sh
uv venv --python 3.13
uv pip install -r requirements.txt
source .venv/bin/activate
```

Then install the package (editable), so `neoracer_mujoco` is importable by the
examples and the validation suite:

```sh
pip install -e .
```

Install from a clone, not PyPI: the package is intentionally not published, and
the wheel does not bundle `assets/` — the model XML and meshes are the product
and stay at the repo root, so `load()` resolves them from your checkout.

## Quick start

```sh
# Run the model in the viewer (macOS needs mjpython for the passive viewer):
mjpython -m examples.run

# Run the validation suite (any Python with mujoco + pytest installed):
python3 -m pytest validation/ -v
```

## Directory layout

```
.
├── README.md              — this file
├── assets/
│   ├── neoracer.xml       — MuJoCo MJCF model (the source of truth)
│   └── meshes/            — visual STL meshes (cosmetic; mass="0")
├── src/neoracer_mujoco/   — importable package (the reusable toolbox)
│   ├── contract.py        — the car contract (single source of truth)
│   ├── assets.py          — cars() / tracks() / load() discovery + compose()
│   ├── sensors.py         — SensorReadings/IMUReading/LidarScan + read()
│   ├── sim.py             — compile/settle/run + physics probes
│   ├── control/           — classical controllers (track_centering)
│   └── track_generation/  — procedural racetracks (generate_track → Track → MJCF)
├── examples/
│   ├── run.py                  — viewer launch script (mjpython entry point)
│   ├── manual_drive.py         — arrow-key teleop (game-style, python3 entry point)
│   ├── track_centering_demo.py — PD centering controller on the corridor track
│   └── track_generation.py     — generate procedural tracks, describe or drive one
├── validation/            — pytest physics + logic conformance suite
└── docs/                  — (reserved) design notes and parameter log
```

## Model summary (`assets/neoracer.xml`)

| Property | Value | Source |
|---|---|---|
| Wheelbase | 288 mm | URDF (joint origins) |
| Track width | 235 mm | URDF (joint origins) |
| Wheel radius | 50 mm | ESTIMATED — STL bbox |
| Chassis mass | 1.62 kg | URDF (base_link) |
| Chassis CoM | ~9 mm ahead of axle center, z≈0.067 m | URDF, re-anchored to axle plane |
| Chassis inertia | diag (0.00238, 0.00584, 0.00701) kg·m² | URDF (base_link) |
| Wheel mass | ~0.058 kg each | URDF — verify vs real wheel+tyre |
| Total mass | 1.86 kg | URDF |
| Steering | Ackermann (4th-order polynomial equality constraints) | |
| Drive | AWD — 4 independent torque motors | |
| Suspension | Coil-over approximation (slide Z joint, k=600 N/m, c=25 Ns/m) | ESTIMATED |
| Sensors | IMU (accel/gyro/quat/vel) · steer · wheel vel · susp pos · 8-beam LiDAR | |

### Actuator interface (`ctrl[0:4]` drive + `ctrl[4]` steer)

| Index | Name | Unit | Direction |
|---|---|---|---|
| ctrl[0] | fl_motor | N·m | + = forward |
| ctrl[1] | fr_motor | N·m | + = forward |
| ctrl[2] | rl_motor | N·m | + = forward |
| ctrl[3] | rr_motor | N·m | + = forward |
| ctrl[4] | steer_servo | rad | + = left turn |

Steering command range is ±0.4 rad; the two Ackermann equality constraints split
it into the correct inner/outer front-wheel angles automatically.

## Track generation (`src/neoracer_mujoco/track_generation/`)

Two calls are the whole API — one makes the geometry, the other puts the car on it:

```python
from neoracer_mujoco import compose, generate_track

track = generate_track(seed=7, difficulty=2)   # pure geometry, no MuJoCo
model = compose(track)                         # a compiled MjModel, car included

model = compose("straight_corridor")           # or a hand-written assets/tracks/ XML
```

`generate_track(seed, difficulty)` returns a `Track`: N evenly spaced samples
around a closed loop, each with a centerline position, tangent, left normal,
curvature, and corridor half-width, plus the loop's total length. Same seed, same
track, always. `difficulty` runs 0 (gentle and wide) to 3 (tight and narrow); pass
a full `TrackSettings` instead if you want every knob.

Generation draws a candidate, checks it for corners tighter than the car can take,
a loop that doubles back and touches itself, and a total length outside the target
range — then nudges the control points behind each complaint and re-checks, rather
than redrawing the whole loop. A candidate that can't be repaired is thrown away
and a fresh one drawn; `TrackGenerationError` means the limits are asking for
something the generator can't draw.

Five files, split by pipeline stage:

| File | Role |
|---|---|
| `track.py` | the `Track` itself and the attempt/repair loop — **read this first** |
| `shape.py` | control points → spline → even resample → corridor width |
| `reject.py` | what makes a candidate invalid, and how to nudge it |
| `config.py` | the knobs (`TrackSettings`, `settings_for_difficulty`) |
| `mjcf.py` | `to_mjcf(track)` → scenery-only walls + floor; the only module here that touches MuJoCo |

Everything except `mjcf.py` is plain NumPy/SciPy with no simulator assumptions, so
the geometry is usable outside MuJoCo. `compose()` lives in `assets.py` rather than
here because it also serves hand-written tracks that never touch the generator.

## Usage / demo scripts (`examples/`)

`examples/` holds runnable, hackable demos — start here to drive the car yourself
or to read sensors. They are meant to be copied and modified, not imported as a
stable API.

- **`run.py`** — launches the MuJoCo viewer and drives the car with a built-in
  demo controller (constant throttle + sinusoidal steering) so you can watch it
  move. Flip `SAFE_MODE` at the top: `True` caps speed and warns on rollover,
  `False` uses aggressive inputs. Run with `mjpython -m examples.run`.
- **`manual_drive.py`** — drive the car yourself with the arrow keys, video-game
  style: it owns a glfw window and polls held keys each frame, so you hold to build
  speed, release to coast down, and let the wheel re-center. A back-top chase camera
  follows the car's heading. Use this to *feel* the physics — grip, rollover,
  suspension, wall contacts. Run with plain `python3 -m examples.manual_drive` (NOT
  mjpython — the glfw window must be created on the main thread). Camera and control
  feel are tunable via the constants at the top of the file.
- **`track_centering_demo.py`** — runs the classical PD `TrackCenteringController`
  on the straight corridor track, headless by default or with an interactive
  viewer. Composes `neoracer.xml` onto `assets/tracks/straight_corridor.xml` at
  runtime, settles, then drives the controller and reports centering performance.
  `python3 -m examples.track_centering_demo` (headless) or
  `mjpython -m examples.track_centering_demo --viewer`.

- **`track_generation.py`** — generates procedural tracks and reports what came
  out (loop length, corridor width, tightest corner, geom count) across seeds and
  difficulties. `--save PATH` writes one as standalone MJCF; `--drive` puts the car
  on it under the wall-following controller in the viewer.
  `python3 -m examples.track_generation` (headless) or
  `mjpython -m examples.track_generation --drive --seed 7 --difficulty 3`.

Sensor reading lives in the package, not the examples: import from
`neoracer_mujoco.sensors` (`read`, `wheel_speed_ms`, `print_sensors`,
`lidar_scan`) to log IMU / wheel / steer / suspension / LiDAR off a compiled model.

Add your own demos here — a pure-pursuit follower, a data recorder, etc. — using
`run.py` as the template for the load → step → control loop.

## Validation suite (`validation/`)

A standardized gate every car model must clear before it's trained on. The suite
globs `assets/*.xml`, so a new car is tested automatically the moment its XML lands
in `assets/` — no edits needed.

```sh
python3 -m pytest validation/ -v
```

- **`test_conformance.py`** — physics checks, one per failure mode: actuator/sensor
  contract, won't-explode-at-rest (no NaN, rests on ground, stays upright), correct
  control response (throttle drives forward, steer yaws the right way, Ackermann
  inner > outer), and RL-exploit guards (no free energy, bounded top speed, finite
  sensors under load, determinism).
- **`test_logic.py`** — pure-logic checks: wheel-speed unit conversion, sensor read,
  and the Ackermann polyfit validated against exact arctan geometry.
- **`test_track_centering.py`** — `TrackCenteringController` sign/safety logic plus
  one physics-integration run on the corridor track.
- **`test_track_generation.py`** — the generator's own rules (every track valid,
  determinism, difficulty ordering, impossible limits raise) plus the MJCF layer's
  conventions (wall material name, spawn clearance, geom budget). The slow one:
  difficulty 0 burns most of its attempt budget per track, so it dominates suite
  runtime (~45 s).

The car contract (expected actuators, required sensors, mass band, ctrl layout)
lives in `src/neoracer_mujoco/contract.py` — update it there if the contract changes.
