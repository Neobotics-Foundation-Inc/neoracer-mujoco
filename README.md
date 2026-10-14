# NeoRacer MuJoCo Simulator

Clean, team-facing MuJoCo baseline for the NeoRacer 1/12-scale RC car.

The Gymnasium environment and the reinforcement learning examples, such as a PPO agent that learns to lap a track, live in [neoracer-rl](https://github.com/Neobotics-Foundation-Inc/neoracer-rl). It installs this package from its GitHub Release.

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
uv venv --python 3.15
uv pip install -r requirements.txt
source .venv/bin/activate
```

Then install the package (editable), so `neoracer_mujoco` is importable by the
examples and the validation suite:

```sh
pip install -e .
```

The package is not on PyPI. To use it from another project, install a wheel
from a GitHub Release. The wheel contains the car and track XMLs and the
meshes, so it needs no Git LFS. In a clone, `load()` reads them from
`assets/` at the repo root.

## Quick start

```sh
# Run the model in the viewer (macOS needs mjpython for the passive viewer):
mjpython -m examples.run

# Run the validation suite (Python 3.15 or newer, with the steps above done):
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
│   ├── assets.py          — cars() / load() / compose(): find and compile a Scene
│   ├── scene.py           — Scene (spec + model) and Car, the car's parts as spec elements
│   ├── sensors.py         — SensorReadings, find_sensors() + read()
│   ├── truth.py           — sensors that only the simulator has: car pose, wall contact
│   ├── batch.py           — step many copies of a scene at once (mujoco.rollout, or MuJoCo Warp)
│   ├── sim.py             — settle, step and command helpers, physics probes
│   ├── control/           — classical controllers (track_centering)
│   └── track_generation/  — procedural racetracks (generate_track → Track → MJCF)
├── examples/
│   ├── run.py                  — viewer launch script (mjpython entry point)
│   ├── manual_drive.py         — arrow-key teleop (game-style, python3 entry point)
│   ├── pygame_drive/           — WASD or gamepad driving with a LiDAR radar (pygame)
│   ├── track_centering_demo.py — PD centering controller on the corridor track
│   └── track_generation.py     — generate procedural tracks, describe or drive one
├── validation/            — pytest physics + logic conformance suite
├── users/<name>/          — personal experiments; not checked by ruff, ty, tests or CI
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

### Actuator interface

| Actuator | Unit | Direction | Spec element |
|---|---|---|---|
| fl_motor | N·m | + = forward | `scene.car.drive[0]` |
| fr_motor | N·m | + = forward | `scene.car.drive[1]` |
| rl_motor | N·m | + = forward | `scene.car.drive[2]` |
| rr_motor | N·m | + = forward | `scene.car.drive[3]` |
| steer_servo | rad | + = left turn | `scene.car.steer` |

Code writes an actuator by binding its element to the data, for example `data.bind_scalar(scene.car.steer).ctrl = 0.2` (see `scene.py`), so it does not depend on the order of the actuators in the XML. `sim.apply_command(scene.car, data, command)` writes a throttle and a steer.

Steering command range is ±0.4 rad; the two Ackermann equality constraints split
it into the correct inner/outer front-wheel angles automatically.

### Truth sensors

The real car cannot measure where it is or whether it touches a wall, so `neoracer.xml` has no such sensors. Every scene gets three that only the simulator has (`truth.py`):

| Sensor | Values | Spec element |
|---|---|---|
| truth_pos | x, y, z of the car body (m, world frame) | `scene.car.truth.pos` |
| truth_zaxis | the car's +Z axis, in the world frame | `scene.car.truth.zaxis` |
| truth_wall_contact | the number of contacts between the car and a wall | `scene.car.truth.wall_contact` |

A wall is a geom of the body named `walls`. The tracks in `assets/tracks/` and the generated tracks put every wall there. With these sensors, a step can read everything it needs from `data.sensordata`, which `mujoco.rollout` and MuJoCo Warp also return for every step of a batch. `collision.car_touches_wall(data, scene.car)` and `sim.car_upright_cos(data, scene.car)` read them.

### Batches

`batch.py` steps many copies of one scene at once. `CpuBatch` steps rows of physics state with `mujoco.rollout` on a pool of threads, which run in C and use every core. `batch_warp.WarpBatch` steps a batched `mjx.Data` with MJX on MuJoCo Warp, on a CUDA GPU; it needs the `jax` extra (`pip install "neoracer-mujoco[jax]"`). `make_batch(model, nbatch, default_backend())` picks Warp when JAX sees a GPU.

```python
sim = batch.make_batch(scene.model, nbatch=64, backend=batch.default_backend())
state = sim.initial(data)  # every row a copy of data
ctrl = batch.command(scene.model, scene.car, throttle, steer)  # (64, nu)
state, sensordata = sim.step(state, ctrl, nstep=17)  # (64, 17, nsensordata)
upright = batch.read(scene.model, sensordata[:, -1], scene.car.truth.zaxis)[:, 2]
```

Each `step` starts the constraint solver cold, and a CPU row is then the same, bit for bit, as one MjData stepped the same way. Warp computes collisions and constraints with its own code, so its rows are close to the CPU rows, not equal: after ten env steps from rest, within 1 mm. On 4 CPU cores, a batch of 64 runs about 3,200 env steps/s (17 physics steps each), against about 2,400 for threads that step one MjData each.

## Track generation (`src/neoracer_mujoco/track_generation/`)

`generate_track` makes the track geometry. `assets.compose` attaches the car to it:

```python
from neoracer_mujoco import assets
from neoracer_mujoco.track_generation import (
    NoValidTrack,
    Track,
    generate_track,
    settings_for_difficulty,
    to_spec,
)

match generate_track(7, settings_for_difficulty(2)):  # pure geometry, no MuJoCo
    case Track() as track:
        scene = assets.compose(to_spec(track))  # a compiled Scene, car included
    case NoValidTrack():
        ...  # try another seed or looser settings
```

`generate_track(seed, settings)` returns a `Track` or a `NoValidTrack`. A `Track` holds N evenly spaced samples around a closed loop, each with a centerline position, tangent, left normal, curvature and corridor half-width, plus the loop's total length. The same seed always gives the same track. `settings_for_difficulty` gives the limits for difficulty 0 (gentle and wide) to 3 (tight and narrow); use `dataclasses.replace` on the result to change other knobs in `TrackSettings`.

Generation draws a candidate and checks it for corners tighter than the car can take, a loop that touches itself, and a total length outside the target range. Then it moves the control points behind each problem and checks again, instead of drawing the whole loop again. A candidate that cannot be repaired is thrown away and a new one is drawn. If every attempt fails, `generate_track` returns `NoValidTrack`. Then try another seed, a larger `max_attempts` or looser limits.

| File | Role |
|---|---|
| `track.py` | the `Track` itself and the attempt and repair loop. **Read this first.** |
| `shape.py` | control points → spline → even resample → corridor width |
| `reject.py` | what makes a candidate invalid, and how to move it |
| `config.py` | the knobs (`TrackSettings`, `settings_for_difficulty`) |
| `mjcf.py` | `to_mjcf(track)` and `to_spec(track)`: walls and a floor, no car. The only module here that uses MuJoCo. |

Everything except `mjcf.py` is plain NumPy and SciPy with no simulator assumptions, so the geometry is usable outside MuJoCo.

## Usage / demo scripts (`examples/`)

`examples/` holds runnable, hackable demos — start here to drive the car yourself
or to read sensors. They are meant to be copied and modified, not imported as a
stable API.

- **`run.py`** — launches the MuJoCo viewer and drives the car with a built-in
  demo controller (constant throttle + sinusoidal steering) so you can watch it
  move. Run with `mjpython -m examples.run`. Safe mode (the default) uses a low
  throttle and a speed limit. `--mode full` uses a high throttle and no speed limit,
  so the car rolls over at high speed. `DriveParams` holds the values of each mode.
- **`manual_drive.py`** — drive the car yourself with the arrow keys, video-game
  style: it owns a glfw window and polls held keys each frame, so you hold to build
  speed, release to coast down, and let the wheel re-center. A back-top chase camera
  follows the car's heading. Use this to *feel* the physics — grip, rollover,
  suspension, wall contacts. Run with plain `python3 -m examples.manual_drive` (NOT
  mjpython — the glfw window must be created on the main thread). Add `--bare` for
  the car alone, or `--track PATH` for another track. To change the camera and how
  the car responds to the keys, change the defaults in `ManualDriveParams`.
- **`pygame_drive/`** — drive the car with WASD or a gamepad in a pygame window,
  with a chase camera, a LiDAR radar, the held keys and a lap timer. The throttle is
  a DC-motor model (back-EMF drag and rolling friction), so the car has a top speed
  and slows down when you let go. `drive_sim.py` is the physics, `drive.py` the UI.
  Install pygame-ce with `pip install -e ".[demo]"`, then run
  `python3 -m examples.pygame_drive.drive` (the F1 track),
  `--track ramp_course.xml` for another track, or `--bare` for the car alone.
- **`track_generation.py`** — generates procedural tracks and prints what came out (loop length, corridor width, tightest corner, geom count) across seeds and difficulties. `--save PATH` writes one as MJCF; `--drive` puts the car on it under the wall-following controller in the viewer. `python3 -m examples.track_generation` (headless) or `mjpython -m examples.track_generation --drive --seed 7 --difficulty 3`.
- **`track_centering_demo.py`** — runs the classical PD track-centering controller (`control.track_centering.step`)
  on the straight corridor track, headless by default or with an interactive
  viewer. Composes `neoracer.xml` onto `assets/tracks/straight_corridor.xml` at
  runtime, settles, then drives the controller and reports centering performance.
  `python3 -m examples.track_centering_demo` (headless) or
  `mjpython -m examples.track_centering_demo --viewer`.

The package reads the sensors, not the examples. To log the IMU, wheel, steer, suspension and LiDAR values of a scene, call `sensors.read(scene.car.sensors, data)` at each step. `scene.car.sensors` holds the sensor elements of the scene's MjSpec, and `read` binds each one to the data. `lidar_ranges` and `motor_encoder_rad_s` read from the result.

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
- **`test_track_centering.py`** — track-centering `step` sign/safety logic plus
  one physics-integration run on the corridor track.

The car contract (expected actuators, required sensors, mass band, ctrl layout)
lives in `src/neoracer_mujoco/contract.py` — update it there if the contract changes.
