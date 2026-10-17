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

Each release also holds `neoracer.mjz`: the car XML and its meshes in one
archive, which MuJoCo loads without Python, for example in its WebAssembly
build as `neoracer.mjz/model.xml`. `assets.write_mjz(xml_path, out)` writes
such an archive of any XML in `assets/`, with the XML as written, so it
compiles to the same model, bit for bit.

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
│   ├── neoracer.xml       — MuJoCo MJCF model, its numbers written from twin/
│   └── meshes/            — visual STL meshes (cosmetic; mass="0")
├── src/neoracer_mujoco/   — importable package (the reusable toolbox)
│   ├── contract.py        — the car contract (single source of truth)
│   ├── twin/              — the car's digital twin: its parts with their sources (neoracer.py), the model numbers and settings derived from them, and its XML, MjSpec and .mjz (export.py)
│   ├── write_car.py       — writes the NeoRacer into assets/: the XML's numbers and neoracer.json
│   ├── assets.py          — cars() / load() / compose(): find and compile a Scene
│   ├── scene.py           — Scene (spec + model) and Car, the car's parts as spec elements
│   ├── sensors.py         — SensorReadings, find_sensors() + read()
│   ├── truth.py           — sensors that only the simulator has: car pose, wall contact
│   ├── batch.py           — step many copies of a scene at once (mujoco.rollout, or MuJoCo Warp)
│   ├── physics.py         — change the car's grip, mass, motor torque and steering stiffness per row
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
| Wheelbase | 287.6 mm | production CAD model, measured |
| Track width | 224 mm | production CAD model, measured |
| Wheel radius | 42.5 mm | spec (85 mm tyres) |
| Chassis box | 481 × 241 × 184 mm, from the bottom of the tub to the top of the LiDAR | production CAD model, measured |
| Chassis mass | 2.755 kg | spec total less the wheels |
| Chassis CoM | above the middle of the wheelbase, 18.9 mm above the axles | URDF height |
| Chassis inertia | diag (0.00404, 0.00992, 0.01191) kg·m² | URDF (base_link), scaled by mass |
| Wheel mass | 58.7 g each | URDF — verify vs real wheel+tyre |
| Total mass | 3.0 kg | spec |
| Steering | Ackermann (4th-order polynomial equality constraints) | |
| Drive | One RS-545SH DC motor (MuJoCo's `dcmotor`), 10.55:1 to the wheels through open differentials, on an 11.1 V battery | the rotor's inertia is left out |
| Suspension | Coil-over approximation (slide Z joint, k=600 N/m, c=25 Ns/m), preloaded to rest at the CAD ride height | ESTIMATED |
| Sensors | IMU (accel/gyro/quat/vel) · steer · wheel vel · drivetrain angle · susp pos · 8 rangefinder beams · a spinning LiDAR of 1440 slots a turn at 30 Hz, its rear quarter blocked (`lidar.py`) | the car's Richbeam LakiBeam1: 270° seen, 25 m on a bright target; its noise, minimum range and missed returns are not modelled |
| Time step | 1/360 s, with the implicitfast integrator: 12 steps to a 30 Hz LiDAR scan, 6 to a 60 Hz frame | every track sets the same `<option>` |

### How closely the model matches the real car

The model has not yet been checked against recordings of the real car; issue #4 tracks that system identification. Until then, treat the simulator as a tool for developing and comparing controllers, not as a prediction of the real car's lap times or speeds.

What the car is lives in one place, `twin/neoracer.py`: the NeoRacer part by part (chassis, wheels, suspension, battery, motor, ESC, steering servo, steering, the firmware's speed loop, the motor encoder, the IMU, the LiDAR and the camera), each value with its source, marked ESTIMATED or ASSUMED where nobody measured it, and INCONSISTENT where sources disagree. The part types are in `twin/parts.py`. Pure functions in `twin/derive.py` turn a car into the model's numbers (`model`: the motor's and the servo's `nominal`, full lock, the Ackermann polynomials, the springs' preload, the chassis' mass and inertia, positions) and into the settings beside the model (`settings`: the LiDAR's scan, the encoder's filter, full lock); `LidarParams` and `EncoderParams` default to the NeoRacer's. A test fails if `neoracer.xml` does not hold the model's numbers. To change the car, change `twin/neoracer.py` and run `python -m neoracer_mujoco.write_car assets`; a variant is `dataclasses.replace(NEORACER, ...)`. `twin/export.py` turns any car into its XML, an MjSpec (`assets.compose(track, export.spec(car))` puts it on a track, for CPU or GPU training) and a folder with its `.mjz` archive for MuJoCo's WebAssembly build (`export.write(car, folder)`).

| Part | Where the value comes from |
|---|---|
| Wheelbase, track width, chassis box and the LiDAR's position | measured on the production car's CAD model (the playground's `neoracer-race.glb`) |
| Total mass (3.0 kg) and tyre diameter (85 mm) | the spec (2026-09-29) |
| Chassis inertia and center-of-mass height; wheel masses | the URDF of an older CAD model; the inertia is scaled to the spec mass, and nobody has measured either |
| Suspension stiffness and damping | estimated; the springs are preloaded so the car rests at the CAD ride height |
| Tire friction (1.5) and wheel joint damping | estimated; the damping, 0.0005 N·m·s/rad per wheel, stands in for drivetrain losses and holds the top speed to 3.5 m/s |
| Steering servo: a DC motor with a position loop; at the steering input, 5 V, 3.28 N·m stalled, 5.99 rad/s free, kp 50 V/rad, kd 1 V·s/rad | the car's SPT5435LV-180W, its datasheet (35 kg·cm and 0.14 s per 60° at 6.0 V, 29 kg·cm and 0.16 s at 4.8 V) interpolated to the OSCORE board's 5 V rail that powers the servo header (NeoRacer docs), through a 1.116:1 linkage derived from the servo's 90° over 1000–2000 us and the driver's 0.625 lock; the real lock-to-lock time is open (issue #115), and the loop's gains are picked so it settles without chatter at the 1/360 s time step |
| IMU position | at the center of mass, not measured |
| Cosmetic meshes | the older CAD model, placed to roughly fill the measured body |
| Drive: one RS-545 through open differentials, on an 11.1 V battery; at the wheels, 0.92 N·m from rest and a free speed of 101 rad/s (4.3 m/s) | the docs' 11,000 rpm with no load, taken at 12 V; the stall torque, 93.8 mN·m, scaled from the RS-545SH-4075's datasheet to that speed; the firmware's 10.55:1 reduction from motor to wheels and the battery's voltage; the rotor's inertia is not in a datasheet and is left out (issue #114) |

Known gaps that matter for sim-to-real work:

- At full throttle the car reaches 3.0 m/s after 1 s and tops out at 3.5 m/s, against the motor's free speed of 4.3 m/s at the tyre: the wheel joint damping is the only loss, and its value is a guess. The car's firmware runs a speed controller that the simulator does not model (issue #118): a throttle here is a share of the battery voltage. The firmware caps the speed at 6 m/s, which this motor does not reach.
- The sensors have no noise, bias or latency, and the LiDAR has no minimum range or missed returns. The car has no camera yet (issue #3).
- The steering gives under load. Full lock is 0.4405 rad of steering input, which puts the inner front wheel at the car's measured 30 degrees. At full lock and 0.3 m/s, the servo reaches 0.429 rad (the inner wheel 28.9 degrees), the front wheels lag their Ackermann angles by 0.004 rad, and the car turns on a 0.67 m radius, against 0.61 m for ideal Ackermann steering at full lock: the rest is the tyres' slip. From rest, the servo reaches 90% of a full-lock step in 0.078 s and swings from full left to full right in 0.16 s; people who drive the car find the real steering slower (issue #115).

### Actuator interface

| Actuator | Unit | Direction | Spec element |
|---|---|---|---|
| motor | V, the voltage at its terminals | + = forward | `scene.car.drive` |
| steer_servo | rad | + = left turn | `scene.car.steer` |

The motor is MuJoCo's `dcmotor`: its torque is K (V − K ω) / R at a speed ω, so it falls as the wheels speed up, the car settles at a speed roughly proportional to the voltage, and 0 V brakes it. Its ctrlrange is the battery voltage, which `contract.actuator_limits(model, car).voltage` reads. Code above the model speaks of a throttle in [-1, 1], the fraction of the battery voltage that the speed controller applies: `sim.apply_command` and `batch.command` multiply it by the battery voltage.

The steering servo is a `dcmotor` too, with `input="pos"`: its ctrl is a target angle, and its own position loop sets the motor's voltage, so it turns the wheels no faster than the motor's free speed.

Code writes an actuator by binding its element to the data, for example `data.bind_scalar(scene.car.steer).ctrl = 0.2` (see `scene.py`), so it does not depend on the order of the actuators in the XML. `sim.apply_command(scene.car, data, command)` writes a throttle and a steer.

Steering command range is ±0.4405 rad, the car's full lock (the inner front wheel at 30 degrees); the two Ackermann equality constraints split
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

`batch.py` steps many copies of one scene at once, for example to train a policy. Pick the version that fits your machine. Both have the same methods, and `batch.command` and `batch.read` build the controls and read a sensor out of the result on either one.

Nothing renders a batch, so build its scene with `assets.compose(track, visuals=False)`. That leaves out the meshes that only show the car: the physics and the sensors stay the same, bit for bit, while the model shrinks from 35 MB to 1.6 MB on `loop_corridor`, compiles in 0.02 s instead of 0.3 s, and steps about 17% faster.

#### Optimized for CPU

`CpuBatch` steps the copies with `mujoco.rollout` on a pool of threads that run in C and use every core. It needs nothing beyond the package.

```python
import numpy as np

from neoracer_mujoco import assets, batch, sim
from neoracer_mujoco.physics import Physics

scene = assets.load()
cpu = batch.CpuBatch(scene, nbatch=64, nthread=4)
# 64 copies of a car at rest, each with the physics of the XML
state = cpu.initial([sim.settle(scene.model)] * 64, [scene] * 64, [Physics()] * 64)
throttle, steer = np.full(64, 0.5), np.zeros(64)  # in [-1, 1], rad
ctrl = batch.command(scene.model, scene.car, throttle, steer)
state, sensordata = cpu.step(state, ctrl, nstep=17)  # sensordata: (64, 17, n)
pos = batch.read(scene.model, sensordata[:, -1], scene.car.truth.pos)  # (64, 3)
cpu.close()
```

On 4 cores, a batch of 64 runs about 3,200 env steps/s (17 physics steps each), against about 2,400 for Python threads that each step one MjData. Each `step` starts the constraint solver cold, and a row is then the same, bit for bit, as one MjData stepped the same way.

#### Optimized for GPU

`batch_warp.WarpBatch` steps the copies with MJX on MuJoCo Warp, on a CUDA GPU. Install the `jax` extra and the CUDA build of JAX:

```bash
pip install "neoracer-mujoco[jax]" "jax[cuda12]"
```

```python
import jax.numpy as jnp

from neoracer_mujoco import assets, batch, batch_warp, sim
from neoracer_mujoco.physics import Physics

scene = assets.load()
gpu = batch_warp.WarpBatch(scene, nbatch=4096)
state = gpu.initial(
    [sim.settle(scene.model)] * 4096, [scene] * 4096, [Physics()] * 4096
)
throttle, steer = jnp.full(4096, 0.1), jnp.zeros(4096)
ctrl = batch.command(scene.model, scene.car, throttle, steer)
state, sensordata = gpu.step(state, ctrl, nstep=17)
pos = batch.read(scene.model, sensordata[:, -1], scene.car.truth.pos)
```

Warp computes collisions and constraints with its own code, so its results are close to the CPU version's, not equal: after ten env steps from rest, the car positions are within 1 mm. Without a GPU, JAX runs Warp on the CPU, correctly but much more slowly than `CpuBatch`.

#### Many tracks at once

`initial` takes one `Scene` per row, so a `CpuBatch` can step each row on its own track, as long as every scene has the sizes of the batch's. Generated tracks with a fixed number of wall boxes do (see `boxes_per_wall` under Track generation):

```python
from neoracer_mujoco import assets, batch, sim
from neoracer_mujoco.physics import Physics
from neoracer_mujoco.track_generation import (
    WallParams,
    generate_track,
    settings_for_difficulty,
    to_spec,
)

walls = WallParams(boxes_per_wall=192)
# Seeds 0 to 7 all give a Track, not a NoValidTrack.
tracks = [generate_track(seed, settings_for_difficulty(1)) for seed in range(8)]
scenes = [assets.compose(to_spec(t, walls), visuals=False) for t in tracks]
cpu = batch.CpuBatch(scenes[0], nbatch=8, nthread=4)
state = cpu.initial([sim.settle(s.model) for s in scenes], scenes, [Physics()] * 8)
cpu.close()
```

A `WarpBatch` steps every row on its own scene: MuJoCo Warp collides and casts rays against the geometry of the model it was made from, whatever geometry each environment gets later. For another track on Warp, make another `WarpBatch`.

### Physics randomization

Nobody knows the real car's grip, mass or motor strength exactly, so a controller trained on one simulated car may fail on the real one. Domain randomization trains it on many slightly different cars instead. `physics.py` changes four parameters of the compiled model, as factors on the values in the XML:

| Factor | Changes |
|---|---|
| `wheel_friction` | the sliding friction of the wheel contacts, and so how much the wheels slip |
| `chassis_mass` | the mass and inertia of the car body |
| `motor_torque` | the motor's torque at every speed; its free speed does not change |
| `steer_stiffness` | the gain kp of the steering servo's position loop, and so its stiffness near the target |

`PhysicsRanges` gives the interval of each factor (the default changes nothing), `physics.sample` draws a `Physics` from it, and `apply_physics` writes one into a copy of the model:

```python
import copy

import numpy as np

from neoracer_mujoco import assets, physics
from neoracer_mujoco.physics import PhysicsRanges, Range

scene = assets.load()
ranges = PhysicsRanges(wheel_friction=Range(0.5, 1.2), motor_torque=Range(0.7, 1.3))
params = physics.sample(ranges, np.random.default_rng(0))
model = copy.deepcopy(scene.model)
physics.apply_physics(model, scene, params)
```

A batch takes one `Physics` per row in `initial`, and a row's physics is part of its state, so `reset` gives a row that starts over the physics of `initial` too. The model is the same, bit for bit, as one compiled with the changed values, and a row of `CpuBatch` steps the same as one MjData on that model.

On 4 cores, with 64 rows on `loop_corridor`:

| | CpuBatch | WarpBatch, on the CPU |
|---|---|---|
| env steps/s, one physics for all rows | 7,300 to 8,100 | 89 |
| env steps/s, 64 different physics | 7,000 to 7,500 | 87 |
| `initial` with 64 new physics | 100 to 180 ms | 60 ms |

`CpuBatch` keeps one MjModel per different `Physics`, so build its scene with `assets.compose(track, visuals=False)`, which makes each model 1.6 MB instead of 35 MB. With the visuals, the 64 copies take 11 s to make. `WarpBatch` keeps only the fields that `apply_physics` changes, `physics.FIELDS`, for each environment. There was no GPU to measure Warp on, so its numbers above show only that per-environment physics costs nothing extra.

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

The loop runs counterclockwise through control points that a closed random walk draws: each step turns from the last by its share of one full turn plus random noise. So the loop also turns right in places, which makes S-bends, chicanes and dents, and not only ovals. Over seeds 0 to 199, the largest bend to the right has a median of 45° at difficulty 1 and 68° at difficulty 3, and a quarter of the difficulty 3 tracks have one of 90° or more.

Generation draws a candidate and checks it for corners tighter than the car can take, a loop that touches itself, and a total length outside the target range. Then it moves the control points behind each problem and checks again, instead of drawing the whole loop again. A candidate that cannot be repaired is thrown away and a new one is drawn. If every attempt fails, `generate_track` returns `NoValidTrack`. Then try another seed, a larger `max_attempts` or looser limits.

To train on many tracks at once, a batch steps one track per row, and that needs every track to compile to a model of the same size. By default the walls use as few boxes as `segment_length` and `max_deviation` allow, so each track has its own number. `WallParams(boxes_per_wall=192)` fixes the number: it splits the longest boxes until each wall has 192, which keeps both limits. Over 240 tracks of every difficulty, a wall needed 78 to 161 boxes. `to_spec(track, WallParams(boxes_per_wall=192))` then gives the same sizes for every track.

`lap_distance(track, xy)` gives how far along the centerline, in meters from the start, a point of the scene is, for the progress of a car around a generated track.

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

`motor_encoder_rad_s` is the raw speed of the drivetrain. The car reports a different speed: every 20 ms its firmware turns the motor encoder's count over the window into the tyre's surface speed and filters it, 0.95 of the new window to 0.05 of the last value. `encoder.py` computes that speed from the `drive_pos` sensor, one physics step at a time (`encoder.start` and `encoder.step`, with the window and the filter in `EncoderParams`), and the racecar_core adapter returns it from `rc.physics.get_encoder_speed()`.

The 8 rangefinder beams are for controllers and policies that need a few cheap distances each step. The real car's LiDAR spins: `lidar.py` measures each of its 1440 slots when the head passes it, so at 360 Hz each physics step casts the 120 slots of a twelfth of a turn, from that step's pose, with one `mj_multiRay` call, and the scan of a moving car is skewed as the real one is. `lidar.start` gives a whole turn from where the car stands and `lidar.sweep` advances one step, with the scan's numbers in `LidarParams`; slot i points i × 0.25° clockwise from the front, and reads −1 where the car blocks the rear or nothing lies within 25 m. The racecar_core adapter returns the last complete turn from `rc.lidar.get_samples()`, in cm, with 0 for no return. The sweep makes a step about two to four times as long, depending on the track's geom count.

Add your own demos here — a pure-pursuit follower, a data recorder, etc. — using
`run.py` as the template for the load → step → control loop.

## JavaScript package (`js/`)

`js/` is the same car for MuJoCo's WebAssembly build, for a browser or Node. It puts the car on a track given as MJCF, writes commands, and computes the encoder speed and the swept LiDAR scan as the Python package does, with the same numbers. Its tests run the same cases in both, from `js/test/fixtures.py`, and the results agree to 1e-14 m after a second of driving. `validation/test_js_package.py` checks that the names and numbers it copies agree with this package. `js/README.md` shows how to use it.

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

## Citing

If you use this simulator in your work, cite it with the "Cite this repository" button on GitHub, which reads `CITATION.cff`. The authors in that file are the five people with the most commits in the git history. A commit counts for its author and for each person in a `Co-authored-by` trailer, `.mailmap` merges the names and emails of one person, and AI tools and bots do not count. `.github/citation.py` writes the file, so do not edit it by hand.
