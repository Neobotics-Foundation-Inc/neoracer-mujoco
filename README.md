# NeoRacer MuJoCo Simulator

The NeoRacer is a 1/12-scale autonomous racing car that students program in Python through the `racecar_core` API. This package simulates it in MuJoCo. The same car runs in three places: one car that you step yourself, batches of hundreds for training, and the browser, through the JavaScript package in `js/`.

All three take the car from one description, `NEORACER`, where every number carries its source. The sources do not always agree. The CAD model, the docs and the ROS driver give three different wheelbases, and nobody has weighed the car. So each number also keeps the values that disagree with it. Every number a simulator uses is computed from the description, not copied. Code reaches the car's parts through the elements of the model it was compiled from, so the same code drives the car alone, on any track, or in a batch.

The sections below follow the order in which the pieces build on one another: the spec, a scene compiled from it, the car on a track, what the car and the track give together, a variant of the car, and batches of cars. After them come the layout of the repository, installation, and how close the model is to the real car.

The Gymnasium environment and the reinforcement learning live in [neoracer-rl](https://github.com/Neobotics-Foundation-Inc/neoracer-rl).

## The spec

`NEORACER`, in `src/neoracer_spec/neoracer.py`, describes the car part by part: the chassis, the wheels, the motor, the servo, the firmware's speed loop and each sensor. Every number in it is a `Sourced[T]`. That holds the value, how we know it (`Measured`, `Documented`, `Assumed`, `Estimated` or `Chosen`), and the values other sources give where they disagree:

```python
from neoracer_spec.neoracer import NEORACER

wheelbase = NEORACER.chassis.wheelbase
wheelbase.value      # 0.28764 (m)
wheelbase.status     # Measured(source=Source(name='CAD', ...))
wheelbase.conflicts  # the docs say 0.280, the ROS driver 0.285
```

`provenance.walk` yields every value with its path. That is how you find what is still a guess:

```python
from neoracer_spec.provenance import Estimated, walk

[path for path, value in walk(NEORACER) if isinstance(value.status, Estimated)]
# ['chassis.com_height', 'chassis.inertia', 'wheel.mass', 'wheel.friction', ...]
```

The simulators never read the spec itself. `derive(spec)` turns a `Spec` into `DerivedParams`, the numbers a simulator needs:
- the model's masses, inertias, springs and motor (`ModelParams`);
- the LiDAR's scan rate and resolution;
- the encoder's filter;
- full lock.

`derive` is a pure function, and `neoracer_spec` needs only numpy, so it also runs under Pyodide in a browser. Two commands write its output where the simulators load it:

```sh
python -m neoracer_mujoco.write_car assets          # the numbers in assets/neoracer.xml
python -m neoracer_spec.typescript js/src/spec.ts   # the spec as TypeScript, for js/
```

Run both after you edit the spec. A test fails while either file is out of step with it.

## A scene

MuJoCo compiles an `MjSpec`, the editable form of a model, into an `MjModel`, the arrays it simulates. A `Scene` keeps the two together; `scene.py` defines it as:

```python
@dataclass(frozen=True)
class Scene:
    spec: mujoco.MjSpec    # what was compiled
    model: mujoco.MjModel  # the compiled model
    car: Car               # the car's parts, as elements of spec
```

`Car` holds the car's body, free joint, two actuators, wheel geoms, LiDAR site and sensors. Each one is an element of `spec`. Code binds an element to the model or to the data where it uses it, and never computes an index. With `data` an `MjData` of the scene's model:

```python
data.bind_scalar(scene.car.steer).ctrl = 0.2  # rad, + = left
```

The two actuators are MuJoCo `dcmotor`s:
- **The drive motor** takes the voltage at its terminals. Its torque falls as the car speeds up, and 0 V brakes.
- **The steering servo** runs its own position loop, so it turns the wheels no faster than a real servo.

Two equality constraints split the steering angle into the front wheels' Ackermann angles.

`load()` compiles the car alone, on a floor. This drives it for a second and reads its sensors:

```python
import mujoco

from neoracer_mujoco import load, read, sim
from neoracer_mujoco.control.steering import ControlCommand

scene = load()
data = sim.settle(scene.model)  # the car at rest on the floor
sim.apply_command(scene.car, data, ControlCommand(throttle=0.5, steer=0.2))
for _ in range(360):  # 1 s at 360 Hz
    mujoco.mj_step(scene.model, data)
read(scene.car.sensors, data)  # IMU, steering angles, wheel speeds, suspension, rangefinders
```

The throttle is a share of the battery's 11.1 V, from -1 to 1. The steer is the steering angle in radians, + to the left, up to full lock at 0.4405.

## The car on a track

A track is an `MjSpec` of scenery with every wall in a body named `walls`. `assets.compose(track)` attaches the car to the track and compiles the pair into a Scene with the same `Car`:

```python
import os

from neoracer_mujoco import assets

track = mujoco.MjSpec.from_file(os.path.join(assets.TRACK_DIR, "loop_corridor.xml"))
scene = assets.compose(track)
```

`apply_command` takes `scene.car` and `read` takes `scene.car.sensors`, not the whole scene, so the loop above drives the car on the track unchanged. Code that works on the car keeps working wherever the car is.

`assets/tracks/` holds four tracks, and `track_generation` makes new ones. `generate_track` is geometry only: it takes a seed and settings, and returns either a `Track` or `NoValidTrack`. A `Track` is the sampled centerline of a closed corridor, with its width at each sample. A closed random walk draws the control points, so tracks have S-bends and hairpins as well as ovals. A candidate that has a corner too tight for the car, touches itself or has the wrong length is repaired where it fails or drawn again, and `NoValidTrack` means none passed. The same seed always gives the same track. Difficulty runs from 0, gentle and wide, to 3, tight and narrow. `to_spec` turns a `Track` into a track `MjSpec`:

```python
from neoracer_mujoco.track_generation import (
    NoValidTrack, Track, generate_track, settings_for_difficulty, to_spec,
)

match generate_track(7, settings_for_difficulty(1)):
    case Track() as track:
        scene = assets.compose(to_spec(track))
    case NoValidTrack():
        ...  # another seed, or looser settings
```

## The car and the track together

Some quantities exist only once the car is on a track. Every scene gets three sensors that only a simulator has: the car's position, its up axis, and its number of contacts with the walls. Together with the `Track`, the position gives the progress along it:

```python
from neoracer_mujoco.collision import car_touches_wall
from neoracer_mujoco.track_generation import lap_distance

data = sim.settle(scene.model)
...  # drive, as in the loop above
xy = data.bind_scalar(scene.car.truth.pos).data[:2]
lap_distance(track, xy)            # m along the centerline from the start
car_touches_wall(data, scene.car)  # True while the car touches a wall
```

The LiDAR sees the walls. The real one measures each of its 1440 slots as its head passes, 30 turns a second, so the scan of a moving car is skewed. `lidar.py` models it as a state that each physics step advances. `sweep` casts the rays of the slots the head passed during the step, from the car's pose at that step, and returns the next `LidarState`.

`mj_step` leaves the poses at the start of the step it took, so the loop steps in two halves. `mj_step1` brings the poses up to `data.time`, the sweep reads them, and `mj_step2` integrates:

```python
from neoracer_mujoco import lidar

params = lidar.LidarParams()  # the NeoRacer's LakiBeam L1, from derive
dt = scene.model.opt.timestep  # s
data = sim.settle(scene.model)
mujoco.mj_step1(scene.model, data)
state = lidar.start(params, scene.model, data, scene.car)
for _ in range(360):
    mujoco.mj_step2(scene.model, data)
    mujoco.mj_step1(scene.model, data)
    state = lidar.sweep(params, scene.model, data, scene.car, state, dt)
state.scan  # the last whole turn: 1440 distances (m), clockwise from the front
```

`encoder.py` follows the same pattern for the speed the firmware reports. It counts the motor encoder over each 20 ms window and filters the result as the firmware does. `racecar_core_adapter.py` combines the actuators, the LiDAR and the encoder behind the real car's `racecar_core` API, so a student's program runs unchanged:

```sh
python examples/run_racecar_core.py my_controller.py
```

## Another car

A variant of the car is a new `Spec`. The model is derived from the spec, so every number that depends on a change follows it. Here, a heavier chassis also gets a larger inertia and a spring preload that carries the extra weight:

```python
from dataclasses import replace

from neoracer_mujoco import export
from neoracer_spec.provenance import estimated

heavy = replace(NEORACER, chassis=replace(NEORACER.chassis, mass=estimated(3.5, "with a camera mount")))
scene = assets.compose(to_spec(track), export.spec(heavy))
```

The scene still has the same `Car`, so every function above runs on it unchanged. `export.write(heavy, folder)` writes the car's XML, its meshes and an `.mjz` archive, for programs that load MuJoCo models without Python.

## Many cars at once

`batch.py` steps many copies of a scene at once, for example to train a policy:
- `CpuBatch` steps them with `mujoco.rollout` on every core.
- `batch_warp.WarpBatch` steps them with MJX on MuJoCo Warp, on a CUDA GPU. It needs `neoracer-mujoco[jax]` and `jax[cuda12]`.

Both take the same `Scene` and the same spec elements, and return arrays with one row per car:

```python
import numpy as np

from neoracer_mujoco import batch
from neoracer_mujoco.physics import Physics

scene = assets.compose(to_spec(track), visuals=False)
cpu = batch.CpuBatch(scene, nbatch=64, nthread=4)
state = cpu.initial([sim.settle(scene.model)] * 64, [scene] * 64, [Physics()] * 64)
ctrl = batch.command(scene.model, scene.car, np.full(64, 0.5), np.zeros(64))
state, sensordata = cpu.step(state, ctrl, nstep=12)  # (64, 12, n)
pos = batch.read(scene.model, sensordata[:, -1], scene.car.truth.pos)  # (64, 3)
cpu.close()
```

A spec variant and `Physics` both change the car, in different ways:
- **A spec variant is a different car.** Its model is compiled from its own numbers.
- **`Physics` scales a compiled car's numbers without recompiling.** It holds factors on the wheel friction, the chassis mass, the motor torque and the steering stiffness. `physics.sample` draws one from `PhysicsRanges`, so each row of a batch can train on a slightly different car.

A row of a `CpuBatch` steps the same, bit for bit, as one `MjData` stepped the same way. A `WarpBatch` row is close but not equal: after ten steps from rest, the cars are within 1 mm. On 4 cores, 64 rows run 7,000 to 8,000 env steps a second.

`visuals=False` matters for speed: without the meshes that only show the car, the model shrinks from 35 MB to 1.6 MB, with the same physics. A `CpuBatch` can also step each row on its own track, if every track compiles to a model of the same size: generate them with `WallParams(boxes_per_wall=192)`. `batch.py` has the details.

## Where the code lives

- `src/neoracer_spec/` holds the spec and the pure functions on it, and needs only numpy. Code that computes something from the spec without MuJoCo belongs here, so every simulator can use it.
- `src/neoracer_mujoco/` holds everything that needs MuJoCo:
  - scenes: `scene.py`, `assets.py`;
  - sensors: `sensors.py`, `encoder.py`, `lidar.py`, `truth.py`;
  - batches: `batch.py`, `batch_warp.py`, `physics.py`;
  - tracks: `track_generation/`, where only `mjcf.py` uses MuJoCo;
  - the classical controllers in `control/`, and the `racecar_core` adapter.
- Code that two parts of the package share goes in the module both import. For example, the steering limits and the throttle shaping that the wall follower and the track-centering controller both use are in `control/steering.py`.
- `assets/` holds `neoracer.xml`, whose structure is written by hand and whose numbers are written by `write_car`. It also holds the meshes (Git LFS objects) and the tracks.
- `js/` is the same car in MuJoCo's WebAssembly build. Its tests run the same cases as Python's and agree to 1e-14 m after a second of driving. See [js/README.md](js/README.md).
- `examples/` holds runnable programs, and `validation/` the tests: golden runs and laws.
- `users/<github-name>/` holds personal experiments.

## Install

The package is not on PyPI. Each [GitHub Release](https://github.com/Neobotics-Foundation-Inc/neoracer-mujoco/releases) holds:
- the wheel, with the car, the tracks and the meshes;
- `neoracer.mjz`, the car for MuJoCo without Python;
- the JS package.

```sh
pip install https://github.com/Neobotics-Foundation-Inc/neoracer-mujoco/releases/download/v0.12.0/neoracer_mujoco-0.12.0-py3-none-any.whl
```

To work on the package, clone it with [Git LFS](https://git-lfs.com) installed, since the meshes are LFS objects. With [Nix](https://nixos.org), run `nix develop`. It gives Python 3.14 with every dependency from nixpkgs' binary cache, and Node for `js/`. The package runs from `src/`, so an edit takes effect on the next run. The flake reads the dependencies from `pyproject.toml` and stops if a nixpkgs version breaks a pin.

Without Nix, install it editable on Python 3.14 (`environment.yml` does the same with conda):

```sh
uv venv --python 3.14 && source .venv/bin/activate
uv pip install -e ".[dev]"
python -m pytest validation
```

## Examples

Run these in a clone; the examples are not in the wheel.

| Command | What it does |
|---|---|
| `mjpython -m examples.run` | the car in the viewer under a demo controller |
| `python -m examples.manual_drive` | drive with the arrow keys, with a chase camera |
| `python -m examples.pygame_drive.drive` | drive with WASD or a gamepad, with a LiDAR radar and a lap timer (needs the `demo` extra) |
| `python -m examples.track_centering_demo` | the PD track-centering controller on the straight corridor |
| `python -m examples.track_generation` | generate tracks and describe them; add `--drive` under `mjpython` to drive one |
| `python examples/run_racecar_core.py CONTROLLER.py` | run a `racecar_core` program in the simulator |

## How close the model is to the real car

The model has not been checked against recordings of the real car (issue #4). Use it to develop and compare controllers, not to predict lap times. These are the known gaps:
- **Speed.** At full throttle the car tops out at about 3.5 m/s, from the RS-545 motor at the docs' 11,000 rpm. The wheel damping is a guess that stands in for every drivetrain loss. The firmware's speed controller is not modelled, so a throttle is a share of the battery voltage (issue #118).
- **Mass and inertia.** The 3.0 kg total comes from the spec; nobody has weighed the car. The inertia and the height of the centre of mass come from an older car's URDF.
- **Tyres and suspension.** The tyre friction and the springs are estimates.
- **Steering.** Full lock puts the inner front wheel at the measured 30 degrees. The servo model swings from full left to full right in 0.16 s, and people who drive the real car find its steering slower (issue #115).
- **Sensors.** They have no noise, bias or latency, and the IMU does not saturate. The LiDAR has no minimum range and never misses a return. There is no camera (issue #3).

## Citing

Use the "Cite this repository" button on GitHub, which reads `CITATION.cff`. `.github/citation.py` writes that file from the git history, so do not edit it by hand.
