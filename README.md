# NeoRacer MuJoCo Simulator

A MuJoCo simulation of the NeoRacer, a 1/12-scale autonomous racing car. You can drive one car in your own loop, step hundreds at once to train a policy, or run the same car in a browser with the JavaScript package in `js/`.

The car is described once, in `neoracer_spec`. The MuJoCo model, the sensor settings and the JavaScript package's numbers are all computed from that description.

The Gymnasium environment and the reinforcement learning are in [neoracer-rl](https://github.com/Neobotics-Foundation-Inc/neoracer-rl).

## The spec

`NEORACER`, in `src/neoracer_spec/neoracer.py`, describes the car part by part: the chassis, the wheels, the motor, the servo, the firmware's speed loop and each sensor. Every number in it is a `Sourced[T]`. That holds the value, how we know it (`Measured`, `Documented`, `Assumed`, `Estimated` or `Chosen`), and what other sources say when they disagree:

```python
from neoracer_spec.neoracer import NEORACER

wheelbase = NEORACER.chassis.wheelbase
wheelbase.value  # 0.28764 (m)
wheelbase.status  # Measured(source=Source(name='CAD', ...))
wheelbase.conflicts  # the docs say 0.280, the ROS driver 0.285
```

To list what is still a guess:

```python
from neoracer_spec.provenance import Estimated, walk

[path for path, value in walk(NEORACER) if isinstance(value.status, Estimated)]
# ['chassis.com_height', 'chassis.inertia', 'wheel.mass', 'wheel.friction', ...]
```

The simulators don't read the spec directly. `derive(spec)` computes what they need:
- the model's masses, inertias, springs and motor;
- the LiDAR's scan rate and resolution;
- the encoder's filter;
- full lock.

After you edit the spec, regenerate the two files that come from it. A test fails if either one is stale.

```sh
python -m neoracer_mujoco.write_car assets          # the numbers in assets/neoracer.xml
python -m neoracer_spec.typescript js/src/spec.ts   # the spec as TypeScript, for js/
```

## A scene

MuJoCo compiles an `MjSpec`, the editable model, into an `MjModel`, the arrays it simulates. A `Scene` keeps both, along with the car's parts:

```python
@dataclass(frozen=True)
class Scene:
    spec: mujoco.MjSpec  # what was compiled
    model: mujoco.MjModel  # the compiled model
    car: Car  # the car's parts, as elements of spec
```

`Car` holds the car's body, free joint, actuators, wheel geoms, LiDAR site and sensors, each as an element of `spec`. To read or write one, bind it to the model or to an `MjData` instead of looking up an index:

```python
data.bind_scalar(scene.car.steer).ctrl = 0.2  # rad, + = left
```

`load()` compiles the car on its own, on a floor. To drive it for a second and read its sensors:

```python
import mujoco

from neoracer_mujoco import load, read, sim
from neoracer_mujoco.control.steering import ControlCommand

scene = load()
data = sim.settle(scene.model)  # the car at rest on the floor
sim.apply_command(scene.car, data, ControlCommand(throttle=0.5, steer=0.2))
for _ in range(360):  # 1 s at 360 Hz
    mujoco.mj_step(scene.model, data)
read(scene.car.sensors, data)  # IMU, steering angles, wheel speeds, ...
```

The throttle runs from -1 to 1, as a fraction of the battery's 11.1 V. The steer is the steering angle in radians, positive to the left, up to 0.4405 at full lock.

## The car on a track

A track is an `MjSpec` with all its walls in a body named `walls`. `assets.compose` attaches the car to a track and compiles both into a `Scene`:

```python
import os

from neoracer_mujoco import assets

track = mujoco.MjSpec.from_file(os.path.join(assets.TRACK_DIR, "loop_corridor.xml"))
scene = assets.compose(track)
```

The scene has the same kind of `Car`, so the loop above drives the car on the track without changes. `apply_command` only needs `scene.car`, and `read` only `scene.car.sensors`.

Besides the four tracks in `assets/tracks/`, `track_generation` makes new ones from a seed. A closed random walk shapes them, so they have S-bends and hairpins. Difficulty runs from 0, wide and gentle, to 3, narrow and tight. `generate_track` works on geometry alone. It returns a `Track`, the centerline and the width along it, or `NoValidTrack` if no candidate passed its checks. `to_spec` turns a `Track` into an `MjSpec`:

```python
from neoracer_mujoco.track_generation import (
    NoValidTrack,
    Track,
    generate_track,
    settings_for_difficulty,
    to_spec,
)

match generate_track(7, settings_for_difficulty(1)):
    case Track() as track:
        scene = assets.compose(to_spec(track))
    case NoValidTrack():
        ...  # another seed, or looser settings
```

## The car and the track together

Every scene gets three sensors that only a simulator has: the car's position, its up axis, and how many contacts it has with the walls. With the `Track`, the position tells you how far around the lap the car is:

```python
from neoracer_mujoco.collision import car_touches_wall
from neoracer_mujoco.track_generation import lap_distance

data = sim.settle(scene.model)
...  # drive, as in the loop above
xy = data.bind_scalar(scene.car.truth.pos).data[:2]
lap_distance(track, xy)  # m along the centerline from the start
car_touches_wall(data, scene.car)  # True while the car touches a wall
```

The real LiDAR spins 30 times a second and measures its 1440 slots one after another, so the scan of a moving car is skewed. `lidar.sweep` models this. At each physics step it casts the slots that the head passed during that step, from the car's pose at that step, and returns the new `LidarState`.

`mj_step` leaves the poses at the start of its step, so the loop splits each step. `mj_step1` updates the poses, the sweep reads them, and `mj_step2` integrates:

```python
from neoracer_mujoco import lidar

params = lidar.LidarParams()  # the NeoRacer's LakiBeam L1
dt = scene.model.opt.timestep  # s
data = sim.settle(scene.model)
mujoco.mj_step1(scene.model, data)
state = lidar.start(params, scene.model, data, scene.car)
for _ in range(360):
    mujoco.mj_step2(scene.model, data)
    mujoco.mj_step1(scene.model, data)
    state = lidar.sweep(params, scene.model, data, scene.car, state, dt)
state.scan  # the last full turn: 1440 distances (m), clockwise from the front
```

`encoder.py` models the speed that the firmware reports from the motor encoder in the same way. `racecar_core_adapter.py` puts the actuators, the LiDAR and the encoder behind the `racecar_core` API that programs for the real car use. Those programs run in the simulator unchanged:

```sh
python examples/run_racecar_core.py my_controller.py
```

## Another car

A different car is a different `Spec`. The model is derived from the spec, so everything that depends on a change follows it. For example, a heavier chassis also gets more inertia, and a spring preload that carries the extra weight:

```python
from dataclasses import replace

from neoracer_mujoco import export
from neoracer_spec.provenance import estimated

heavy = replace(
    NEORACER,
    chassis=replace(NEORACER.chassis, mass=estimated(3.5, "with a camera mount")),
)
scene = assets.compose(to_spec(track), export.spec(heavy))
```

It is still a `Scene` with a `Car`, so all the code above works on it. `export.write(heavy, folder)` writes its XML, its meshes and an `.mjz` archive, for use outside Python.

## Many cars at once

To train a policy you usually want many cars at once:
- `batch.CpuBatch` steps copies of a scene on every core, with `mujoco.rollout`.
- `batch_warp.WarpBatch` steps them on a CUDA GPU, with MuJoCo Warp. It needs `neoracer-mujoco[jax]` and `jax[cuda12]`.

Both take the same `Scene` and spec elements, and return arrays with one row per car:

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

- **`visuals=False`** drops the meshes that only make the car look right. The physics stays the same, and the model shrinks from 35 MB to 1.6 MB.
- **Bit-for-bit rows.** A CPU row matches a single `MjData` stepped the same way, bit for bit. A Warp row stays within 1 mm of it after ten steps.
- **`Physics`** is the other way to change the car. It scales a compiled car's wheel friction, chassis mass, motor torque and steering stiffness without recompiling, so each row can drive a slightly different car. `physics.sample` draws one from `PhysicsRanges`.
- **A track per row.** Rows can also run on different tracks, if the tracks compile to the same size. Generate them with `WallParams(boxes_per_wall=192)`; see `batch.py`.

## Where the code lives

- `src/neoracer_spec/` holds the spec and the pure functions on it, and needs only numpy. Anything computed from the spec that doesn't need MuJoCo belongs here, so that every simulator can use it.
- `src/neoracer_mujoco/` holds everything that needs MuJoCo:
  - scenes, in `scene.py` and `assets.py`;
  - sensors, in `sensors.py`, `encoder.py`, `lidar.py` and `truth.py`;
  - batches, in `batch.py`, `batch_warp.py` and `physics.py`;
  - track generation, in `track_generation/`;
  - controllers, in `control/`;
  - the `racecar_core` adapter.
- Code that two modules share goes in a module both of them import. For example, the steering limits and throttle shaping that both controllers use are in `control/steering.py`.
- `assets/` holds `neoracer.xml` (its structure is written by hand and its numbers by `write_car`), the meshes (Git LFS objects) and the tracks.
- `js/` is the same car in MuJoCo's WebAssembly build. See [js/README.md](js/README.md).
- `examples/` holds runnable programs, `validation/` the tests, and `users/<github-name>/` personal experiments.

## Install

The package is not on PyPI. Each [GitHub Release](https://github.com/Neobotics-Foundation-Inc/neoracer-mujoco/releases) has:
- the wheel, with the car, the tracks and the meshes;
- `neoracer.mjz`, the car for MuJoCo without Python;
- the JS package.

```sh
pip install https://github.com/Neobotics-Foundation-Inc/neoracer-mujoco/releases/download/v0.12.0/neoracer_mujoco-0.12.0-py3-none-any.whl
```

To work on the package, clone it with [Git LFS](https://git-lfs.com) installed, since the meshes are LFS objects.

With [Nix](https://nixos.org), `nix develop` gives you Python 3.14 with every dependency, and Node for `js/`, all from nixpkgs' binary cache. The package runs from `src/`, so an edit takes effect on the next run.

Without Nix, install it editable on Python 3.14. `environment.yml` does the same with conda.

```sh
uv venv --python 3.14 && source .venv/bin/activate
uv pip install -e ".[dev]"
python -m pytest validation
```

## Examples

Run these from a clone; the examples aren't in the wheel.

| Command | What it does |
|---|---|
| `mjpython -m examples.run` | the car in the viewer under a demo controller |
| `python -m examples.manual_drive` | drive with the arrow keys, with a chase camera |
| `python -m examples.pygame_drive.drive` | drive with WASD or a gamepad, with a LiDAR radar and a lap timer (needs the `demo` extra) |
| `python -m examples.track_centering_demo` | the PD track-centering controller on the straight corridor |
| `python -m examples.track_generation` | generate tracks and describe them; add `--drive` under `mjpython` to drive one |
| `python examples/run_racecar_core.py CONTROLLER.py` | run a `racecar_core` program in the simulator |

## How close the model is to the real car

Nobody has checked the model against recordings of the real car yet (issue #4). Use it to develop and compare controllers, not to predict lap times. These are the known gaps:
- **Speed.** At full throttle the car tops out at about 3.5 m/s, from the RS-545 motor at the docs' 11,000 rpm. The wheel damping is a guess that stands in for every drivetrain loss. The firmware's speed controller isn't modelled, so the throttle is a fraction of the battery voltage (issue #118).
- **Mass and inertia.** The 3.0 kg total comes from the spec; nobody has weighed the car. The inertia and the height of the centre of mass come from an older car's URDF.
- **Tyres and suspension.** The tyre friction and the springs are estimates.
- **Steering.** At full lock the inner front wheel turns to the measured 30 degrees. The model's servo swings from full left to full right in 0.16 s. People who drive the real car find its steering slower (issue #115).
- **Sensors.** They have no noise, bias or latency, and the IMU never saturates. The LiDAR has no minimum range and never misses a return. There is no camera (issue #3).

## Citing

Use the "Cite this repository" button on GitHub, which reads `CITATION.cff`. `.github/citation.py` writes that file from the git history, so don't edit it by hand.
