# NeoRacer MuJoCo Simulator

The NeoRacer RC car in MuJoCo: the car model, its sensors as the real car reports them, generated racetracks, and batches of cars for training on CPU or GPU. The same car runs in the browser through the JS package in `js/`.

The Gymnasium environment and the reinforcement learning examples live in [neoracer-rl](https://github.com/Neobotics-Foundation-Inc/neoracer-rl).

## Install

The package is not on PyPI. Each [GitHub Release](https://github.com/Neobotics-Foundation-Inc/neoracer-mujoco/releases) holds:

- the wheel, with the car and track XMLs and the meshes;
- `neoracer.mjz`, the car XML and its meshes in one archive, which MuJoCo loads without Python;
- the JS package as an npm tarball.

```sh
pip install https://github.com/Neobotics-Foundation-Inc/neoracer-mujoco/releases/download/v0.11.0/neoracer_mujoco-0.11.0-py3-none-any.whl
```

To work on the package, clone it with [Git LFS](https://git-lfs.com) installed (the meshes in `assets/meshes/` are LFS objects) and install it editable on Python 3.15:

```sh
uv venv --python 3.15 && source .venv/bin/activate
uv pip install -e ".[dev]"
python -m pytest validation/
```

`environment.yml` makes the same environment with conda; then run `pip install -e ".[dev]"` in it.

## Quick start

```python
import mujoco

from neoracer_mujoco import load, read, sim
from neoracer_mujoco.control.steering import ControlCommand

scene = load()  # assets/neoracer.xml, compiled
data = sim.settle(scene.model)  # the car at rest on the floor
sim.apply_command(scene.car, data, ControlCommand(throttle=0.5, steer=0.2))
for _ in range(360):  # 1 s at 360 Hz
    mujoco.mj_step(scene.model, data)
print(read(scene.car.sensors, data))
```

`assets.compose(track)` puts the car on a track, given as an MjSpec: one of `assets/tracks/` or a generated one. To watch the car, run `mjpython -m examples.run` in a clone: the examples are not in the wheel.

## The car

What the car is lives in one file, `src/neoracer_spec/neoracer.py`: the NeoRacer part by part, from the chassis and the motor to the firmware's speed loop and each sensor. Every value is a `Sourced[T]`: the value, how we know it (measured, documented, assumed, estimated, or chosen for the model), its source, and the values that other sources give where they disagree. `provenance.walk(NEORACER)` lists every value by its path, so a program can find what is estimated or disputed.

`neoracer_spec` needs only numpy. Pure functions turn the spec into what the simulators run:

- `derive.derive(spec)` gives `DerivedParams`: the model's numbers (`ModelParams`), the LiDAR's scan, the encoder's filter and full lock.
- `mjcf.car_xml` writes the model's numbers into `assets/neoracer.xml`, which keeps its structure and comments.
- `typescript.module()` writes the types and values as TypeScript, `js/src/spec.ts`, for the JS package.

To change the car, edit `neoracer.py`, then run:

```sh
python -m neoracer_mujoco.write_car assets
python -m neoracer_spec.typescript js/src/spec.ts
```

Tests fail if either file is out of step with the spec. A variant is `dataclasses.replace(NEORACER, ...)`. `neoracer_mujoco.export` turns any car into its XML, an MjSpec (`assets.compose(track, export.spec(car))`), or a folder with its `.mjz` (`export.write(car, folder)`).

### How close the model is to the real car

The model has not been checked against recordings of the real car (issue #4). Use it to develop and compare controllers, not to predict lap times. The known gaps:

- **Speed.** The motor is an RS-545 at the docs' 11,000 rpm. At full throttle the car tops out at about 3.5 m/s. The wheel joint damping, a guess, stands in for every drivetrain loss. A throttle is a share of the battery voltage, because the firmware's speed controller is not modelled (issue #118).
- **Mass and inertia.** The total of 3.0 kg comes from the spec; nobody has weighed the car. The inertia and the centre of mass height come from the URDF of an older car.
- **Tyres and suspension.** The tyre friction and the springs are estimates.
- **Steering.** Full lock puts the inner front wheel at the measured 30 degrees. The servo model swings from full left to full right in 0.16 s; people who drive the car find the real steering slower (issue #115).
- **Sensors.** They have no noise, bias or latency, and the IMU does not saturate. The LiDAR has no minimum range or missed returns. There is no camera (issue #3).

## Actuators and sensors

| Actuator | ctrl | Spec element |
|---|---|---|
| `motor` | voltage at its terminals, up to the battery's 11.1 V; + = forward | `scene.car.drive` |
| `steer_servo` | target steering angle, up to full lock, 0.4405 rad; + = left | `scene.car.steer` |

Both are MuJoCo `dcmotor`s. The motor's torque falls as the car speeds up, and 0 V brakes. The servo runs its own position loop, so it turns the wheels no faster than its free speed. Two equality constraints split the steering angle into the Ackermann angles of the front wheels.

Code above the model speaks of a throttle in [-1, 1], the share of the battery voltage: `sim.apply_command` writes a `ControlCommand`. Code binds an element to the data to read or write it, for example `data.bind_scalar(scene.car.steer).ctrl = 0.2`, so it does not depend on the order of the elements in the XML (`scene.py`).

`sensors.read(scene.car.sensors, data)` gives the IMU, the steering angles, the wheel speeds, the suspension travel and eight fixed rangefinders. Two modules compute what the real car reports:

- `encoder.py`: the speed the firmware reports, from the motor encoder's count over each 20 ms window, filtered as the firmware filters it.
- `lidar.py`: the spinning LiDAR, 1440 slots a turn at 30 Hz with the rear quarter blocked. Each physics step casts the slots that the head passed in it, from that step's pose, so the scan of a moving car is skewed as the real one is.

Every scene also gets sensors that only a simulator has (`truth.py`): the car's position, its up axis, and the number of its contacts with a wall.

`racecar_core_adapter.py` gives a controller written for the real car's `racecar_core` API the simulated car, and `examples/run_racecar_core.py` runs one unchanged.

## Batches

`batch.py` steps many copies of a scene at once, for example to train a policy. `CpuBatch` uses `mujoco.rollout` on every core, and `batch_warp.WarpBatch` uses MJX on MuJoCo Warp on a CUDA GPU (install `neoracer-mujoco[jax]` and `jax[cuda12]`). Both have the same methods:

```python
import os

import mujoco
import numpy as np

from neoracer_mujoco import assets, batch, sim
from neoracer_mujoco.physics import Physics

track = mujoco.MjSpec.from_file(os.path.join(assets.TRACK_DIR, "loop_corridor.xml"))
scene = assets.compose(track, visuals=False)
cpu = batch.CpuBatch(scene, nbatch=64, nthread=4)
state = cpu.initial([sim.settle(scene.model)] * 64, [scene] * 64, [Physics()] * 64)
ctrl = batch.command(scene.model, scene.car, np.full(64, 0.5), np.zeros(64))
state, sensordata = cpu.step(state, ctrl, nstep=12)  # (64, 12, n)
pos = batch.read(scene.model, sensordata[:, -1], scene.car.truth.pos)  # (64, 3)
cpu.close()
```

- **Same results as one car.** Each `step` starts the constraint solver cold, and a row of `CpuBatch` then steps the same, bit for bit, as one MjData stepped the same way. Warp's results are close but not equal: after ten steps from rest the cars are within 1 mm.
- **Speed.** On 4 cores, 64 rows run 7,000 to 8,000 env steps a second.
- **Leave out the visuals.** Build a batch's scenes with `visuals=False`. That drops the meshes that only show the car, and the model shrinks from 35 MB to 1.6 MB with the same physics.
- **Many tracks.** A `CpuBatch` can step each row on its own track if every track compiles to a model of the same size: generate them with `WallParams(boxes_per_wall=192)`.

`physics.py` randomizes the car for each row, as factors on the XML's values: wheel friction, chassis mass, motor torque and steering stiffness. `PhysicsRanges` sets the interval of each factor, `physics.sample` draws a `Physics`, and `initial` takes one per row.

## Track generation

```python
from neoracer_mujoco import assets
from neoracer_mujoco.track_generation import (
    NoValidTrack,
    Track,
    generate_track,
    settings_for_difficulty,
    to_spec,
)

match generate_track(7, settings_for_difficulty(2)):  # geometry only, no MuJoCo
    case Track() as track:
        scene = assets.compose(to_spec(track))
    case NoValidTrack():
        ...  # try another seed or looser settings
```

A `Track` is a closed corridor that runs counterclockwise, held as evenly spaced samples of its centerline with the tangent, normal, curvature and half-width at each. A closed random walk draws its control points, so tracks have S-bends and hairpins, not only ovals. Each candidate is checked for corners too tight for the car, a loop that touches itself and a length out of range; a failing candidate is repaired where it fails, or drawn again. The same seed gives the same track. Difficulty runs from 0, gentle and wide, to 3, tight and narrow, and `TrackSettings` holds every knob. `lap_distance(track, xy)` gives the progress along the centerline.

Only `mjcf.py` uses MuJoCo; the rest is numpy and scipy.

## Examples

Run these in a clone of the repository.

| Command | What it does |
|---|---|
| `mjpython -m examples.run` | the car in the viewer under a demo controller |
| `python -m examples.manual_drive` | drive with the arrow keys, with a chase camera |
| `python -m examples.pygame_drive.drive` | drive with WASD or a gamepad, with a LiDAR radar and a lap timer (`pip install -e ".[demo]"`) |
| `python -m examples.track_centering_demo` | the PD track-centering controller on the straight corridor |
| `python -m examples.track_generation` | generate tracks and describe them; `mjpython -m examples.track_generation --drive` drives one in the viewer |
| `python examples/run_racecar_core.py CONTROLLER.py` | run a `racecar_core` controller in the simulator |

## JavaScript package

`js/` is the same car for MuJoCo's WebAssembly build, in a browser or Node: it puts the car on a track, writes commands, and computes the encoder speed and the LiDAR scan as the Python package does. It exports the spec, typed, as `spec`. Its tests run the same cases as Python's and agree to 1e-14 m after a second of driving. See `js/README.md`.

## Validation

`validation/` holds the tests. They check the car against its contract (`contract.py`: actuator and sensor names, mass band), its physics (it rests, drives, steers the right way, gains no free energy), the spec and the files derived from it, and each module of the package.

## Citing

Cite this simulator with the "Cite this repository" button on GitHub, which reads `CITATION.cff`. `.github/citation.py` writes that file from the git history; do not edit it by hand.
