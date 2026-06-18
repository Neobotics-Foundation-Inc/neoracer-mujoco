# NeoRacer MuJoCo Simulator

Clean, team-facing MuJoCo baseline for the NeoRacer 1/8-scale RC car.

## Setup

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

## Quick start

```sh
# Run the model in the viewer (macOS needs mjpython for the passive viewer):
mjpython scripts/run.py

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
├── scripts/
│   ├── run.py             — viewer launch script (mjpython entry point)
│   └── sensor_logger.py   — sensor read/print helpers
├── validation/            — pytest physics + logic conformance suite
└── docs/                  — design notes and parameter log
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

## Usage / demo scripts (`scripts/`)

`scripts/` holds runnable, hackable demos — start here to drive the car yourself
or to read sensors. They are meant to be copied and modified, not imported as a
stable API.

- **`run.py`** — launches the MuJoCo viewer and drives the car with a built-in
  demo controller (constant throttle + sinusoidal steering) so you can watch it
  move. Flip `SAFE_MODE` at the top: `True` caps speed and warns on rollover,
  `False` uses aggressive inputs. Run with `mjpython scripts/run.py`.
- **`sensor_logger.py`** — helpers to read every named sensor off a compiled model
  into a dict and pretty-print them (`read`, `wheel_speed_ms`, `print_sensors`).
  Import it from your own script to log IMU / wheel / steer / suspension / LiDAR.

Add your own demos here — a pure-pursuit follower, a keyboard teleop, a data
recorder, etc. — using `run.py` as the template for the load → step → control loop.

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

The car contract (expected actuators, required sensors, mass band, ctrl layout)
lives in `validation/_sim.py` — update it there if the contract changes.
