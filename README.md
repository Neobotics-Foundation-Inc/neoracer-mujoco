# mujoco

## Setup

Dependencies are listed in `requirements.txt` and can be installed with either conda or uv.

### conda

```sh
conda env create -f environment.yml
conda activate neoracer-mujoco
```

### uv

```sh
uv venv
uv pip install -r requirements.txt
```

## Layout

```
assets/
  xml_assets/        # MuJoCo XML models
    car.xml          # the Ackermann-steered car
  spec_assets/       # Python helpers that build scenes via MjSpec
    car.py           # load/compile the car model
    scene.py         # drop the car onto a procedural track
    driving.py       # keyboard tele-operation
    tracks/          # procedural track builders (oval, L, checkpoints)
examples/            # runnable demos (run from the repo root)
scratch/             # local-only reference models, gitignored
```

## Examples

Run from the repository root. On macOS the MuJoCo viewer must be launched with
`mjpython` (use `python` on Linux/Windows):

```sh
mjpython examples/oval_track_demo.py   # view the oval track
mjpython examples/l_track_demo.py      # view the L-shaped track
mjpython examples/oval_drive_demo.py   # drive the oval track by keyboard
mjpython examples/l_drive_demo.py      # drive the L track by keyboard
```

### Drive controls

Throttle and steering are persistent: each press nudges the command and it holds
until you change it (like an RC car). Tap repeatedly to build up throttle / steering.

| Key       | Action                        |
| --------- | ----------------------------- |
| up / W    | accelerate (more throttle)    |
| down / S  | decelerate / reverse          |
| left / A  | steer more left               |
| right / D | steer more right              |
| space     | stop and straighten (reset)   |
