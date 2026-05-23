# MuJoCo Learning Experiments

Initial MuJoCo exploration and onboarding work.

This folder contains small experiments created while learning the MuJoCo physics engine and understanding concepts needed for future NeoRacer development.

## Environment

Created with:

- Python 3.13
- uv
- MuJoCo 3.x
- NumPy

Install dependencies:

```bash
uv add mujoco numpy

Activate environment:

source .venv/bin/activate

Run examples on macOS:

mjpython <filename>.py

Note: MuJoCo passive viewer on macOS requires mjpython.

Files
01_test.py

Basic MuJoCo "Hello World"

Features:

    Creates a simple box
    Uses MjModel
    Uses MjData
    Steps physics using mj_step()
    Launches passive viewer

Expected result:

A cube falls due to gravity.

02_car.py

Simple moving body experiment.

Features:

    Ground plane
    Rectangular body
    Forward velocity using data.qvel

Expected result:

A box moves slowly across the scene.

03_simple_car_with_wheels.py

First vehicle style experiment.

Features:

    Car body
    Four wheel geometries
    Forward movement

Expected result:

A basic car-like structure moves across the scene.

Concepts learned

MuJoCo core architecture:

    MjSpec
    MjModel
    MjData
    mj_step()
    passive viewer workflow

Relationship to Unity concepts:

Unity	        MuJoCo
GameObject	    Body
Rigidbody	    Physics state
Transform	    qpos
AddForce	    qvel / control


Next goals
    Add steering
    Add wheel behavior
    Add checkpoints
    Add reset logic
    Explore RL integration
    Build toward NeoRacer environment