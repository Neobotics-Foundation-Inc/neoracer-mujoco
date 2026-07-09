# neoracer_sim

A small MuJoCo sandbox for the NeoRacer car. Build a scene from plain functions,
then run it. No classes for the assets — each function takes the spec and adds to it.

## Files

- `config.py` — all the constants (physics, car size, track). Edit these first.
- `car.py` — `add_car`, `add_wheel`, `add_motor`, `add_steering_servo`.
- `world.py` — `add_ground`, `add_oval_track`. Mix and match; add your own course here.
- `utils.py` — `compile_spec`, `run_viewer`, and position/velocity readers.
- `experiments/` — runnable examples.

## How a scene goes together

```python
spec = mujoco.MjSpec()
add_ground(spec)            # world
add_oval_track(spec)        # optional course
joints = add_car(spec)      # the car, returns its joints
add_motor(spec, "rear_left_motor", joints["rear_left"])   # optional actuators
model, data = compile_spec(spec)
```

The car is the first freejoint body, so `data.qvel[0:3]` is its velocity.

## Two ways to move the car

- **cvel push** — write `data.qvel` directly, no actuators. See `experiments/push.py`.
- **actuators** — `add_motor` on the rear wheels, `add_steering_servo` on the front
  knuckles, then drive with `data.ctrl`. See `experiments/drive.py`.

## Steering

Front wheels sit on a vertical-hinge knuckle. Turn them by commanding the steering
servo (`data.ctrl[2:4]`), or set the knuckle joint angle directly in `data.qpos` to
pose the wheels by hand.

## Run

From `users/amoghmpanhale/` (needs the `neoracer` conda env + `mjpython` on macOS):

```bash
mjpython -m neoracer_sim.experiments.push
mjpython -m neoracer_sim.experiments.drive
```
