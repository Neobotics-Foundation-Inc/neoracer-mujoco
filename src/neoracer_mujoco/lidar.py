"""
The spinning LiDAR: 1440 slots a turn, one every 0.25 degrees, 30 turns a
second, with the rear 90 degrees blocked by the car.

The real LiDAR measures each slot when its head passes it, not the whole
turn at once. So sweep() casts, at each physics step, the rays of the slots
that the head passed in that step, from the car's pose at that step: at
360 Hz, 120 slots a step and 12 steps a scan. The scan of a moving car is
then skewed as the real one is. The rays start at the LiDAR site
(contract.LIDAR_SITE) and lie in its XY plane, so they tilt with the car.
Like a MuJoCo rangefinder, a ray does not see the geoms of the site's body.

Slot i points i * 0.25 degrees clockwise from the front, the order of a
racecar_core scan. It holds the distance (m) to the first geom along its
ray, or NO_RETURN (-1, as a rangefinder reads) for a blocked slot or no geom
within MAX_RANGE.

PLACEHOLDERS until the LiDAR model is known: MAX_RANGE and the blocked
sector, centered on the rear. The LiDAR's noise, minimum range and missed
returns are not modelled.
"""

import math
from dataclasses import dataclass

import mujoco
import numpy as np
import numpy.typing as npt

from .scene import Car

SLOTS = 1440  # slots a turn
SCAN_RATE = 30.0  # Hz, turns a second
MAX_RANGE = 12.0  # m
NO_RETURN = -1.0
FIELD = 1.5 * math.pi  # rad, the bearings it sees, centered on the front

type Scan = npt.NDArray[np.float64]  # shape (SLOTS,), m

BEARINGS = np.arange(SLOTS) * (2 * math.pi / SLOTS)  # rad, clockwise from the front
# The slots it sees: bearings in [-FIELD / 2, FIELD / 2), 1080 of them.
VISIBLE = (BEARINGS < FIELD / 2) | (BEARINGS >= 2 * math.pi - FIELD / 2)
# Each slot's ray in the site frame. Clockwise from +X is toward -Y.
_RAYS = np.stack([np.cos(BEARINGS), -np.sin(BEARINGS), np.zeros(SLOTS)], axis=1)

# The part of a slot below which the head has not reached it: a step that
# lands on a slot, up to rounding, does not measure it.
_EPS = 1e-9


@dataclass(frozen=True)
class LidarState:
    """The LiDAR between two physics steps. Start it with start()."""

    head: float  # slots of the turn in progress already measured, in [0, SLOTS)
    sweeping: Scan  # the turn in progress; slots at or past head are stale
    scan: Scan  # the last complete turn: what the LiDAR reports


def cast(
    model: mujoco.MjModel, data: mujoco.MjData, car: Car, slots: npt.NDArray[np.intp]
) -> Scan:
    """The distance (m) of each of slots from the LiDAR, at the pose of data,
    or NO_RETURN. Every slot of slots must be VISIBLE."""
    site = data.bind_scalar(car.lidar)
    rays = _RAYS[slots] @ site.xmat.reshape(3, 3).T  # world frame
    n = len(slots)
    dist, geom = np.empty(n), np.empty(n, dtype=np.int32)
    body = int(model.bind_scalar(car.lidar.parent).id)
    mujoco.mj_multiRay(
        model,
        data,
        site.xpos,
        rays.ravel(),
        None,
        True,
        body,
        geom,
        dist,
        None,
        n,
        MAX_RANGE,
    )
    return np.where((geom >= 0) & (dist <= MAX_RANGE), dist, NO_RETURN)


def _measure(
    model: mujoco.MjModel, data: mujoco.MjData, car: Car, slots: npt.NDArray[np.intp]
) -> Scan:
    """cast for any slots: a blocked one reads NO_RETURN."""
    out = np.full(len(slots), NO_RETURN)
    seen = VISIBLE[slots]
    if seen.any():
        out[seen] = cast(model, data, car, slots[seen])
    return out


def start(model: mujoco.MjModel, data: mujoco.MjData, car: Car) -> LidarState:
    """The LiDAR of a car that has stood still at the pose of data for a
    turn: its scan is complete, and a new turn begins."""
    scan = _measure(model, data, car, np.arange(SLOTS))
    return LidarState(head=0.0, sweeping=scan, scan=scan)


def sweep(
    model: mujoco.MjModel, data: mujoco.MjData, car: Car, state: LidarState, dt: float
) -> LidarState:
    """The LiDAR after a physics step of dt (s), less than a turn, that left
    the car at the pose of data: it measures the slots that its head passed."""
    end = state.head + SLOTS * SCAN_RATE * dt
    if not 0 < end - state.head < SLOTS:
        raise ValueError(f"a step of {dt} s is not between 0 and a turn")
    slots = np.arange(math.ceil(state.head - _EPS), math.ceil(end - _EPS))
    values = _measure(model, data, car, slots % SLOTS)
    sweeping, scan = state.sweeping.copy(), state.scan
    done = slots < SLOTS
    sweeping[slots[done]] = values[done]
    if end >= SLOTS - _EPS:  # the turn is complete
        scan, sweeping = sweeping, np.full(SLOTS, NO_RETURN)
        sweeping[slots[~done] - SLOTS] = values[~done]
        end -= SLOTS
    return LidarState(head=max(end, 0.0), sweeping=sweeping, scan=scan)
