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
within range. LidarParams holds the numbers.

LidarParams (neoracer_spec.derive) defaults to the NeoRacer's LakiBeam L1. The model
sees every geom out to its range on a bright target; its range on a dark
one, its accuracy, a minimum range and missed returns are not modelled.
"""

import functools
import math
from dataclasses import dataclass

import mujoco
import numpy as np
import numpy.typing as npt

from neoracer_spec.derive import LidarParams

from .scene import Car

NO_RETURN = -1.0  # what a slot reads with nothing in range, as a rangefinder
type Scan = npt.NDArray[np.float64]  # shape (slots,), m
type Slots = npt.NDArray[np.intp]  # slot indices


@dataclass(frozen=True)
class LidarState:
    """The LiDAR between two physics steps. Start it with start()."""

    head: float  # slots of the turn in progress already measured, in [0, slots)
    sweeping: Scan  # the turn in progress; slots at or past head are stale
    scan: Scan  # the last complete turn: what the LiDAR reports


def start(
    params: LidarParams, model: mujoco.MjModel, data: mujoco.MjData, car: Car
) -> LidarState:
    """The LiDAR of a car that has stood still at the pose of data for a
    turn: its scan is complete, and a new turn begins."""
    scan = _measure(params, model, data, car, np.arange(params.slots))
    return LidarState(head=0.0, sweeping=scan, scan=scan)


# The part of a slot below which the head has not reached it: a step that
# lands on a slot, up to rounding, does not measure it.
_EPS = 1e-9


def sweep(
    params: LidarParams,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    car: Car,
    state: LidarState,
    dt: float,
) -> LidarState:
    """The LiDAR after a physics step of dt (s), less than a turn, that left
    the car at the pose of data: it measures the slots that its head passed."""
    n = params.slots
    end = state.head + n * params.scan_rate * dt
    if not 0 < end - state.head < n:
        raise ValueError(f"a step of {dt} s is not between 0 and a turn")
    slots = np.arange(math.ceil(state.head - _EPS), math.ceil(end - _EPS))
    values = _measure(params, model, data, car, slots % n)
    sweeping, scan = state.sweeping.copy(), state.scan
    done = slots < n
    sweeping[slots[done]] = values[done]
    if end >= n - _EPS:  # the turn is complete
        scan, sweeping = sweeping, np.full(n, NO_RETURN)
        sweeping[slots[~done] - n] = values[~done]
        end -= n
    return LidarState(head=max(end, 0.0), sweeping=sweeping, scan=scan)


def _measure(
    params: LidarParams,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    car: Car,
    slots: Slots,
) -> Scan:
    """cast for any slots: a blocked one reads NO_RETURN."""
    out = np.full(len(slots), NO_RETURN)
    seen = visible(params)[slots]
    if seen.any():
        out[seen] = cast(params, model, data, car, slots[seen])
    return out


@functools.cache
def visible(params: LidarParams) -> npt.NDArray[np.bool_]:
    """The slots it sees: bearings in [-field / 2, field / 2)."""
    b = bearings(params)
    return (b < params.field / 2) | (b >= 2 * math.pi - params.field / 2)


def cast(
    params: LidarParams,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    car: Car,
    slots: Slots,
) -> Scan:
    """The distance (m) of each of slots from the LiDAR, at the pose of data,
    or NO_RETURN. Every slot of slots must be visible."""
    site = data.bind_scalar(car.lidar)
    rays = _rays(params)[slots] @ site.xmat.reshape(3, 3).T  # world frame
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
        params.max_range,
    )
    return np.where((geom >= 0) & (dist <= params.max_range), dist, NO_RETURN)


@functools.cache
def _rays(params: LidarParams) -> npt.NDArray[np.float64]:
    """Each slot's ray in the site frame, shape (slots, 3). Clockwise from +X
    is toward -Y."""
    b = bearings(params)
    return np.stack([np.cos(b), -np.sin(b), np.zeros(params.slots)], axis=1)


def bearings(params: LidarParams) -> npt.NDArray[np.float64]:
    """Each slot's bearing (rad), clockwise from the front."""
    return np.arange(params.slots, dtype=np.float64) * (2 * math.pi / params.slots)
