"""
This project's task, as functions of neoracer-rl's Step, which its
NeoRacerRace calls for each car that the policy drives: what the car sees,
when its race ends, and what it earns.

Observation (HeadingFree): neoracer-rl's CarObservation, with the
world-frame quaternion replaced by gravity in the body frame, so that a
policy cannot steer by its heading, which the real IMU cannot measure
without drift. In a race, the car's place follows: 0 when it leads, 1 when
it is last.
    [0:8]   LiDAR ranges (m, -1 = no hit)
    [8:11]  IMU acceleration (m/s^2)
    [11:14] IMU turn rate (rad/s)
    [14:17] gravity in the body frame, (0, 0, -1) when level
    [17]    steering angle (rad)
    [18]    the encoder's speed (m/s)
    [19]    place, in a race

Ends (RaceEnds): neoracer-rl's LapEnds over laps laps, and collision after
crash: the car touched another car.

Reward (RaceReward): neoracer-rl's ProgressReward, less time_penalty every
step, plus corner_weight times lidar_corner of extra reward per radian of
forward progress, plus overtake_reward for each car that the car passed in the
step and less as much for each car that passed it, and at a collision, less
collision_penalty for the car behind on the track, whatever the laps, and
victim_share of it for the car ahead.

lidar_corner: how close the car is to the inside of a corner, from the LiDAR
alone, so it works on any track and on the real car.
"""

from dataclasses import dataclass

import numpy as np
from neoracer_mujoco.track_generation import shortest_loop_distance
from neoracer_rl.envs.task import (
    CarObservation,
    Ends,
    LapEnds,
    Observations,
    ProgressReward,
    Rows,
    Step,
)

GRAVITY = slice(14, 17)
"""Where HeadingFree's gravity is."""


def gravity_in_body(quat: np.ndarray) -> np.ndarray:
    """The world's -Z in the body frame of each (w, x, y, z) quaternion of
    quat (rows, 4), shape (rows, 3)."""
    w, x, y, z = quat.T
    return -np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)


@dataclass(frozen=True)
class HeadingFree:
    car: CarObservation = CarObservation()

    def __call__(self, step: Step) -> Observations:
        obs = self.car(step)
        parts = [obs[:, :14], gravity_in_body(obs[:, 14:18]), obs[:, 18:]]
        if step.rivals:
            ahead = np.stack([r.ahead > 0 for r in step.rivals], -1)
            parts.append(ahead.mean(axis=-1, keepdims=True))
        return np.concatenate(parts, -1).astype(np.float32)


BEARINGS = np.deg2rad(np.arange(0, 360, 45))
"""CarObservation's beams: 0 = forward, + = left."""
LEFT, RIGHT = (1, 2, 3), (7, 6, 5)
"""The wall-facing beams of each side, front to back."""
HALF_WIDTH = 0.12
"""Half the car's width (m), from the LiDAR at its center to its side."""
TIGHTEST = 1.0
"""The curvature (1/m) of difficulty 3's tightest corner: closeness 1."""
SIGMA = 0.25
"""The clearance (m) from the car's side at which closeness falls to 1/e."""


def lidar_corner(ranges: np.ndarray) -> Rows:
    """How close each row's car is to the inside of a corner, in [0, 1], from
    its 8 LiDAR ranges (rows, 8), -1 where a beam has no return.

    For each side, the circle through the three wall-facing returns (45, 90
    and 135 degrees) gives the wall's curvature. The inside wall of a corner
    bends toward the car: its middle return is nearer than the chord of the
    other two. The outside wall bends away. Closeness is the inside wall's
    curvature, over the tightest corner's, times how near the car's side is
    to it. On 360 poses on generated tracks of difficulty 2, it correlates
    0.91 with the same quantity from the track's geometry."""

    def side(beams: tuple[int, int, int]) -> Rows:
        r = ranges[:, beams].astype(np.float64)
        a = BEARINGS[list(beams)]
        p = np.stack([r * np.cos(a), r * np.sin(a)], -1)  # (rows, 3, 2)
        p1, p2, p3 = p[:, 0], p[:, 1], p[:, 2]
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (
            p3[:, 0] - p1[:, 0]
        )
        lengths = (
            np.linalg.norm(p2 - p1, axis=1)
            * np.linalg.norm(p3 - p2, axis=1)
            * np.linalg.norm(p3 - p1, axis=1)
        )
        curvature = 2 * np.abs(cross) / np.maximum(lengths, 1e-9)
        # The middle beam meets the chord p1-p3 at t along its direction.
        u = np.array([np.cos(a[1]), np.sin(a[1])])
        d = p3 - p1
        denom = u[0] * d[:, 1] - u[1] * d[:, 0]
        t = (p1[:, 0] * d[:, 1] - p1[:, 1] * d[:, 0]) / np.where(denom == 0, np.inf, denom)
        bends_in = r[:, 1] < t
        near = np.exp(-np.maximum(r[:, 1] - HALF_WIDTH, 0.0) / SIGMA)
        seen = (r > 0).all(1)
        return np.where(seen & bends_in, np.minimum(curvature / TIGHTEST, 1.0) * near, 0.0)

    return np.maximum(side(LEFT), side(RIGHT))


@dataclass(frozen=True)
class RaceEnds:
    lap: LapEnds = LapEnds()
    laps: int = 1

    def __call__(self, step: Step) -> Ends:
        touched = np.zeros(len(step.contact), np.bool_)
        for rival in step.rivals:
            touched = touched | rival.contact
        ends: Ends = {}
        for name, mask in self.lap(step).items():
            ends[name] = mask
            if name == "crash":
                ends["collision"] = touched
        ends["lap_complete"] = step.progress.total >= 2.0 * np.pi * self.laps
        return ends


@dataclass(frozen=True)
class RaceReward:
    progress: ProgressReward = ProgressReward()
    time_penalty: float = 0.0
    overtake_reward: float = 1.0
    collision_penalty: float = 10.0
    victim_share: float = 0.2
    """The share of collision_penalty that the car ahead pays: it might have
    let the other by."""
    corner_weight: float = 0.2
    """Extra progress reward, as a share, at the inside of a corner."""

    def __call__(self, step: Step, ends: Ends) -> Rows:
        reward = self.progress(step, ends) - self.time_penalty
        corner = lidar_corner(CarObservation()(step)[:, :8])
        reward = reward + self.corner_weight * corner * np.maximum(step.progress.delta, 0.0)
        for rival in step.rivals:
            # How far ahead the rival was before the step.
            before = rival.ahead - (rival.progress.delta - step.progress.delta)
            passed = (before > 0) & (rival.ahead <= 0)
            lost = (before <= 0) & (rival.ahead > 0)
            reward = reward + self.overtake_reward * (passed.astype(float) - lost)
            # The car ahead on the track, which a car a lap down can be.
            near = shortest_loop_distance(rival.ahead, 2.0 * np.pi)
            share = np.where(near > 0, 1.0, self.victim_share)
            hit = ends["collision"] & rival.contact
            reward = reward - np.where(hit, self.collision_penalty * share, 0.0)
        return reward
