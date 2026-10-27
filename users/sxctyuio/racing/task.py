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
step, plus overtake_reward for each car that the car passed in the step and
less as much for each car that passed it, and at a collision, less
collision_penalty for the car behind on the track, whatever the laps, and
victim_share of it for the car ahead.
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

    def __call__(self, step: Step, ends: Ends) -> Rows:
        reward = self.progress(step, ends) - self.time_penalty
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
