"""
A run's settings (Setup), which scripts/train_ppo.py takes from its flags
and saves beside the policy as setup.json, so scripts/render_policy.py
replays a policy on the race it trained in; and the neoracer-rl RaceConfig
that they make.

One car (cars=1) is the single-car task. Several start on a grid at the
track's start: rows of two side by side, grid_lane_m apart across the
track and grid_row_m between rows, the front row abreast of the start.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, assert_never

from neoracer_mujoco.physics import PhysicsRanges, Range
from neoracer_mujoco.sim import SpawnRange
from neoracer_rl.envs import (
    GeneratedTracks,
    LoopCorridor,
    NeoRacerEnvConfig,
    ProgressReward,
    RaceConfig,
    Racer,
)

from .task import HeadingFree, RaceEnds, RaceReward

GENTLE = PhysicsRanges(
    wheel_friction=Range(0.75, 1.25),
    chassis_mass=Range(0.9, 1.1),
    motor_torque=Range(0.9, 1.1),
    steer_stiffness=Range(0.85, 1.15),
)
"""The physics of sim-to-real, drawn for each car at each spawn: grip
within a quarter of the model's, mass and motor within a tenth, steering
stiffness within 15%. First guesses, not measurements."""

GRID_SPAWN = SpawnRange(lateral=0.02, yaw=0.1)
"""How far a car moves off its grid slot, little, since side by side the
cars are 9 cm from a wall."""


@dataclass(frozen=True)
class Setup:
    cars: int = 1
    track: Literal["loop", "procedural"] = "loop"
    difficulty: int = 1
    """Of the generated tracks, from 0 to 3."""
    pool: int = 16
    """The generated tracks that the races are drawn from."""
    laps: int = 1
    grid_lane_m: float = 0.55
    grid_row_m: float = 1.2
    sim2real: bool = False
    """Draw each car's physics from GENTLE (and, in train_ppo.py, add sensor
    noise and action delay: sim2real.py)."""
    history: int = 0
    """The past observations and actions in the observation (history.py)."""
    time_penalty: float = 0.0
    overtake_reward: float = 1.0
    collision_penalty: float = 10.0
    victim_share: float = 0.2
    wall_penalty: float = 5.0
    stuck_penalty: float = 5.0
    rollover_penalty: float = 10.0
    lap_bonus: float = 10.0

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=1))

    @staticmethod
    def load(path: Path) -> "Setup":
        return Setup(**json.loads(path.read_text()))


def grid(setup: Setup) -> tuple[Racer, ...]:
    """A Learner in each grid slot."""
    half = setup.grid_lane_m / 2.0
    if setup.cars == 1:
        return (Racer(),)
    return tuple(
        Racer(offset=(-(k // 2) * setup.grid_row_m, half if k % 2 else -half))
        for k in range(setup.cars)
    )


def race_config(setup: Setup) -> RaceConfig:
    match setup.track:
        case "loop":
            tracks: LoopCorridor | GeneratedTracks = LoopCorridor()
        case "procedural":
            tracks = GeneratedTracks(setup.difficulty, setup.pool)
        case _:
            assert_never(setup.track)
    progress = ProgressReward(
        rollover_penalty=setup.rollover_penalty,
        crash_penalty=setup.wall_penalty,
        stuck_penalty=setup.stuck_penalty,
        lap_bonus=setup.lap_bonus,
    )
    reward = RaceReward(
        progress,
        setup.time_penalty,
        setup.overtake_reward,
        setup.collision_penalty,
        setup.victim_share,
    )
    env = NeoRacerEnvConfig(
        tracks=tracks,
        max_episode_steps=1800 * setup.laps,
        spawn=SpawnRange() if setup.cars == 1 else GRID_SPAWN,
        physics=GENTLE if setup.sim2real else PhysicsRanges(),
        observe=HeadingFree(),
        terminate=RaceEnds(laps=setup.laps),
        reward=reward,
    )
    return RaceConfig(env, grid(setup))
