"""
Multi-car racing on loop_corridor.xml: 2-4 NeoRacer cars in one MuJoCo scene,
all driven by one shared policy (parameter sharing).

Each car is a separate agent. MultiRacerEnv steps every car at once; for
training, MultiRacerVecEnv runs several races in worker processes and shows
each car to Stable-Baselines3 as its own env slot, so a plain PPO trains one
policy on every car's experience.

Scene: the track plus n_cars copies of the car, attached with the name prefix
"c0_", "c1_", ... so each car keeps its own actuators and sensors. The LiDAR
rangefinders hit any geom, so every car sees the other cars.

Observation (per car, OBS_DIM = 21) -- only what a real car could know:
    [0:20]  exactly the single-car NeoRacerEnv observation (see its docstring).
            Other cars appear only where the car's own 8 LiDAR beams hit them;
            there is no privileged position of the rivals.
    [20]    place       0 = leading .. 1 = last  (place / (n_cars - 1)),
                        as this car's own LiDAR counts it (see LiDAR place).

LiDAR place: each car starts knowing its grid place (and who is ahead of and
behind it), then counts overtakes with its own 8 beams only. Beams are
grouped as front (315, 0, 45 deg), side (90, 270) and rear (135, 180, 225).
An overtake moves the passed car's bearing continuously from front to rear,
so it must cross the side beam: a rival seen front -> side -> rear means
"I passed it" (place - 1), rear -> side -> front means "it passed me"
(place + 1); front -> side -> front (pulled alongside, dropped back) is no
pass. A pass is missed only if the car never shows in the beams, e.g. hidden
behind a third car. Whether a return is a car or a wall comes from the
simulator (the geom the beam hit); a real car would tell them apart by
separating moving returns from the static track. The overtake reward and the
info["place"] scoreboard use true positions; info["place_lidar"] is the
car's own count.

Sim-to-real (MultiRacerSim2Real, train_ppo.py --cars N --sim2real): every
car gets its own draw of the upstream domain randomization (physics.py:
wheel friction, chassis mass, motor torque, steering stiffness) each race and
on respawn, plus its own steering trim, action delay and sensor biases, and
per-step sensor and action noise (sim2real.Sim2RealConfig). The wheels have
contact priority, so each car's tyre friction decides its own grip. The observation then loses the heading-leaking quaternion like the
single-car wrapper: [0:19] the 19-D sim2real observation, [19] place
(SIM2REAL_OBS_DIM = 20).

Collisions: both cars' episodes end, and the penalty follows the fault. Each
car's share is its speed toward the other car just before contact (the
physics substep before they touched): the car that drove into the other
pays collision_penalty, the car it hit pays victim_share of it (it might
have avoided it), and two cars that swerved into each other equally split
it. Penalty = collision_penalty * max(share, victim_share).

Near misses: each step, for every LiDAR beam that hits a rival, clearance =
beam range - the car's own body extent along that beam (a side beam reads
~0.13 m more than the actual gap). Below near_miss_m the car pays
near_miss_weight * (1 - clearance / near_miss_m), worst beam only: 0 at
near_miss_m, near_miss_weight at contact. Like LiDAR place, the "this
return is a car" label comes from the simulator; the range is the car's own
LiDAR reading.

Start modes (RaceConfig.start):
    "grid"      2-4 cars on the bottom straight in rows of two side by side,
                grid_lane_m apart across the track and grid_row_m between
                rows. Any car-to-car contact ends the race (crash_speed_ms 0).
    "opposite"  RoboRacer / F1TENTH head-to-head: 2 cars start half a lap
                apart, each on its own start line (the bottom and top
                straights), and the first to complete laps of its own wins.
                Contact closing slower than crash_speed_ms is a light touch:
                no penalty beyond the near-miss cost, the race goes on. The
                cars mostly cannot see each other, so place is unobservable:
                the place input is held at 0.5 and there is no overtake
                reward -- the race is decided by pace and clean passing when
                one car catches the other.

Race progress: angle swept around the loop center, as in NeoRacerEnv, plus
each car's start offset from the start line (the bottom straight, -pi/2).
This track position decides overtakes and place.

Winning is by time: every car races the same distance, laps * 2*pi of
angle from its own starting position, and all start together, so the first
car to cover it has the lowest race time -- a car starting at the back of
the grid is not penalized for it. info["lap_times_s"] has each completed
lap's time and info["race_time_s"] the total when a car finishes.

Reward (per car, RewardConfig from NeoRacerEnv plus RaceConfig):
    the single-car terms: progress, corner bonus, wall contact
    - time_penalty                  every step
    + overtake_reward * (cars this car overtook - cars that overtook it)
    one-time, when the race ends:
        + lap_bonus           the car(s) that finished first ("won")
        - rollover / wall / stuck penalty, as in NeoRacerEnv
        - collision penalty   car-to-car contact, by fault (see Collisions)
    - near-miss penalty, every step a LiDAR beam shows a rival within
      near_miss_m of the car's body (see Near misses)

Respawn (RaceConfig.respawn, the default): a car that crashes into a wall,
rolls over or gets stuck ends its own episode and is put back on the
starting grid -- in the clearest grid slot -- while the race goes on: it
rejoins behind everyone, which is the cost of crashing, with its race
distance and timer restarted. A car-to-car crash does not end anything: the
cars pay the collision penalty once, at the moment of impact (by fault, see
Collisions), and drive on; while they stay in contact the near-miss cost
keeps applying, and two cars wedged together get respawned by the stuck
check. The race -- every car at once -- ends only when a car wins or time
runs out.

Without respawn (respawn=False), the original rules: race end -- every car's
episode ends together, then a new race starts:
    a car finishes          winners "won", everyone else "lost" (terminated:
                            losing ends the episode without the bonus)
    car-to-car contact      both cars terminated: the car at fault
                            "collision", the car it hit "hit"
    a car crashes / rolls / gets stuck on its own
                            that car terminated with its reason; the other
                            cars are truncated -- cut off, not penalized, so
                            PPO bootstraps their value instead of blaming them
    max_episode_steps       everyone truncated
"""

import multiprocessing as mp
import os
from collections import deque
from dataclasses import dataclass, field, replace

import mujoco
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from .. import assets, collision, contract, sensors
from .drive_model import (REAR_BEAM, DriveModel, DriveModelConfig, apply_surface_friction, clearance_cost,
                          mask_rear, wall_clearance)
from .car_model import car_info
from .physics import PhysicsRanges, apply_physics, car_parts, sample, set_const
from .sim2real import Sim2RealConfig, gravity_in_body
from .neoracer_env import (
    _TRACK,
    RewardConfig,
    TerminationConfig,
    _advance,
    _corner_closeness,
    _is_stuck,
    _observation_space,
    _start_progress,
    NeoRacerEnvConfig,
)

OBS_DIM = 21
SIM2REAL_OBS_DIM = 20
MAX_CARS = 4

# Beam sector by beam index (0, 45, ..., 315 deg): 1 front, 0 side, -1 rear.
_SECTOR = (1, 1, 0, -1, -1, -1, 0, 1)

# Car body half extents (m), the chassis collision box: for a beam at angle a
# from the car's center, the body ends after min(hx/|cos a|, hy/|sin a|).
_BODY_HALF = (0.223, 0.134)
_BEAM_EXTENT = np.array([
    min(_BODY_HALF[0] / max(abs(np.cos(a)), 1e-9), _BODY_HALF[1] / max(abs(np.sin(a)), 1e-9))
    for a in np.deg2rad(np.arange(0, 360, 45))
])

_START_ANGLE = -np.pi / 2.0  # the start line: middle of the bottom straight
_SPAWN_Z = 0.0  # frame z; the car body's own pos in the XML lifts it
# Starting grid on the bottom straight (x forward, y across the corridor,
# which spans y in [-3, -2]): rows of two cars side by side, grid_lane_m
# apart across, grid_row_m between rows. 2 cars: one row; 3: two in front
# and one behind; 4: two rows. Grid slots are shuffled every race. The
# corridor is 1 m wide and the cars 0.27 m, so lanes can be at most ~0.6 m
# apart: 0.55 (default) leaves 28 cm between cars and ~9 cm to each wall;
# at 0.6 (~5 cm to the wall) a few degrees of start heading hit the wall.
_GRID_CENTER_Y = -2.5  # centerline of the bottom straight
# Cars start toward the back of the straight (which runs x = -2 .. 2 before
# the corners), so they have ~2.5 m of straight to sort out before turn one.
_GRID_FRONT_X = -0.5  # m; x of the front row
_OPPOSITE_X = 1.5  # m; opposite starts: at the back of each car's own straight


def _grid_slots(race: "RaceConfig", n: int) -> list[tuple[float, float, float]]:
    """(x, y, yaw) of the n start slots, before jitter."""
    if race.start == "opposite":  # own start lines: bottom straight +x, top straight -x
        return [(-_OPPOSITE_X, _GRID_CENTER_Y, 0.0), (_OPPOSITE_X, -_GRID_CENTER_Y, np.pi)][:n]
    half = race.grid_lane_m / 2
    return [
        (_GRID_FRONT_X - (k // 2) * race.grid_row_m, _GRID_CENTER_Y + (half if k % 2 == 0 else -half), 0.0)
        for k in range(n)
    ]
_CAR_COLORS = np.array(
    [[0.9, 0.15, 0.15], [0.15, 0.35, 0.95], [0.15, 0.8, 0.25], [0.95, 0.8, 0.1]]
)


@dataclass(frozen=True)
class RaceConfig:
    """Racing-specific settings. Weights are UNTUNED starting points."""

    overtake_reward: float = 1.0  # per car passed (and -1 per time passed)
    respawn: bool = True  # crashed cars respawn, car-to-car crashes don't end the race; see docstring
    collision_penalty: float = 10.0  # the car at fault in a car-to-car crash (once, at impact)
    victim_share: float = 0.2  # fraction of it the car that got hit still pays
    near_miss_m: float = 0.15  # m of clearance to a rival below which near misses cost
    near_miss_weight: float = 0.03  # per step at contact; ~ progress per step 0.035-0.06
    time_penalty: float = 0.0  # per step; rewards finishing sooner
    laps: int = 1
    start: str = "grid"  # "grid" or "opposite" (2 cars, half a lap apart); see module docstring
    grid_lane_m: float = 0.55  # m between side-by-side cars' centers: 28 cm gap, ~9 cm to each wall (0.6 max)
    grid_row_m: float = 1.2  # m between grid rows
    grid_jitter_m: float = 0.05  # m; random offset of each slot along the track
    grid_lateral_jitter_m: float = 0.02  # m; across the track (lanes are near the walls)
    grid_yaw_rad: float = 0.1  # rad; random heading of each car
    # Car-to-car contact ends the race only when the cars close at this speed
    # (m/s) or faster; slower is a light touch. 0: any contact ends it.
    crash_speed_ms: float = 0.0


@dataclass(frozen=True)
class MultiRacerConfig:
    car_path: str = assets.DEFAULT_CAR
    n_cars: int = 3
    control_hz: float = 30.0
    max_episode_steps: int = 1800  # 60 s
    lidar_high_m: float = 50.0
    termination: TerminationConfig = field(default_factory=TerminationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    race: RaceConfig = field(default_factory=RaceConfig)
    # Realistic drive per car (drive_model.py); None: the original model.
    drive: DriveModelConfig | None = field(default_factory=DriveModelConfig)


def observation_space(cfg: MultiRacerConfig, sim2real: bool = False) -> spaces.Box:
    base = _observation_space(NeoRacerEnvConfig(lidar_high_m=cfg.lidar_high_m))
    if sim2real:  # quaternion (4) -> gravity in body frame (3), as sim2real.py
        lo, hi = base.low, base.high
        low = np.concatenate([lo[0:14], np.full(3, -1.5), lo[18:20], [0.0]])
        high = np.concatenate([hi[0:14], np.full(3, 1.5), hi[18:20], [1.0]])
        return spaces.Box(low.astype(np.float32), high.astype(np.float32), dtype=np.float32)
    low = np.concatenate([base.low, [0.0]]).astype(np.float32)
    high = np.concatenate([base.high, [1.0]]).astype(np.float32)
    return spaces.Box(low=low, high=high, shape=(OBS_DIM,), dtype=np.float32)


ACTION_SPACE = spaces.Box(
    low=np.array([-1.0, -1.0], dtype=np.float32),
    high=np.array([1.0, 1.0], dtype=np.float32),
    dtype=np.float32,
)


# LiDAR target: the car's collision box tops out at z = 0.081 m in its own
# frame -- exactly the height of the LiDAR beams -- so beams skim along the
# top face of other cars and hit or miss on a hair of suspension tilt. A real
# car has parts at its LiDAR height (the LiDAR itself, electronics, body
# shell); this box stands in for them. It is invisible, massless and does not
# collide, so physics is unchanged; a car's own beams skip their own body.
_LIDAR_TARGET_HALF = (0.223, 0.134, 0.04)  # m; the chassis footprint, 4-12 cm up
_LIDAR_TARGET_Z = 0.081


def _compose(car_path: str, n_cars: int) -> mujoco.MjModel:
    scene = mujoco.MjSpec.from_file(_TRACK)
    for i in range(n_cars):
        car = mujoco.MjSpec.from_file(str(car_path))  # fresh copy per attach
        # Read before attaching: attach moves the car's elements into the scene.
        chassis_size, chassis_pos = np.array(car.geom("chassis").size), np.array(car.geom("chassis").pos)
        lidar_z = float(car.site("lidar_000").pos[2])
        frame = scene.worldbody.add_frame()
        frame.pos = np.array([_GRID_FRONT_X - (i // 2) * 1.2, _GRID_CENTER_Y + (0.3 if i % 2 == 0 else -0.3), _SPAWN_Z])
        frame.attach_body(car.body("car"), f"c{i}_", "")
        target = scene.body(f"c{i}_car").add_geom()
        target.name = f"c{i}_lidar_target"
        target.type = mujoco.mjtGeom.mjGEOM_BOX
        # The chassis footprint, centered on this car's LiDAR height (the
        # legacy car's LiDAR sits at its chassis top, the GitHub car's on top).
        target.size = np.array([chassis_size[0], chassis_size[1], _LIDAR_TARGET_HALF[2]])
        target.pos = np.array([chassis_pos[0], chassis_pos[1], lidar_z])
        target.contype, target.conaffinity = 0, 0
        target.density = 0.0
        target.group = 3  # hidden in the viewer and in recordings
        target.rgba = np.array([1.0, 1.0, 1.0, 1e-3])  # not 0: rays skip fully transparent geoms
    model = scene.compile()
    from .car_model import apply_car_options

    apply_car_options(model, car_path)
    return model


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


class _AutoReset:
    def step_autoreset(self, actions: np.ndarray):
        """step(); then start a new race if it ended for every car, or respawn
        just the cars whose episode ended (SB3 VecEnv semantics: each car is
        its own env slot)."""
        obs, rewards, term, trunc, infos = self.step(actions)
        dones = term | trunc
        for i in np.where(dones)[0]:
            infos[i]["terminal_observation"] = obs[i]
            infos[i]["TimeLimit.truncated"] = bool(trunc[i] and not term[i])
        if dones.all():
            obs, _ = self.reset()
        elif dones.any():
            for i in np.where(dones)[0]:
                obs = self.respawn(int(i))
        return obs, rewards, dones, infos


class MultiRacerEnv(_AutoReset):
    """One race: n_cars cars stepped together. Not a gym.Env (it has n_cars
    agents); use MultiRacerVecEnv to train with Stable-Baselines3.

        reset(seed) -> obs (n_cars, OBS_DIM), infos
        step(actions (n_cars, 2)) -> obs, rewards, terminated, truncated, infos
    """

    def __init__(self, config: MultiRacerConfig = MultiRacerConfig()):
        if not 2 <= config.n_cars <= MAX_CARS:
            raise ValueError(f"n_cars must be 2..{MAX_CARS}, got {config.n_cars}")
        if config.race.start not in ("grid", "opposite"):
            raise ValueError(f"race.start must be 'grid' or 'opposite', got {config.race.start!r}")
        if config.race.start == "opposite" and config.n_cars != 2:
            raise ValueError("race.start='opposite' is a 2-car format")
        self._opposite = config.race.start == "opposite"
        self.config = config
        self.n = config.n_cars
        self.model = _compose(config.car_path, self.n)
        self.data = mujoco.MjData(self.model)
        m = self.model

        self.car_bodies = [m.body(f"c{i}_car").id for i in range(self.n)]
        self._qadr = [m.jnt_qposadr[m.body_jntadr[b]] for b in self.car_bodies]
        self._dadr = [m.jnt_dofadr[m.body_jntadr[b]] for b in self.car_bodies]
        # Every qpos / dof of each car (all its joints), to respawn one car.
        self._car_qpos, self._car_dofs = [], []
        for b in self.car_bodies:
            joints = [j for j in range(m.njnt) if m.body_rootid[m.jnt_bodyid[j]] == b]
            qpos, dofs = [], []
            for j in joints:
                nq = {0: 7, 1: 4}.get(int(m.jnt_type[j]), 1)
                nv = {0: 6, 1: 3}.get(int(m.jnt_type[j]), 1)
                qpos += list(range(m.jnt_qposadr[j], m.jnt_qposadr[j] + nq))
                dofs += list(range(m.jnt_dofadr[j], m.jnt_dofadr[j] + nv))
            self._car_qpos.append(np.array(qpos))
            self._car_dofs.append(np.array(dofs))
        self._z0 = [float(m.qpos0[a + 2]) for a in self._qadr]

        car_of_body = np.full(m.nbody, -1)
        for i, b in enumerate(self.car_bodies):
            car_of_body[m.body_rootid == b] = i
        self._car_of_geom = car_of_body[m.geom_bodyid]
        self._is_wall = np.zeros(m.ngeom, dtype=bool)
        self._is_wall[list(collision.wall_geom_ids(m))] = True
        self._color_cars()

        layout = sensors.sensor_layout(m)
        self._layouts = []
        for i in range(self.n):
            p = f"c{i}_"
            self._layouts.append(tuple((name[len(p):], s) for name, s in layout if name.startswith(p)))
        self._cars_info = [car_info(m, f"c{i}_") for i in range(self.n)]
        self._drive = [ci.drive_ids for ci in self._cars_info]
        self._steer = [ci.steer_id for ci in self._cars_info]
        self._beam_extent_car = self._cars_info[0].beam_extent  # identical cars
        # LiDAR beams in sensors.lidar_ranges order: 0, 45, ..., 315 deg.
        self._beam_sites = [
            np.array([m.site(f"c{i}_lidar_{a:03d}").id for a in range(0, 360, 45)]) for i in range(self.n)
        ]
        self._beam_body = [int(m.site_bodyid[sites[0]]) for sites in self._beam_sites]
        self._torque_limit = self._cars_info[0].throttle_limit
        self._steer_limit = self._cars_info[0].steer_limit
        self._drives = ([DriveModel(m, config.drive, f"c{i}_") for i in range(self.n)]
                        if config.drive is not None else None)
        self._rear_blind = config.drive is not None and config.drive.rear_blind
        apply_surface_friction(m, config.drive)

        self._substeps = max(1, round((1.0 / config.control_hz) / float(m.opt.timestep)))
        self.effective_control_dt = self._substeps * float(m.opt.timestep)
        self._window_len = max(1, round(config.termination.stuck_window_s * config.control_hz)) + 1
        self.observation_space = observation_space(config)
        self.action_space = ACTION_SPACE
        self._rng = np.random.default_rng()
        self._step_count = 0

    # --- setup ------------------------------------------------------------

    def _color_cars(self) -> None:
        """Tint each car so the race is readable: red, blue, green, yellow."""
        m = self.model
        for g in range(m.ngeom):
            i = self._car_of_geom[g]
            if i < 0:
                continue
            c = _CAR_COLORS[i]
            if m.geom_matid[g] >= 0:  # per-car materials (attach prefixed them)
                m.mat_rgba[m.geom_matid[g], :3] = 0.4 * m.mat_rgba[m.geom_matid[g], :3] + 0.6 * c
            else:
                m.geom_rgba[g, :3] = 0.4 * m.geom_rgba[g, :3] + 0.6 * c

    # --- per-step helpers ---------------------------------------------------

    def _angle(self, i: int) -> float:
        x, y = self.data.xpos[self.car_bodies[i]][:2]
        return float(np.arctan2(y, x))

    def _contacts(self) -> tuple[np.ndarray, set]:
        """(touching a wall: one bool per car, touching car pairs (i, j), i < j)."""
        wall = np.zeros(self.n, dtype=bool)
        pairs: set = set()
        for k in range(self.data.ncon):
            g1, g2 = self.data.contact[k].geom1, self.data.contact[k].geom2
            c1, c2 = self._car_of_geom[g1], self._car_of_geom[g2]
            if c1 >= 0 and c2 >= 0 and c1 != c2:
                pairs.add((min(c1, c2), max(c1, c2)))
            elif c1 >= 0 and self._is_wall[g2]:
                wall[c1] = True
            elif c2 >= 0 and self._is_wall[g1]:
                wall[c2] = True
        return wall, pairs

    def _car_state(self) -> tuple[np.ndarray, np.ndarray]:
        """(xy position, xy world velocity) per car, from the free joints."""
        xy = np.array([self.data.xpos[b][:2] for b in self.car_bodies])
        v = np.array([self.data.qvel[a:a + 2] for a in self._dadr])
        return xy, v

    @staticmethod
    def _fault_shares(xy: np.ndarray, v: np.ndarray, i: int, j: int) -> tuple[float, float, float]:
        """Each car's share of the blame (its speed toward the other car), and
        the closing speed (m/s)."""
        u = xy[j] - xy[i]
        u = u / max(np.linalg.norm(u), 1e-9)
        a_i, a_j = max(float(v[i] @ u), 0.0), max(float(-v[j] @ u), 0.0)
        if a_i + a_j < 1e-6:
            return 0.5, 0.5, 0.0
        return a_i / (a_i + a_j), a_j / (a_i + a_j), a_i + a_j

    def _near_miss(self, ranges: np.ndarray, targets: np.ndarray) -> float:
        """Smallest clearance (m) to a rival seen by this car's beams; inf if none."""
        hit = targets >= 0
        return float((ranges[hit] - self._beam_extent_car[hit]).min()) if hit.any() else np.inf

    def _race_pos(self) -> np.ndarray:
        return np.array([self._offset[i] + self._progress[i].total for i in range(self.n)])

    def _cars_ahead(self, pos: np.ndarray) -> np.ndarray:
        return np.array([(pos > pos[i]).sum() for i in range(self.n)])

    def _beam_targets(self, i: int) -> np.ndarray:
        """For each of car i's 8 beams, the car it hits (-1: wall or nothing).
        The same ray as the rangefinder: from the site, along its +Z."""
        out = np.full(8, -1)
        geomid = np.zeros(1, dtype=np.int32)
        for b, site in enumerate(self._beam_sites[i]):
            pnt = self.data.site_xpos[site]
            vec = self.data.site_xmat[site].reshape(3, 3)[:, 2].copy()
            if mujoco.mj_ray(self.model, self.data, pnt, vec, None, 1, self._beam_body[i], geomid) >= 0:
                out[b] = self._car_of_geom[geomid[0]]
        if self._rear_blind:
            out[REAR_BEAM] = -1  # the real LiDAR does not see straight back
        return out

    def _update_lidar_place(self, targets: list[np.ndarray]) -> None:
        """Advance every car's LiDAR overtake count by one step. See the
        module docstring (LiDAR place)."""
        for i in range(self.n):
            seen: dict[int, set] = {}
            for b, k in enumerate(targets[i]):
                if k >= 0 and k != i:
                    seen.setdefault(int(k), set()).add(_SECTOR[b])
            for k, sectors in seen.items():
                if 0 in sectors:
                    self._crossing[i, k] = True
                ends = sectors - {0}
                if len(ends) != 1:
                    continue  # only side, or front and rear at once: no direction yet
                (sector,) = ends
                if self._crossing[i, k] and sector != self._last_sector[i, k]:
                    self._place_lidar[i] += -1 if sector < 0 else 1  # -1 rear: I passed it
                self._last_sector[i, k] = sector
                self._crossing[i, k] = False
        np.clip(self._place_lidar, 0, self.n - 1, out=self._place_lidar)

    @staticmethod
    def _overtakes(before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """passed[i, j]: car i overtook car j in this step."""
        return (before[:, None] <= before[None, :]) & (after[:, None] > after[None, :])

    def _observe(self) -> np.ndarray:
        obs = np.zeros((self.n, OBS_DIM), dtype=np.float32)
        for i in range(self.n):
            place = 0.5 if self._opposite else self._place_lidar[i] / (self.n - 1)  # opposite: unobservable
            r = sensors.read_layout(self._layouts[i], self.data)
            speed = sensors.motor_encoder_rad_s(r) * self._cars_info[i].wheel_radius
            obs[i] = np.concatenate([
                mask_rear(sensors.lidar_ranges(r)) if self._rear_blind else sensors.lidar_ranges(r),
                r.imu_accel, r.imu_gyro, r.imu_quat, r.steer_cmd_pos, [speed],
                [place],
            ])
        return obs

    def _infos(self, obs, wall, car, closeness, reasons) -> list[dict]:
        pos = self._race_pos()
        laps = self.config.race.laps
        return [
            {
                "car": i,
                "progress_rad": self._progress[i].total,
                "lap_frac": max(pos[i], 0.0) / (2.0 * np.pi * laps),
                "place": int(self._place[i]) + 1,  # true
                # the car's own count; None for opposite starts (unobservable)
                "place_lidar": None if self._opposite else int(self._place_lidar[i]) + 1,
                "speed_ms": float(obs[i, 19]),
                "contact": bool(wall[i]),
                "car_contact": bool(car[i]),
                "corner_closeness": float(closeness[i]),
                "episode_time_s": self._step_count * self.effective_control_dt,
                "termination_reason": reasons[i],
            }
            for i in range(self.n)
        ]

    # --- API --------------------------------------------------------------

    def respawn(self, i: int) -> np.ndarray:
        """Put car i back on the starting grid, in the slot farthest from the
        other cars, mid-race; start its new episode. Returns every car's
        observation."""
        race = self.config.race
        slots = _grid_slots(race, 2 if self._opposite else MAX_CARS)
        others = np.array([self.data.xpos[b][:2] for k, b in enumerate(self.car_bodies) if k != i])
        clearance = [np.min(np.linalg.norm(others - np.array(sl[:2]), axis=1)) for sl in slots]
        x, y, yaw = slots[int(np.argmax(clearance))]
        x += self._rng.uniform(-race.grid_jitter_m, race.grid_jitter_m)
        y += self._rng.uniform(-race.grid_lateral_jitter_m, race.grid_lateral_jitter_m)
        yaw += self._rng.uniform(-race.grid_yaw_rad, race.grid_yaw_rad)

        m, d = self.model, self.data
        d.qpos[self._car_qpos[i]] = m.qpos0[self._car_qpos[i]]  # wheels, steering, suspension
        d.qvel[self._car_dofs[i]] = 0.0
        a = self._qadr[i]
        d.qpos[a:a + 3] = (x, y, self._z0[i])
        d.qpos[a + 3:a + 7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
        mujoco.mj_forward(m, d)  # positions only: the other cars do not move

        self._progress[i] = _start_progress(self._angle(i))
        self._offset[i] = 0.0 if self._opposite else _wrap(float(np.arctan2(_GRID_CENTER_Y, x)) - _START_ANGLE)
        self._t0[i] = self._step_count * self.effective_control_dt
        if self._drives is not None:
            self._drives[i].reset()
        self._lap_times[i] = []
        self._crashes[i] = self._hits[i] = 0
        self._touching = {p for p in self._touching if i not in p}
        pos = self._race_pos()
        self._place = self._cars_ahead(pos)
        # The respawned car knows it restarted behind the field. The others see
        # the crash: a car they had ahead of them is now behind.
        for k in range(self.n):
            if k != i and self._last_sector[k, i] == 1 and pos[i] < pos[k]:
                self._place_lidar[k] -= 1
            if k != i:
                self._last_sector[k, i] = 1 if pos[i] > pos[k] else -1
                self._last_sector[i, k] = 1 if pos[k] > pos[i] else -1
                self._crossing[i, k] = self._crossing[k, i] = False
        self._place_lidar[i] = self._place[i]
        np.clip(self._place_lidar, 0, self.n - 1, out=self._place_lidar)
        return self._observe()

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        race = self.config.race
        grid = _grid_slots(race, self.n)
        slots = self._rng.permutation(self.n)  # who starts where changes every race
        for i in range(self.n):
            x, y, yaw = grid[slots[i]]
            x += self._rng.uniform(-race.grid_jitter_m, race.grid_jitter_m)
            y += self._rng.uniform(-race.grid_lateral_jitter_m, race.grid_lateral_jitter_m)
            yaw += self._rng.uniform(-race.grid_yaw_rad, race.grid_yaw_rad)
            a = self._qadr[i]
            self.data.qpos[a:a + 3] = (x, y, self._z0[i])
            self.data.qpos[a + 3:a + 7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
        mujoco.mj_forward(self.model, self.data)
        for _ in range(50):  # settle the suspension, as NeoRacerEnv does
            mujoco.mj_step(self.model, self.data)
        if self._drives is not None:
            for dm in self._drives:
                dm.reset()

        self._step_count = 0
        self._progress = [_start_progress(self._angle(i)) for i in range(self.n)]
        # Start position along the straight, not the car's own angle: on the
        # straight the inner-lane car has a larger angle around the center, so
        # a car side by side with another would count ~0.07 rad (18 cm) ahead
        # and start P1 every race. Projected onto the centerline, side-by-side
        # cars start level; the +-grid_jitter in x decides who is P1, and both
        # race the same 2*pi of angle to the finish.
        # Opposite starts: each car races from its own start line, so race
        # position is just its own progress.
        self._offset = [0.0] * self.n if self._opposite else [
            _wrap(float(np.arctan2(_GRID_CENTER_Y, self.data.xpos[self.car_bodies[i]][0])) - _START_ANGLE)
            for i in range(self.n)
        ]
        pos = self._race_pos()
        self._place = self._cars_ahead(pos)  # true; 0 = pole
        # Each driver knows its grid slot and who is ahead of / behind it.
        self._place_lidar = self._place.copy()
        self._last_sector = np.where(pos[None, :] > pos[:, None], 1, -1)  # [i, k]: k front (1) / rear (-1)
        self._crossing = np.zeros((self.n, self.n), dtype=bool)
        self._lap_times: list[list[float]] = [[] for _ in range(self.n)]
        self._t0 = np.zeros(self.n)  # when each car's own race (episode) started, s
        self._touching: set = set()  # car pairs in contact at the end of the last step
        self._crashes = np.zeros(self.n, dtype=int)  # crashes caused this episode
        self._hits = np.zeros(self.n, dtype=int)  # crashes suffered this episode
        obs = self._observe()
        zeros = np.zeros(self.n)
        return obs, self._infos(obs, zeros.astype(bool), zeros.astype(bool), zeros, [None] * self.n)

    def step(self, actions: np.ndarray):
        actions = np.clip(np.asarray(actions, dtype=np.float32).reshape(self.n, 2), -1.0, 1.0)
        for i in range(self.n):
            if self._drives is not None:  # realistic drive: torques set every substep below
                self._drives[i].command(actions[i, 0], actions[i, 1], self.effective_control_dt)
            else:
                self.data.ctrl[self._drive[i]] = actions[i, 0] * self._torque_limit
                self.data.ctrl[self._steer[i]] = actions[i, 1] * self._steer_limit

        pos_before = self._race_pos()
        wall = np.zeros(self.n, dtype=bool)
        car = np.zeros(self.n, dtype=bool)
        fault = np.zeros(self.n)  # each car's worst share of blame this step
        touch = np.zeros(self.n, dtype=bool)  # light car-to-car touch (below crash_speed_ms)
        pairs_seen: set = set()
        for _ in range(self._substeps):
            xy, v = self._car_state()  # just before this substep: before any impact
            if self._drives is not None:
                for dm in self._drives:
                    dm.apply(self.data)
            mujoco.mj_step(self.model, self.data)
            w, pairs = self._contacts()
            wall |= w
            for i, j in pairs - pairs_seen:  # first substep of each touch
                if self.config.race.respawn and (i, j) in self._touching:
                    continue  # still in contact from the last step: impact already charged
                s_i, s_j, closing = self._fault_shares(xy, v, i, j)
                if closing < self.config.race.crash_speed_ms:
                    touch[i] = touch[j] = True  # light touch: the race goes on
                    continue
                fault[i], fault[j] = max(fault[i], s_i), max(fault[j], s_j)
                car[i] = car[j] = True
            pairs_seen |= pairs
        self._touching = pairs_seen

        self._step_count += 1
        self._progress = [_advance(self._progress[i], self._angle(i), self._window_len) for i in range(self.n)]
        now = self._step_count * self.effective_control_dt
        for i in range(self.n):  # lap times, from the car's own (re)start
            done = len(self._lap_times[i])
            if self._progress[i].total >= 2.0 * np.pi * (done + 1):
                self._lap_times[i].append(now - self._t0[i] - sum(self._lap_times[i]))
        pos = self._race_pos()
        passed = self._overtakes(pos_before, pos)
        made, suffered = passed.sum(axis=1), passed.sum(axis=0)  # per car
        self._place = self._cars_ahead(pos)  # true standing (scoreboard); ties count as level
        targets = [self._beam_targets(i) for i in range(self.n)]
        if self._opposite:  # standing is unobservable; no overtake reward
            made, suffered = np.zeros(self.n), np.zeros(self.n)
        else:
            self._update_lidar_place(targets)
        term_cfg, rew_cfg, race = self.config.termination, self.config.reward, self.config.race

        # Per-car reason, in priority order.
        reasons: list[str | None] = [None] * self.n
        for i in range(self.n):
            if float(self.data.xmat[self.car_bodies[i]][8]) < term_cfg.upright_rollover_cos:
                reasons[i] = "rollover"
            elif term_cfg.terminate_on_wall and wall[i]:
                reasons[i] = "wall"
            elif car[i] and not race.respawn:
                reasons[i] = "collision" if fault[i] >= 0.5 else "hit"
            elif _is_stuck(self._progress[i], self._window_len, term_cfg.stuck_progress_rad):
                reasons[i] = "stuck"
            elif self._progress[i].total >= 2.0 * np.pi * race.laps:
                reasons[i] = "won"  # covered the race distance first: lowest time

        for i in range(self.n):
            if car[i]:  # count crashes (both modes)
                if fault[i] >= 0.5:
                    self._crashes[i] += 1
                else:
                    self._hits[i] += 1
        someone_won = "won" in reasons
        # With respawn a crashed car ends only its own episode; the race ends
        # when someone wins (or at the time limit).
        race_over = someone_won if race.respawn else any(r is not None for r in reasons)
        terminated = np.zeros(self.n, dtype=bool)
        truncated = np.zeros(self.n, dtype=bool)
        for i in range(self.n):
            if reasons[i] is not None:
                terminated[i] = True
            elif someone_won:
                reasons[i], terminated[i] = "lost", True
            elif race_over:
                truncated[i] = True  # another car's crash ended it: not this car's fault
        if not race_over and self._step_count >= self.config.max_episode_steps:
            truncated[:] = True

        closeness = np.array([
            _corner_closeness(self.data.xpos[b][:2], rew_cfg.corner_radius_m) for b in self.car_bodies
        ])
        ranges = [sensors.lidar_ranges(sensors.read_layout(self._layouts[i], self.data)) for i in range(self.n)]
        clearance = np.array([self._near_miss(ranges[i], targets[i]) for i in range(self.n)])
        rear = np.ones(8, dtype=bool)
        rear[REAR_BEAM] = False
        # Gap to walls: beams that hit something other than a car.
        wall_gap = np.array([wall_clearance(ranges[i], (targets[i] < 0) & rear, self._beam_extent_car)
                             for i in range(self.n)])
        rewards = np.zeros(self.n)
        for i in range(self.n):
            delta = self._progress[i].delta
            r = rew_cfg.progress_weight * delta
            r += rew_cfg.corner_weight * closeness[i] * max(delta, 0.0)
            r -= rew_cfg.contact_penalty * wall[i]
            r -= race.time_penalty
            r += race.overtake_reward * (made[i] - suffered[i])
            if clearance[i] < race.near_miss_m:
                r -= race.near_miss_weight * (1.0 - max(clearance[i], 0.0) / race.near_miss_m)
            r -= clearance_cost(wall_gap[i], rew_cfg.wall_margin_m, rew_cfg.wall_clearance_weight)
            if car[i]:
                r -= race.collision_penalty * max(fault[i], race.victim_share)
            r += {
                "rollover": -rew_cfg.rollover_penalty,
                "wall": -rew_cfg.wall_penalty,
                "stuck": -rew_cfg.stuck_penalty,
                "won": rew_cfg.lap_bonus,
            }.get(reasons[i], 0.0)
            rewards[i] = r

        obs = self._observe()
        infos = self._infos(obs, wall, car, closeness, reasons)
        for i in range(self.n):
            infos[i]["rival_clearance_m"] = float(clearance[i])  # inf: no rival in the beams
            infos[i]["wall_clearance_m"] = float(wall_gap[i])
            infos[i]["fault"] = float(fault[i])  # share of blame if in a collision this step
            infos[i]["car_touch"] = bool(touch[i])  # light touch that did not end the race
            infos[i]["lap_times_s"] = list(self._lap_times[i])
            infos[i]["race_time_s"] = now - self._t0[i] if reasons[i] == "won" else None
            infos[i]["crashes_caused"] = int(self._crashes[i])
            infos[i]["crashes_hit"] = int(self._hits[i])
        return obs, rewards, terminated, truncated, infos


class MultiRacerSim2Real(_AutoReset):
    """Domain randomization, sensor noise and action delay for every car of a
    MultiRacerEnv, independently per car. Same API as MultiRacerEnv; see the
    module docstring."""

    def __init__(self, env: MultiRacerEnv, config: Sim2RealConfig = Sim2RealConfig()):
        self.env = env
        self.config = config
        self.n = env.n
        m = env.model
        # The upstream domain randomization (physics.py), one draw per car;
        # nominal values captured here, before any randomization.
        self._parts = [car_parts(m, f"c{i}_") for i in range(self.n)]
        self._scratch = mujoco.MjData(m)  # for mj_setConst; never the live race
        self.observation_space = observation_space(env.config, sim2real=True)
        self.action_space = ACTION_SPACE
        self._rng = np.random.default_rng()
        self._cars: list[dict] = []
        self._queues: list[deque] = []

    def __getattr__(self, name):  # model, data, effective_control_dt, ...
        return getattr(self.env, name)

    def _u(self, r) -> float:
        lo, hi = (-r, r) if np.isscalar(r) else r
        return float(self._rng.uniform(lo, hi))

    def _sample(self) -> None:
        c = self.config
        phys, noise = c.randomize_physics, c.sensor_noise
        self._cars = [
            dict(
                physics=sample(c.physics if phys else PhysicsRanges(), self._rng),
                trim=(self._u(c.steer_trim_rad)
                      / (self.env._drives[0].steer_limit_rad if self.env._drives else self.env._steer_limit)
                      if phys else 0.0),
                delay=int(self._rng.integers(0, c.max_delay_steps + 1)) if c.action_delay else 0,
                accel_bias=self._rng.uniform(-c.accel_bias, c.accel_bias, 3) if noise else np.zeros(3),
                gyro_bias=self._rng.uniform(-c.gyro_bias, c.gyro_bias, 3) if noise else np.zeros(3),
                speed_scale=self._u(c.speed_scale) if noise else 1.0,
            )
            for _ in range(self.n)
        ]

    def _apply_physics(self) -> None:
        m = self.env.model
        for i, e in enumerate(self._cars):
            apply_physics(m, self._parts[i], e["physics"],
                          self.env._drives[i] if self.env._drives is not None else None)
        set_const(m, self._scratch)

    def _observe(self, obs: np.ndarray) -> np.ndarray:
        """(n, 21) MultiRacerEnv obs -> (n, 20) noisy, heading-free obs."""
        c, rng = self.config, self._rng
        out = np.zeros((self.n, SIM2REAL_OBS_DIM), dtype=np.float32)
        for i, e in enumerate(self._cars):
            o = obs[i].astype(np.float64)
            lidar, accel, gyro = o[0:8], o[8:11], o[11:14]
            gravity, steer, speed = gravity_in_body(o[14:18]), o[18], o[19]
            if c.sensor_noise:
                hit = lidar >= 0
                lidar = np.where(hit, lidar + rng.normal(0, 1, 8) * (c.lidar_noise_m + c.lidar_noise_frac * lidar), lidar)
                lidar = np.where(hit, np.maximum(lidar, 0.0), lidar)
                lidar = np.where(rng.random(8) < c.lidar_dropout, -1.0, lidar)
                accel = accel + e["accel_bias"] + rng.normal(0, c.accel_noise, 3)
                gyro = gyro + e["gyro_bias"] + rng.normal(0, c.gyro_noise, 3)
                gravity = gravity + rng.normal(0, c.gravity_noise, 3)
                steer += rng.normal(0, c.steer_noise)
                speed = speed * e["speed_scale"] + rng.normal(0, c.speed_noise)
            out[i] = np.concatenate([lidar, accel, gyro, gravity, [steer, speed, o[20]]])
        return np.clip(out, self.observation_space.low, self.observation_space.high)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed + 7919)  # independent of the grid's rng
        self._sample()
        if self.config.randomize_physics:
            self._apply_physics()
        obs, infos = self.env.reset(seed)  # settles the cars on the new physics
        self._queues = [deque([np.zeros(2, dtype=np.float32)] * e["delay"]) for e in self._cars]
        return self._observe(obs), infos

    def respawn(self, i: int) -> np.ndarray:
        """Respawn car i with a freshly randomized build (the track surface
        stays) and a fresh action delay queue."""
        cars = self._cars
        self._sample()
        new_car = self._cars[i]
        self._cars = cars
        self._cars[i] = new_car
        if self.config.randomize_physics:
            self._apply_physics()
        obs = self.env.respawn(i)
        self._queues[i] = deque([np.zeros(2, dtype=np.float32)] * self._cars[i]["delay"])
        return self._observe(obs)

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float32).reshape(self.n, 2).copy()
        if self.config.sensor_noise:
            actions += self._rng.normal(0.0, self.config.action_noise, actions.shape).astype(np.float32)
        applied = np.zeros_like(actions)
        for i, e in enumerate(self._cars):
            actions[i, 1] += e["trim"]
            self._queues[i].append(actions[i])
            applied[i] = self._queues[i].popleft()  # delay 0: this step's action
        obs, rewards, term, trunc, infos = self.env.step(applied)
        return self._observe(obs), rewards, term, trunc, infos


def make_race(config: MultiRacerConfig, sim2real: Sim2RealConfig | None = None, history: int = 0):
    """A MultiRacerEnv, wrapped in MultiRacerSim2Real if sim2real is given,
    and in history.MultiHistory (memory of the last `history` observations
    and actions) if history > 0."""
    env = MultiRacerEnv(config)
    if sim2real is not None:
        env = MultiRacerSim2Real(env, sim2real)
    if history > 0:
        from .history import MultiHistory

        env = MultiHistory(env, history)
    return env


def _worker(remote, parent_remote, config: MultiRacerConfig, sim2real: Sim2RealConfig | None,
            history: int = 0) -> None:
    parent_remote.close()
    env = make_race(config, sim2real, history)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                remote.send(env.step_autoreset(data))
            elif cmd == "reset":
                remote.send(env.reset(data))
            elif cmd == "close":
                break
    except KeyboardInterrupt:
        pass
    finally:
        remote.close()


class MultiRacerVecEnv(VecEnv):
    """Races as SB3 env slots, one slot per car; one shared policy acts for
    every slot -- parameter sharing.

    cars sets the size of each race, e.g. [2, 3, 4, 2, 3, 4]: races of
    different sizes run side by side, so the policy trains on every field
    size at once. (SB3 needs a fixed number of slots, so a race cannot change
    its own size between episodes.) Default: n_races races of config.n_cars.
    Races run in worker processes unless processes=False or there is one.
    """

    def __init__(self, config: MultiRacerConfig, n_races: int = 1, seed: int = 0,
                 start_method: str | None = None, sim2real: Sim2RealConfig | None = None,
                 cars: list[int] | None = None, processes: bool = True, history: int = 0):
        self.config = config
        self.race_sizes = list(cars) if cars else [config.n_cars] * n_races
        self.n_races = len(self.race_sizes)
        self._bounds = np.cumsum([0] + self.race_sizes)
        self._seed = seed
        space = observation_space(config, sim2real=sim2real is not None)
        if history > 0:
            from .history import history_space

            space = history_space(space, history)
        super().__init__(int(self._bounds[-1]), space, ACTION_SPACE)
        configs = [replace(config, n_cars=k) for k in self.race_sizes]
        self._local: list = []
        self._remotes, self._procs = [], []
        if self.n_races == 1 or not processes:
            self._local = [make_race(c, sim2real, history) for c in configs]
        else:
            ctx = mp.get_context(start_method or ("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"))
            for c in configs:
                remote, work_remote = ctx.Pipe()
                proc = ctx.Process(target=_worker, args=(work_remote, remote, c, sim2real, history), daemon=True)
                proc.start()
                work_remote.close()
                self._remotes.append(remote)
                self._procs.append(proc)
        self._actions = None

    def _call(self, cmd, per_race):
        if self._local:
            return [env.step_autoreset(d) if cmd == "step" else env.reset(d)
                    for env, d in zip(self._local, per_race)]
        for remote, data in zip(self._remotes, per_race):
            remote.send((cmd, data))
        return [remote.recv() for remote in self._remotes]

    def reset(self):
        seeds = [self._seed + 1000 * r for r in range(self.n_races)]
        self._seed += 1_000_000  # a later reset() gives new races
        results = self._call("reset", seeds)
        self.reset_infos = [info for _, infos in results for info in infos]
        return np.concatenate([obs for obs, _ in results])

    def step_async(self, actions):
        a = np.asarray(actions).reshape(self.num_envs, 2)
        self._actions = [a[lo:hi] for lo, hi in zip(self._bounds[:-1], self._bounds[1:])]

    def step_wait(self):
        results = self._call("step", self._actions)
        obs = np.concatenate([r[0] for r in results])
        rewards = np.concatenate([r[1] for r in results]).astype(np.float32)
        dones = np.concatenate([r[2] for r in results])
        infos = [info for r in results for info in r[3]]
        return obs, rewards, dones, infos

    def close(self):
        for remote in self._remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for proc in self._procs:
            proc.join(timeout=5)

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None)] * len(self._get_indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError("MultiRacerVecEnv has no per-slot env methods")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))
