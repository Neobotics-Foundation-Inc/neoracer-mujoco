"""
NeoRacer on procedurally generated tracks (neoracer_mujoco.track_generation).

The same car, observation (20-D), action, reward terms and terminations as
NeoRacerEnv, on a different random closed-loop track most episodes. Two
things have to change with the track, and one by request:

Progress: distance along the track's centerline (the generated Track carries
it), not angle around the loop center, which only works for the hand-made
loop. To keep the reward scale of NeoRacerEnv, a meter of progress pays what
a meter pays on loop_corridor (2*pi rad / 21.66 m), so time_penalty, the
corner bonus and the stuck check mean the same here. A lap is the track's own
length, and the lap bonus is paid once per lap as before.

Corner sensing, from the LiDAR only (no track geometry): each side's three
beams that hit that wall (45/90/135 deg left, 225/270/315 deg right) give
three wall points, and the circle through them gives the wall's curvature --
independent of the car's heading and of where it sits across the corridor.
The two walls of a corner bend the same way, so their mean is the corner:
its sign says which side is inside (positive: left), its size how tight.

    corner_closeness = min(|curvature| / corner_k_ref, 1)
                       * exp(-(inner clearance / corner_sigma_m)^2)

inner clearance = the inside side beam minus the car's half width. Checked
against the true track geometry over 3000 random poses on generated tracks:
the turn direction is right in 98% of corners, the curvature correlates 0.83
with the true one and the closeness 0.86 with the true apex closeness. It
feeds the same corner bonus as NeoRacerEnv: corner_weight * closeness *
forward progress.

Tracks: each env keeps a pool of pool_size tracks (generating one takes
0.2-1.4 s, more than a whole episode of simulation) and picks one at random
every episode; every new_track_every episodes one pool track is replaced by
a fresh one, so over a long run the policy sees hundreds of tracks. The car
starts at a random point of the track (random_start) facing forward.

Track bank (bank_path): instead of generating tracks during training, load a
saved set of pre-generated tracks (build_bank: generated in parallel once,
pickled, reused by later runs). The pool then refills from the bank -- a new
pool track every new_track_every episodes costs only composing its model
(~70 ms). Bank tracks are what the policy practices on; test with fresh
tracks (pool_size=1, new_track_every=1 generates a new one every episode),
which the policy has never seen.

With sim2real.Sim2RealWrapper: the wrapper calls select_track() first, so the
randomized physics is applied to the track model the episode will use.
"""

import multiprocessing as mp
import os
import pickle
import time
from collections import deque
from dataclasses import dataclass, field

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from neoracer_mujoco import assets, collision, sim
from .. import sensors
from ..track_generation import Track, generate_track, settings_for_difficulty, to_spec
from ..track_generation.mjcf import to_start_frame
from .neoracer_env import (
    _ACTION_HIGH,
    _ACTION_LOW,
    NeoRacerEnv,
    NeoRacerEnvConfig,
    _observation_space,
    _reward,
    _termination,
)

# Reward per meter of progress: what a meter pays on loop_corridor, whose
# centerline is 21.66 m (straights 2 x 5 + 2 x 3 m, four sqrt(2) m chamfers).
PROGRESS_PER_M = 2.0 * np.pi / 21.66

_BODY_HALF_WIDTH = 0.134  # m; the chassis collision box
_BEAMS = np.deg2rad(np.arange(0, 360, 45))  # sensors.lidar_ranges order


def _circle_curvature(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Signed curvature (1/m) of the circle through p1 -> p2 -> p3; positive
    when the path turns left."""
    a, b, c = p2 - p1, p3 - p2, p3 - p1
    cross = a[0] * b[1] - a[1] * b[0]
    return float(2.0 * cross / (np.linalg.norm(a) * np.linalg.norm(b) * np.linalg.norm(c) + 1e-9))


def lidar_corner(ranges: np.ndarray, k_ref: float = 0.5, sigma_m: float = 0.25) -> tuple[float, float]:
    """(corner closeness 0..1, estimated corner curvature 1/m, + = left) from
    the 8 LiDAR ranges alone. See the module docstring."""
    if np.any(ranges[[1, 2, 3, 5, 6, 7]] <= 0):  # a beam saw nothing
        return 0.0, 0.0
    pts = ranges[:, None] * np.stack([np.cos(_BEAMS), np.sin(_BEAMS)], axis=1)  # car frame
    left = _circle_curvature(pts[3], pts[2], pts[1])  # walking the wall back to front
    right = _circle_curvature(pts[5], pts[6], pts[7])
    curvature = float((np.clip(left, -2, 2) + np.clip(right, -2, 2)) / 2)
    inner = ranges[2] if curvature > 0 else ranges[6]
    clearance = max(float(inner) - _BODY_HALF_WIDTH, 0.0)
    closeness = min(abs(curvature) / k_ref, 1.0) * float(np.exp(-((clearance / sigma_m) ** 2)))
    return closeness, curvature


@dataclass(frozen=True)
class ProceduralEnvConfig:
    # Car, control rate, episode length, termination and reward settings,
    # exactly as NeoRacerEnv reads them.
    base: NeoRacerEnvConfig = field(default_factory=NeoRacerEnvConfig)
    difficulties: tuple[int, ...] = (0, 1, 2, 3)  # each new track draws one (0 gentle .. 3 tight)
    pool_size: int = 8
    new_track_every: int = 20  # episodes; then one pool track is replaced
    random_start: bool = True  # start anywhere on the loop (else at sample 0)
    spawn_lateral_m: float = 0.1
    spawn_yaw_rad: float = 0.1
    stuck_progress_m: float = 0.15  # m of progress needed per stuck window
    corner_k_ref: float = 0.5  # 1/m; curvature at which a corner counts fully
    corner_sigma_m: float = 0.25  # m; inner clearance at which closeness falls to 1/e
    bank_path: str | None = None  # pickled track bank (build_bank); None: generate tracks as needed


def _generate_one(args: tuple[int, int]) -> tuple[int, Track]:
    """(seed, difficulty) -> (difficulty, first valid track from seed upward)."""
    seed, difficulty = args
    while True:
        track = generate_track(seed, settings_for_difficulty(difficulty))
        if isinstance(track, Track):
            return difficulty, track
        seed += 1_000_003  # a rejected shape: try an unrelated seed


def bank_file(directory: str, difficulties, n: int, seed: int = 0) -> str:
    d = "".join(str(x) for x in sorted(difficulties))
    return os.path.join(directory, f"bank_d{d}_n{n}_s{seed}.pkl")


def build_bank(path: str, n: int, difficulties, seed: int = 0, workers: int | None = None) -> str:
    """Generate n tracks (difficulties drawn evenly) in parallel and pickle
    them to path. Returns path; reuses an existing file."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rng = np.random.default_rng(seed)
    jobs = [(int(rng.integers(2**31 - 1)), int(difficulties[k % len(difficulties)])) for k in range(n)]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers or os.cpu_count() or 1) as pool:
        bank = pool.map(_generate_one, jobs)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(bank, f)
    os.replace(path + ".tmp", path)
    print(f"track bank: {n} tracks in {time.time() - t0:.0f} s -> {path}")
    return path


_BANKS: dict[str, list] = {}


def _load_bank(path: str) -> list:
    if path not in _BANKS:
        with open(path, "rb") as f:
            _BANKS[path] = pickle.load(f)
    return _BANKS[path]


@dataclass
class _PoolTrack:
    track: Track
    difficulty: int
    model: mujoco.MjModel
    data: mujoco.MjData | None  # only the current track holds one (~15 MB arena each)
    center: np.ndarray  # (N, 2) centerline in the scene frame
    tangent: np.ndarray  # (N, 2)
    bank_index: int = -1  # position in the track bank, -1 if generated


class ProceduralRacerEnv(NeoRacerEnv):
    """See the module docstring. Same spaces as NeoRacerEnv."""

    def __init__(self, config: ProceduralEnvConfig = ProceduralEnvConfig(), render_mode: str | None = None):
        gym.Env.__init__(self)
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode!r}")
        self.render_mode = render_mode
        self.proc = config
        self.config = config.base  # what the inherited NeoRacerEnv helpers read
        self.action_space = spaces.Box(low=_ACTION_LOW, high=_ACTION_HIGH, dtype=np.float32)
        self.observation_space = _observation_space(config.base)
        self._stuck_window_ticks = max(1, round(config.base.termination.stuck_window_s * config.base.control_hz))
        self._viewer = None
        self._viewer_model = None
        self._pool: list[_PoolTrack] = []
        self._pending: _PoolTrack | None = None
        self._episodes = 0
        self._step_count = 0
        self._episode_time_s = 0.0
        self._use(self._new_track())  # a model to inspect before the first reset

    # --- tracks -------------------------------------------------------------

    def _new_track(self) -> _PoolTrack:
        rng = self.np_random
        bank_index = -1
        if self.proc.bank_path is not None:  # draw from the saved bank, avoiding tracks already in the pool
            bank = _load_bank(self.proc.bank_path)
            in_pool = {pt.bank_index for pt in self._pool}
            free = [k for k in range(len(bank)) if k not in in_pool] or list(range(len(bank)))
            bank_index = int(rng.choice(free))
            difficulty, track = bank[bank_index]
        else:
            while True:
                seed = int(rng.integers(2**31 - 1))
                difficulty = int(rng.choice(self.proc.difficulties))
                track = generate_track(seed, settings_for_difficulty(difficulty))
                if isinstance(track, Track):
                    break
        model = assets.compose(to_spec(track), self.config.car_path)
        from .car_model import apply_car_options

        apply_car_options(model, self.config.car_path)
        from .drive_model import apply_surface_friction

        apply_surface_friction(model, self.config.drive)
        center = to_start_frame(track, track.center)
        heading = np.arctan2(track.tangent[0, 1], track.tangent[0, 0])
        c, s = np.cos(-heading), np.sin(-heading)
        tangent = track.tangent @ np.array([[c, -s], [s, c]]).T
        return _PoolTrack(track, difficulty, model, None, center, tangent, bank_index)

    def _use(self, pt: _PoolTrack) -> None:
        """Make pt the current scene. Only the current track keeps simulation
        state: each MjData reserves a ~15 MB arena, which across a 50-track
        pool and 19 processes would be ~16 GB."""
        previous = getattr(self, "_current", None)
        if previous is not None and previous is not pt:
            previous.data = None
        if pt.data is None:
            pt.data = mujoco.MjData(pt.model)
        self._current = pt
        self.model, self.data = pt.model, pt.data
        self._car_body = self.model.body("car").id
        self._walls = collision.wall_geom_ids(self.model)
        self._sensor_layout = sensors.sensor_layout(self.model)
        physics_dt = float(self.model.opt.timestep)
        self._substeps_per_action = max(1, round((1.0 / self.config.control_hz) / physics_dt))
        self.effective_control_dt = self._substeps_per_action * physics_dt
        from .car_model import car_info

        self._car = car_info(self.model)
        self._torque_limit = self._car.throttle_limit
        self._steer_limit = self._car.steer_limit
        from .drive_model import DriveModel  # one per track model

        self._drive = DriveModel(self.model, self.config.drive) if self.config.drive is not None else None

    def select_track(self, seed: int | None = None) -> None:
        """Choose the next episode's track now (reset() uses it). Sim2Real
        calls this before applying its randomized physics."""
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)
        self._episodes += 1
        if len(self._pool) < self.proc.pool_size:
            self._pool.append(self._new_track())
            pt = self._pool[-1]
        else:
            if self._episodes % self.proc.new_track_every == 0:
                self._pool[int(self.np_random.integers(len(self._pool)))] = self._new_track()
            pt = self._pool[int(self.np_random.integers(len(self._pool)))]
        self._pending = pt
        self._use(pt)

    @property
    def track(self) -> Track:
        return self._current.track

    # --- progress -------------------------------------------------------------

    def _nearest_index(self, near: int | None) -> int:
        xy = self.data.xpos[self._car_body][:2]
        center = self._current.center
        n = len(center)
        if near is None:
            return int(np.argmin(np.sum((center - xy) ** 2, axis=1)))
        idx = (near + np.arange(-80, 81)) % n  # the car moves < 15 samples per step
        return int(idx[np.argmin(np.sum((center[idx] - xy) ** 2, axis=1))])

    def _advance_progress(self) -> float:
        """Move the progress tracker one step; returns this step's progress (m)."""
        n = len(self._current.center)
        new = self._nearest_index(self._idx)
        d_idx = (new - self._idx + n // 2) % n - n // 2  # signed, across the wrap
        self._idx = new
        delta = d_idx * self._current.track.total_length / n
        self._s += delta
        self._window.append(self._s)
        return delta

    # --- info -----------------------------------------------------------------

    def _build_info(self, obs, contact, upright_cos, closeness, reason):
        return {
            "progress_m": self._s,
            "lap_frac": abs(self._s) / self._current.track.total_length,
            "speed_ms": float(obs[19]),
            "upright_cos": upright_cos,
            "contact": contact,
            "corner_closeness": closeness,
            "corner_curvature_est": self._curvature_est,
            "episode_time_s": self._episode_time_s,
            "termination_reason": reason,
            "track_seed": self._current.track.seed,
            "track_difficulty": self._current.difficulty,
            "track_length_m": float(self._current.track.total_length),
        }

    def _corner(self) -> float:
        ranges = sensors.lidar_ranges(sensors.read_layout(self._sensor_layout, self.data))
        closeness, self._curvature_est = lidar_corner(ranges, self.proc.corner_k_ref, self.proc.corner_sigma_m)
        return closeness

    # --- Gymnasium API --------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self._pending is None:
            self.select_track(seed)
        else:
            gym.Env.reset(self, seed=seed)  # keeps np_random as select_track left it if seed is None
        self._pending = None

        pt, rng = self._current, self.np_random
        n = len(pt.center)
        i0 = int(rng.integers(n)) if self.proc.random_start else 0
        hw = float(pt.track.half_width[i0])
        lateral = float(rng.uniform(-1, 1)) * min(self.proc.spawn_lateral_m, max(hw - 0.2, 0.0))
        normal = np.array([-pt.tangent[i0, 1], pt.tangent[i0, 0]])
        xy = pt.center[i0] + lateral * normal
        yaw = float(np.arctan2(pt.tangent[i0, 1], pt.tangent[i0, 0]) + rng.uniform(-self.proc.spawn_yaw_rad, self.proc.spawn_yaw_rad))

        mujoco.mj_resetData(self.model, self.data)
        q = self.model.jnt_qposadr[self.model.body_jntadr[self._car_body]]
        self.data.qpos[q:q + 3] = (xy[0], xy[1], self.model.qpos0[q + 2])
        self.data.qpos[q + 3:q + 7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
        mujoco.mj_forward(self.model, self.data)
        sim.run(self.model, self.data, 0.0, sim.SETTLE_STEPS)

        self._step_count = 0
        self._episode_time_s = 0.0
        self._idx = self._nearest_index(None)
        self._s = 0.0
        self._window: deque = deque([0.0], maxlen=self._stuck_window_ticks + 1)
        self._curvature_est = 0.0
        if self._drive is not None:
            self._drive.reset()
        obs = self._build_observation()
        return obs, self._build_info(obs, False, self._upright_cos(), self._corner(), None)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), _ACTION_LOW, _ACTION_HIGH)
        self._apply_action(action)
        contact = False
        for _ in range(self._substeps_per_action):
            self._physics_step()
            if collision.car_touches_wall(self.model, self.data, self._car_body, self._walls):
                contact = True

        self._step_count += 1
        self._episode_time_s = self._step_count * self.effective_control_dt
        delta_m = self._advance_progress()
        stuck = len(self._window) == self._window.maxlen and (self._s - self._window[0]) < self.proc.stuck_progress_m

        upright_cos = self._upright_cos()
        lap_rad = 2.0 * np.pi * self._s / self._current.track.total_length  # a lap = 2*pi here
        reason = _termination(self.config.termination, upright_cos, contact, stuck, lap_rad)
        terminated = reason is not None
        truncated = (not terminated) and self._step_count >= self.config.max_episode_steps

        closeness = self._corner()
        reward = _reward(self.config.reward, delta_m * PROGRESS_PER_M, contact, closeness, reason)
        from .drive_model import clearance_cost

        clearance = self._wall_clearance()
        reward -= clearance_cost(clearance, self.config.reward.wall_margin_m, self.config.reward.wall_clearance_weight)
        obs = self._build_observation()
        info = self._build_info(obs, contact, upright_cos, closeness, reason)
        info["wall_clearance_m"] = clearance
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        if self._viewer is not None and self._viewer_model is not self.model:  # new track
            self._viewer.close()
            self._viewer = None
        if self._viewer is None:
            import mujoco.viewer as mj_viewer

            self._viewer = mj_viewer.launch_passive(self.model, self.data)
            self._viewer_model = self.model
        self._viewer.sync()
