"""Stages 5-7: ProcTrackEnv -- reset, step, reward, termination, observation.

One compiled model for the whole env lifetime (INV-1): reset rewrites geom and
DR fields in place, never recompiles. RNG draw order in reset is fixed forever
(track seed -> DR -> spawn) so T3 determinism holds across the full rollout.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import EnvConfig, env_config_for_phase
from .generator import sample_valid_track
from .projection import project
from .walls import build_model

TIMESTEP = 0.002  # MuJoCo default; the XML sets no <option>
OBS_DIM = 23


@dataclass
class EnvState:
    s: float                    # last committed arc length (projection state)
    total_progress: float       # signed cumulative ds; lap at >= L
    prev_action: np.ndarray     # last [steer, throttle], for the action-rate penalty
    step_count: int
    ds_ring: np.ndarray         # trailing e4_window ds values (E4 stall detector)
    ring_ptr: int = 0
    last_proj: object = None     # last ProjResult, feeds _obs
    lapped: bool = False


def yaw_from_quat(q):
    """Yaw (rad) from a MuJoCo (w, x, y, z) quaternion."""
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_from_yaw(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class ProcTrackEnv:
    def __init__(self, cfg: EnvConfig = None, phase: int = 0):
        self.cfg = cfg if cfg is not None else env_config_for_phase(phase)
        self.model, self.wall_gid_start = build_model(self.cfg.walls)
        self.data = mujoco.MjData(self.model)

        self._cache_indices()
        self._cache_base_values()

        self.track = None
        self.state = None

    # ----- one-time index + pristine-value caches -----

    def _cache_indices(self):
        m = self.model
        self.drive_dof = np.array(
            [m.joint(n).dofadr[0] for n in
             ("fl_drive", "fr_drive", "rl_drive", "rr_drive")]
        )
        self.steer_qpos = m.joint("steer_input").qposadr[0]
        self.car_bid = m.body("car").id
        # wall geom id block: [start, start+2K)
        self.wall_gids = self.wall_gid_start + np.arange(2 * self.cfg.walls.K)
        self.wall_gid_lo = int(self.wall_gids[0])
        self.wall_gid_hi = int(self.wall_gids[-1])
        # floor: the single plane geom on worldbody
        self.floor_gid = int(
            np.flatnonzero(m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)[0]
        )
        # wheel collision geoms: cylinders that actually collide (contype != 0)
        wheel_bids = [m.body(n).id for n in
                      ("fl_wheel_body", "fr_wheel_body",
                       "rl_wheel_body", "rr_wheel_body")]
        self.wheel_gids = np.array([
            g for g in range(m.ngeom)
            if m.geom_bodyid[g] in wheel_bids and m.geom_contype[g] != 0
        ])

    def _cache_base_values(self):
        m = self.model
        self.base = {
            "floor_fric": m.geom_friction[self.floor_gid].copy(),
            "car_mass": float(m.body_mass[self.car_bid]),
            "car_inertia": m.body_inertia[self.car_bid].copy(),
            "wheel_fric0": m.geom_friction[self.wheel_gids, 0].copy(),
            "gainprm": m.actuator_gainprm[:, 0].copy(),
        }

    # ----- reset -----

    def reset(self, seed):
        cfg = self.cfg
        rng = np.random.default_rng(seed)

        # fixed draw order: track seed -> DR -> spawn
        track_seed = int(rng.integers(2**63))
        track = sample_valid_track(track_seed, cfg.gen)

        self._write_walls(track)
        self._apply_dr(rng)

        # spawn
        M = track.C.shape[0]
        s0 = rng.uniform(0.0, track.L)
        i = int(s0 / track.L * M) % M
        lateral = rng.uniform(-cfg.spawn_lateral_frac, cfg.spawn_lateral_frac) * track.w[i]
        yaw = np.arctan2(track.T[i, 1], track.T[i, 0])

        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        self.data.qpos[0:2] = track.C[i] + lateral * track.Nrm[i]
        self.data.qpos[3:7] = quat_from_yaw(yaw)
        mujoco.mj_forward(self.model, self.data)

        self.track = track
        self.state = EnvState(
            s=float(track.s[i]),
            total_progress=0.0,
            prev_action=np.zeros(2),
            step_count=0,
            ds_ring=np.zeros(cfg.e4_window),
        )
        # seed last_proj so the first obs has a valid lateral/j
        self.state.last_proj = project(track, self.state, self.data.qpos[0:2].copy(), cfg.proj)
        return self._obs(), {"seed": seed, "track_L": track.L}

    def _write_walls(self, track):
        from .walls import wall_params
        cfg = self.cfg.walls
        for side, base in ((+1, 0), (-1, cfg.K)):
            pos, quat, half_len = wall_params(track, side, cfg)
            gid = self.wall_gid_start + base + np.arange(cfg.K)
            self.model.geom_pos[gid] = pos
            self.model.geom_quat[gid] = quat
            self.model.geom_size[gid, 1] = half_len  # size[0]=radius untouched
            self.model.geom_rbound[gid] = cfg.radius + half_len  # refresh bound

    def _apply_dr(self, rng):
        m, cfg, base = self.model, self.cfg, self.base
        f_floor = rng.uniform(*cfg.dr_floor_fric)
        f_mass = rng.uniform(*cfg.dr_mass)
        f_wheel = rng.uniform(*cfg.dr_wheel_fric)
        f_gain = rng.uniform(*cfg.dr_gain)
        m.geom_friction[self.floor_gid] = base["floor_fric"] * f_floor
        m.body_mass[self.car_bid] = base["car_mass"] * f_mass
        m.body_inertia[self.car_bid] = base["car_inertia"] * f_mass
        m.geom_friction[self.wheel_gids, 0] = base["wheel_fric0"] * f_wheel
        m.actuator_gainprm[:, 0] = base["gainprm"] * f_gain

    # ----- step -----

    def step(self, action):
        cfg = self.cfg
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.data.ctrl[0:4] = action[1] * cfg.throttle_scale  # throttle to all wheels
        self.data.ctrl[4] = action[0] * cfg.steer_scale       # steer

        wall_contact = False
        for _ in range(cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)
            if self._wall_contact():
                wall_contact = True

        proj = project(self.track, self.state, self.data.qpos[0:2].copy(), cfg.proj)
        self.state.last_proj = proj

        reward, terminated, truncated = self._reward_and_flags(action, proj, wall_contact)

        self.state.prev_action = action
        self.state.step_count += 1
        info = {
            "wall_contact": wall_contact,
            "proj_valid": proj.valid,
            "total_progress": self.state.total_progress,
        }
        return self._obs(), reward, terminated, truncated, info

    def _wall_contact(self):
        """True if any current contact touches a wall geom (the other is a car geom;
        walls never collide the floor or each other -- same-body worldbody geoms)."""
        for c in self.data.contact[: self.data.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if self.wall_gid_lo <= g1 <= self.wall_gid_hi:
                return True
            if self.wall_gid_lo <= g2 <= self.wall_gid_hi:
                return True
        return False

    def _reward_and_flags(self, action, proj, wall_contact):
        cfg = self.cfg
        st = self.state

        # progress + E4 ring
        ds = proj.ds if proj.valid else 0.0
        st.ds_ring[st.ring_ptr % cfg.e4_window] = ds
        st.ring_ptr += 1
        st.total_progress += ds

        # reward
        da = action - st.prev_action
        reward = ds - cfg.c_wall * float(wall_contact) - cfg.c_act * float(da @ da)

        # lap (E5) -- credit R_lap once, on the step it completes
        lap_now = (not st.lapped) and st.total_progress >= self.track.L
        if lap_now:
            st.lapped = True
            reward += cfg.R_lap

        terminated, truncated = self._check_termination(proj, lap_now)
        if terminated:
            reward = 0.0  # every terminal condition zeroes the step reward
        return reward, terminated, truncated

    def _check_termination(self, proj, lap_now):
        cfg = self.cfg
        st = self.state
        j = proj.j
        w_j = self.track.w[j]
        lateral = proj.lateral
        chassis_z = float(self.data.xpos[self.car_bid, 2])
        t = st.step_count * cfg.frame_skip * TIMESTEP

        # E5 lap complete -> truncated (checked first: a completed lap isn't a crash)
        if lap_now:
            return False, True
        # E2 invalid projection
        if not proj.valid:
            return True, False
        # E1 escaped despite walls
        if abs(lateral) > w_j + cfg.e1_margin:
            return True, False
        # E3 chassis z band
        if not (cfg.e3_z_min <= chassis_z <= cfg.e3_z_max):
            return True, False
        # E4 stall: window full and trailing progress below threshold
        if st.ring_ptr >= cfg.e4_window and st.ds_ring.sum() < cfg.e4_stall_dist:
            return True, False
        # E6 time limit -> truncated
        if t >= cfg.e6_time:
            return False, True
        return False, False

    # ----- observation (Frenet frame, dim 23) -----

    def _obs(self):
        cfg = self.cfg
        track = self.track
        proj = self.state.last_proj
        j = proj.j
        d = self.data
        clip = cfg.obs_clip

        yaw = yaw_from_quat(d.qpos[3:7])
        vx, vy = float(d.qvel[0]), float(d.qvel[1])
        # world velocity into body frame (rotate by -yaw)
        c, s = np.cos(yaw), np.sin(yaw)
        vbx = c * vx + s * vy
        vby = -s * vx + c * vy

        psi_err = yaw - np.arctan2(track.T[j, 1], track.T[j, 0])
        lateral = proj.lateral if proj.valid else 0.0

        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0] = np.clip(lateral / track.w[j], -clip, clip)
        obs[1] = np.sin(psi_err)
        obs[2] = np.cos(psi_err)
        obs[3] = np.clip(vbx / cfg.obs_vel_norm, -clip, clip)
        obs[4] = np.clip(vby / cfg.obs_vel_norm, -clip, clip)
        obs[5] = np.clip(d.qvel[5] / cfg.obs_yawrate_norm, -clip, clip)
        obs[6:10] = d.qvel[self.drive_dof] / cfg.obs_wheel_norm
        obs[10] = d.qpos[self.steer_qpos] / cfg.obs_steer_norm
        obs[11:13] = self.state.prev_action

        M, L = track.C.shape[0], track.L
        k = 13
        for delta in cfg.obs_preview_deltas:
            dsamp = max(1, round(delta * M / L))
            jp = (j + dsamp) % M
            obs[k] = np.clip(track.kappa[jp], -clip, clip)
            obs[k + 1] = np.clip(track.w[jp] / cfg.obs_w_norm, -clip, clip)
            k += 2
        return obs
