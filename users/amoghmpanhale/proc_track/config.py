"""All tunables for the procedural track env live here and nowhere else.

Step 1 only needs GenConfig. WallConfig / EnvConfig get added when their stages
land (see proc_track_plan.md implementation order).
# ponytail: only GenConfig for now, add Wall/Env configs when stages 4/5 arrive
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GenConfig:
    # control points
    # NOTE: deviates from plan's (r 3-6, jitter 0.8, lap 3). Measured: those
    # values give median max|kappa| ~1.06 and median L ~22 m -- zero phase-0
    # validity, repair can't close the gap. Values below give ~37% raw validity
    # at kappa_max=0.4 (sweep in step-1 session), so 20 attempts suffice.
    n_ctrl_min: int = 10
    n_ctrl_max: int = 17
    r_min: float = 6.0
    r_max: float = 9.0
    jitter_std: float = 6.0
    laplacian_rounds: int = 2
    radius_smooth_rounds: int = 2  # circular smoothing on radii (keeps size)
    min_angular_gap: float = 0.15

    # repair-then-reject loop (both HARD bounds)
    max_attempts: int = 20
    max_repair: int = 5
    repair_disp_cap: float = 0.3

    # width (half-widths)
    w_base: float = 0.50
    w_min: float = 0.35   # phase-dependent, see gen_config_for_phase
    w_max: float = 0.80

    # sampling (M % K == 0 required later for walls)
    M: int = 2048
    oversample: int = 4

    # validity thresholds
    kappa_max: float = 1.0        # phase-dependent, see gen_config_for_phase
    wk_max: float = 0.8           # V2: max(w*|kappa|) dimensionless
    # V3: two samples farther apart than this along the arc must not be closer
    # than 2*w.max()+clearance_margin in space. MUST exceed that spatial
    # clearance, else ordinary along-track neighbors on a straight self-flag.
    arc_sep_threshold: float = 2.5
    clearance_margin: float = 0.3   # ...must be >= 2*w.max() + this apart
    L_min: float = 25.0           # V4
    L_max: float = 90.0


@dataclass(frozen=True)
class WallConfig:
    # K=256 boundary points per side; M % K == 0 (M=2048). 2*K=512 capsule pool.
    K: int = 256
    radius: float = 0.03      # capsule radius; spans z 0.04-0.10, in chassis side face
    overlap: float = 1.2      # half-length = overlap*0.5*seg_len -> 20% overlap
    center_z: float = 0.07


@dataclass(frozen=True)
class ProjConfig:
    # forward-biased local window (samples). Global nearest-point is ambiguous
    # where two track sections pass near each other -- never used after spawn.
    window_back: int = 10
    window_fwd: int = 40
    # validity gate: dist > gate_scale*w[j] + gate_margin -> projection invalid
    gate_scale: float = 1.5
    gate_margin: float = 0.1


# phase -> (kappa_max, w_min); curriculum from the plan's V1 table
_PHASE_KAPPA_MAX = (0.4, 0.6, 0.8, 1.0)
_PHASE_W_MIN = (0.60, 0.50, 0.40, 0.35)


def gen_config_for_phase(phase: int) -> GenConfig:
    if phase not in (0, 1, 2, 3):
        raise ValueError(f"phase must be 0-3, got {phase}")
    return GenConfig(
        kappa_max=_PHASE_KAPPA_MAX[phase],
        w_min=_PHASE_W_MIN[phase],
    )


@dataclass(frozen=True)
class EnvConfig:
    gen: GenConfig = None
    walls: WallConfig = None
    proj: ProjConfig = None

    # control: frame_skip 10 @ 0.002 s -> 50 Hz. action [steer, throttle] in [-1,1]^2.
    frame_skip: int = 10
    throttle_scale: float = 1.0   # ctrl[0:4] = throttle * this
    steer_scale: float = 0.4      # ctrl[4]   = steer * this (rad)

    # reward: r = ds - c_wall*[wall contact] - c_act*||da||^2 + R_lap*[lap]
    c_wall: float = 0.05          # T7 calibrates; per control step in contact
    c_act: float = 0.01
    R_lap: float = 5.0

    # termination
    e1_margin: float = 0.1        # |lateral| > w[j] + margin (escaped despite walls)
    e3_z_min: float = 0.02        # chassis z band
    e3_z_max: float = 0.5
    e4_stall_dist: float = 0.2    # progress over trailing e4_window control steps
    e4_window: int = 250          # 5.0 s at 50 Hz
    e6_time: float = 60.0         # s

    # domain randomization (multiplicative on pristine base cached at startup)
    dr_floor_fric: tuple = (0.8, 1.2)
    dr_mass: tuple = (0.9, 1.1)
    dr_wheel_fric: tuple = (0.9, 1.1)
    dr_gain: tuple = (0.95, 1.05)

    # spawn
    spawn_lateral_frac: float = 0.25   # lateral U(-frac, frac) * w[i]

    # observation normalizations
    obs_vel_norm: float = 5.0
    obs_yawrate_norm: float = 5.0
    obs_wheel_norm: float = 100.0
    obs_steer_norm: float = 0.4
    obs_w_norm: float = 0.8
    obs_clip: float = 2.0
    obs_preview_deltas: tuple = (0.25, 0.5, 1.0, 2.0, 4.0)  # meters


@dataclass(frozen=True)
class DriveConfig:
    # scripted pure-pursuit + speed-PD reference controller (tests T1/T5/T7).
    # Constants tuned until T1 passes -- a classical controller failing means the
    # generator/tunables are broken, fix those, never the test.
    wheelbase: float = 0.2878
    lookahead: float = 0.6        # m ahead on the centerline
    a_lat_max: float = 4.0        # m/s^2 lateral accel budget for speed target
    kappa_ahead_dist: float = 1.5  # m window for the curvature lookahead
    kappa_floor: float = 0.05     # min |kappa| so straights don't demand infinite v
    v_min: float = 0.5
    v_max: float = 3.0
    kp_speed: float = 0.8
    wallride_bias: float = 0.6    # T7: fraction of w to bias the target toward outer wall


def env_config_for_phase(phase: int) -> EnvConfig:
    return EnvConfig(
        gen=gen_config_for_phase(phase),
        walls=WallConfig(),
        proj=ProjConfig(),
    )
