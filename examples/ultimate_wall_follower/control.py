"""
Adaptive high-speed wall-following control.

Pure geometry and control math for the Ultimate Wall Follower. No ROS, OpenCV,
or hardware imports, so the logic is unit-testable against synthetic LIDAR scans.

LIDAR convention (RACECAR Neo): a scan is a 1-D array of distances in cm, indexed
clockwise from 0 deg directly ahead. A value of 0.0 means no return (ignored).
Drive convention: angle in [-1, 1], positive = right turn.

The controller blends two behaviors that share one scan:
  - wall centering: two beams per wall give the wall angle and perpendicular
    distance; the distance is projected to a lookahead point so corners register
    before the car reaches them; a PD acts on the left/right projected difference.
  - follow-the-gap: a safety bubble masks the nearest obstacle, then steering
    aims at the center of the widest free angular gap ahead.
Blend weight rises as forward clearance falls, so the car centers on open walls
and escapes through gaps in tight or broken geometry. One knob (SPEED) governs
pace; lookahead and brake thresholds scale with it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

DEFAULT_SPEED = 0.5

# Sentinel forward clearance used when no return is seen ahead (open space).
_OPEN_CM = 1.0e6


def clamp(value: float, low: float, high: float) -> float:
    return high if value > high else low if value < low else value


def remap(value: float, in_lo: float, in_hi: float, out_lo: float, out_hi: float) -> float:
    if in_hi == in_lo:
        return out_lo
    return out_lo + (out_hi - out_lo) * (value - in_lo) / (in_hi - in_lo)


def turn_symbol(angle: float, deadband: float = 0.05) -> str:
    """Dot-matrix glyph for a steering command: '<' left, '>' right, '' straight.

    angle follows the drive convention (positive = right). Within +/- deadband
    of zero the car is treated as going straight and no glyph is shown.
    """
    if angle <= -deadband:
        return "<"
    if angle >= deadband:
        return ">"
    return ""


def _signed_offset_deg(angle_deg: float) -> float:
    """Map a clockwise scan angle to a steering-signed offset from front.

    Returns degrees in (-180, 180]; positive = right of center (clockwise).
    """
    a = angle_deg % 360.0
    return a - 360.0 if a > 180.0 else a


def _window_indices(n: int, lo_deg: float, hi_deg: float) -> np.ndarray:
    """Sample indices spanning [lo_deg, hi_deg] clockwise, wrapping past 360."""
    lo = int(round((lo_deg % 360.0) * n / 360.0)) % n
    hi = int(round((hi_deg % 360.0) * n / 360.0)) % n
    if hi < lo:
        return np.concatenate((np.arange(lo, n), np.arange(0, hi + 1)))
    return np.arange(lo, hi + 1)


def average_distance(scan: np.ndarray, angle_deg: float, window_deg: float = 4.0) -> float:
    """Mean of valid (>0) returns in a window centered on angle_deg. 0.0 if none."""
    n = scan.shape[0]
    if n == 0:
        return 0.0
    half = max(0.0, window_deg) / 2.0
    idx = _window_indices(n, angle_deg - half, angle_deg + half)
    samples = scan[idx]
    valid = samples[samples > 0]
    if valid.size == 0:
        return 0.0
    return float(valid.mean())


def closest_point(scan: np.ndarray, min_deg: float, max_deg: float) -> tuple[float | None, float]:
    """(angle_deg, distance) of the nearest valid return in a window.

    Returns (None, 0.0) when the window holds no valid returns.
    """
    n = scan.shape[0]
    if n == 0:
        return None, 0.0
    idx = _window_indices(n, min_deg, max_deg)
    samples = scan[idx]
    masked = np.where(samples > 0, samples, np.inf)
    k = int(np.argmin(masked))
    if not np.isfinite(masked[k]):
        return None, 0.0
    return float(idx[k] * 360.0 / n), float(samples[k])


def wall_metrics(
    scan: np.ndarray,
    abeam_deg: float,
    forward_deg: float,
    window_deg: float,
) -> tuple[float, float] | None:
    """Perpendicular distance and heading angle to a wall from two beams.

    abeam_deg points roughly perpendicular to the wall; forward_deg points the
    same wall, tilted toward the front by theta. Returns (perp_distance_cm,
    alpha_rad) where alpha is the car heading relative to the wall (0 = parallel),
    or None if either beam lacks a valid return.

    alpha sign convention (verified by tests): positive alpha means the car nose
    is turned away from this wall, so projecting forward increases the distance.
    """
    b = average_distance(scan, abeam_deg, window_deg)
    a = average_distance(scan, forward_deg, window_deg)
    if a <= 0.0 or b <= 0.0:
        return None
    theta = math.radians(abs(_signed_offset_deg(abeam_deg - forward_deg)))
    alpha = math.atan2(a * math.cos(theta) - b, a * math.sin(theta))
    perp = b * math.cos(alpha)
    return perp, alpha


@dataclass
class Config:
    """Non-tuned constants. Defaults are the intended operating point; only the
    SPEED passed to the controller is expected to change between runs."""

    # Wall beams
    abeam_right_deg: float = 90.0
    abeam_left_deg: float = 270.0
    beam_theta_deg: float = 50.0
    beam_window_deg: float = 6.0

    # Forward clearance probe: narrow look straight ahead, so the parallel side
    # walls of a corridor are not mistaken for a frontal obstacle.
    front_window_deg: float = 14.0

    # Lookahead (cm): base + gain * SPEED
    lookahead_base_cm: float = 30.0
    lookahead_gain_cm: float = 120.0

    # Centering PD (acts on projected right-left distance difference in cm)
    kp: float = 0.005
    kd: float = 0.05

    # Follow-the-gap
    gap_arc_deg: float = 150.0
    gap_bubble_deg: float = 16.0
    gap_min_dist_cm: float = 80.0
    gap_steer_gain: float = 1.1  # radians of gap bearing -> steering, pre-clamp

    # Blend by forward clearance (cm): full wall above hi, full gap below lo
    blend_hi_cm: float = 220.0
    blend_lo_cm: float = 90.0
    blend_hi_gain_cm: float = 150.0  # * SPEED added to both thresholds

    # Adaptive speed (cm)
    speed_full_cm: float = 250.0
    speed_stop_cm: float = 45.0
    speed_min_frac: float = 0.25
    turn_slow_k: float = 0.8  # speed *= (1 - k * |angle|)
    turn_min_frac: float = 0.35

    # Output low-pass: fraction of previous command retained (0 = none)
    steer_ema: float = 0.55

    # Steering magnitude below which the turn indicator shows straight (blank)
    turn_deadband: float = 0.06

    @classmethod
    def from_mapping(cls, mapping: dict) -> "Config":
        """Build a Config from a flat or nested mapping (e.g. parsed YAML).

        Leaf keys whose names match field names override the defaults; nested
        sections are flattened; unknown keys are ignored. A top-level 'speed'
        key is not a Config field and is left for the caller to read.
        """
        flat: dict = {}

        def collect(d: dict) -> None:
            for key, value in d.items():
                if isinstance(value, dict):
                    collect(value)
                else:
                    flat[key] = value

        collect(mapping or {})
        names = set(cls.__dataclass_fields__)
        return cls(**{k: float(v) for k, v in flat.items() if k in names})


def load_config(path: str) -> tuple[Config, float]:
    """Read (Config, speed) from a YAML file.

    yaml is imported here so the rest of this module carries no third-party
    dependency and stays importable for unit tests without PyYAML installed.
    """
    import yaml

    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    speed = float(data.get("speed", DEFAULT_SPEED))
    return Config.from_mapping(data), speed


@dataclass
class Result:
    """One control step's outputs plus debug fields for logging."""

    speed: float
    angle: float
    mode: str
    forward_clear_cm: float = 0.0
    wall_angle: float = 0.0
    gap_angle: float = 0.0
    gap_bearing_deg: float = 0.0
    blend: float = 0.0
    error_cm: float = 0.0


class WallFollowController:
    """Stateful adaptive wall follower. Call compute(scan, dt) every frame."""

    def __init__(self, speed: float = DEFAULT_SPEED, config: Config | None = None) -> None:
        self.speed = speed
        self.cfg = config or Config()
        self.reset()

    def reset(self) -> None:
        # None means "no usable previous error", so the derivative term is
        # skipped on the first frame after a reset (or after re-acquiring the
        # walls). Without this, prev_error jumps 0 -> error over a tiny dt and
        # the D term spikes the steering toward the clamp on every re-engage.
        self._prev_error = None
        self._prev_steer = 0.0

    def _forward_clearance(self, scan: np.ndarray) -> float:
        dist = average_distance(scan, 0.0, self.cfg.front_window_deg)
        return _OPEN_CM if dist <= 0.0 else dist

    def _wall_steer(self, scan: np.ndarray, lookahead: float, dt: float):
        cfg = self.cfg
        right = wall_metrics(
            scan, cfg.abeam_right_deg, cfg.abeam_right_deg - cfg.beam_theta_deg, cfg.beam_window_deg
        )
        left = wall_metrics(
            scan, cfg.abeam_left_deg, cfg.abeam_left_deg + cfg.beam_theta_deg, cfg.beam_window_deg
        )
        if right is None or left is None:
            self._prev_error = None
            return None, 0.0
        d_right = right[0] + lookahead * math.sin(right[1])
        d_left = left[0] + lookahead * math.sin(left[1])
        error = d_right - d_left
        if self._prev_error is None or dt <= 0:
            deriv = 0.0
        else:
            deriv = (error - self._prev_error) / dt
        self._prev_error = error
        angle = clamp(cfg.kp * error + cfg.kd * deriv, -1.0, 1.0)
        return angle, error

    def _gap_steer(self, scan: np.ndarray):
        cfg = self.cfg
        n = scan.shape[0]
        half = cfg.gap_arc_deg / 2.0
        idx = _window_indices(n, -half, half)
        ranges = scan[idx]
        # Free if open (no return) or beyond the minimum useful gap depth.
        free = (ranges <= 0.0) | (ranges >= cfg.gap_min_dist_cm)
        valid = ranges > 0.0
        if valid.any():
            masked = np.where(valid, ranges, np.inf)
            nearest = int(np.argmin(masked))
            rad = max(1, int(cfg.gap_bubble_deg * n / 360.0))
            lo = max(0, nearest - rad)
            hi = min(ranges.shape[0], nearest + rad + 1)
            free[lo:hi] = False
        # Longest run of free samples.
        best_start = best_len = cur_start = cur_len = 0
        for i, f in enumerate(free):
            if f:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                if cur_len > best_len:
                    best_len, best_start = cur_len, cur_start
            else:
                cur_len = 0
        if best_len == 0:
            return 0.0, 0.0
        center_local = best_start + best_len // 2
        bearing_deg = _signed_offset_deg(idx[center_local] * 360.0 / n)
        angle = clamp(cfg.gap_steer_gain * math.radians(bearing_deg), -1.0, 1.0)
        return angle, bearing_deg

    def compute(self, scan, dt: float = 1.0 / 60.0) -> Result:
        scan = np.asarray(scan, dtype=float)
        if scan.shape[0] == 0 or not np.any(scan > 0.0):
            self.reset()
            return Result(0.0, 0.0, "no_data")

        cfg = self.cfg
        forward_clear = self._forward_clearance(scan)
        lookahead = cfg.lookahead_base_cm + cfg.lookahead_gain_cm * self.speed

        wall_angle, error = self._wall_steer(scan, lookahead, dt)
        gap_angle, gap_bearing = self._gap_steer(scan)

        walls_ok = wall_angle is not None
        hi = cfg.blend_hi_cm + cfg.blend_hi_gain_cm * self.speed
        lo = cfg.blend_lo_cm + cfg.blend_hi_gain_cm * self.speed
        w_clear = clamp((hi - forward_clear) / (hi - lo), 0.0, 1.0)
        blend = w_clear if walls_ok else 1.0
        wall_term = wall_angle if walls_ok else 0.0

        raw = (1.0 - blend) * wall_term + blend * gap_angle
        angle = cfg.steer_ema * self._prev_steer + (1.0 - cfg.steer_ema) * raw
        angle = clamp(angle, -1.0, 1.0)
        self._prev_steer = angle

        speed = self._speed(forward_clear, angle)
        mode = "wall" if (walls_ok and blend < 0.5) else "gap"
        return Result(
            speed=speed,
            angle=angle,
            mode=mode,
            forward_clear_cm=forward_clear,
            wall_angle=wall_term,
            gap_angle=gap_angle,
            gap_bearing_deg=gap_bearing,
            blend=blend,
            error_cm=error,
        )

    def _speed(self, forward_clear: float, angle: float) -> float:
        cfg = self.cfg
        cf = clamp(
            remap(forward_clear, cfg.speed_stop_cm, cfg.speed_full_cm, cfg.speed_min_frac, 1.0),
            cfg.speed_min_frac,
            1.0,
        )
        sf = clamp(1.0 - cfg.turn_slow_k * abs(angle), cfg.turn_min_frac, 1.0)
        return clamp(self.speed * cf * sf, -1.0, 1.0)
