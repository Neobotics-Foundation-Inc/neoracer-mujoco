"""
How a policy's action becomes motor torque and steering on the NeoRacer --
matched to the real car's drive pipeline (neoracer_ros2_driver).

The real car takes "v <speed_m/s> <steering_deg>": full throttle is
max_speed_mps = 6.0, full steering 30 deg x a 0.625 cap = 18.75 deg (past
that the servo is driven beyond full lock and stalls against the linkage).
Its LiDAR scans 45-315 deg in the sensor frame, whose 0 deg points straight
back: a 90 deg blind sector behind the car.

So, per car:

  Throttle: the DC-motor model of examples/pygame_drive/drive_sim.py --
      torque = throttle * max_torque - back_emf * wheel_spin - rolling friction
  back_emf is set so full throttle settles at top_speed_ms: like the real
  speed command, a steady throttle gives a steady speed (0.5 -> ~3 m/s). It is
  recomputed every physics substep from the current wheel spin.

  Steering: the action maps to +-18.75 deg (max_steer_rad).

  Both follow the action at a slew rate (throttle_slew, steer_slew, full
  range per second), as in drive_sim.py: a real ESC and servo do not jump.

  Grip: every geom's sliding friction is set to surface_friction (0.8, a
  typical value for rubber RC tyres on indoor floors -- not measured on the
  NeoRacer). The MJCF and tracks use 1.5, and MuJoCo takes the larger of the
  two touching geoms', so the floor has to change with the tyres.

  LiDAR: the beam straight back (180 deg, index 4 of the 8 beams) is in the
  real blind sector and always reads "no hit" (-1). The 135 and 225 deg
  beams sit on the edge of the real window and are kept.

The old behavior (throttle straight to +-1 N.m per wheel, +-0.4 rad steering,
360 deg LiDAR) is drive=None in the env configs.

With the GitHub car (car_model.CarInfo.native: an RS-545 DC motor and a DC
servo with its own position loop, simulated by MuJoCo from the real parts'
specs), the throttle model and steering cap above are not used: the action
maps straight to the ESC voltage (throttle x battery) and the servo's target
angle (steer x its full lock), and MuJoCo provides the back-EMF and servo
speed limits. Grip and the rear-blind LiDAR still apply.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from .. import contract, sensors
from .car_model import car_info

REAR_BEAM = 4  # sensors.lidar_ranges order 0, 45, ..., 315 deg: 180 deg is index 4


@dataclass(frozen=True)
class DriveModelConfig:
    top_speed_ms: float = 6.0  # m/s at full throttle (real car: max_speed_mps)
    # The MJCF wheels carry armature 0.005 kg.m^2 (~2 kg of effective mass each
    # at r = 5 cm) and joint damping, so the drive_sim.py demo values (0.15 N.m)
    # crawl here; 0.8 N.m reaches 4.5 m/s in ~1.5 s on flat ground.
    max_torque: float = 0.8  # N.m per wheel at zero speed
    roll_friction: float = 0.02  # N.m per wheel (drive_sim.py)
    roll_friction_slope: float = 0.05  # N.m per rad/s up to roll_friction (drive_sim.py)
    # Back-EMF is computed from top_speed_ms, then scaled by this to cover the
    # tyre and joint losses the simple formula ignores (calibrated so full
    # throttle on a flat straight settles at ~top_speed_ms).
    back_emf_calibration: float = 0.92
    max_steer_rad: float = float(np.deg2rad(18.75))  # real full lock
    throttle_slew: float = 2.5  # full range per second (drive_sim.py)
    steer_slew: float = 4.0  # full range per second (drive_sim.py)
    rear_blind: bool = True  # LiDAR beam straight back reads "no hit"
    # Sliding friction of every geom (tyres, floor, walls). 0.8: typical
    # rubber on an indoor floor -- replace with a measured value when known
    # (pull the car sideways with a spring scale: force / weight). None keeps
    # the MJCF/track values (1.5).
    surface_friction: float | None = 0.8


class DriveModel:
    """One car's drive. command() each control step, apply() before every
    physics substep. back_emf_scale / torque_scale can be set per episode
    (sim2real randomization)."""

    def __init__(self, model: mujoco.MjModel, cfg: DriveModelConfig, prefix: str = ""):
        self.cfg = cfg
        self.car = car_info(model, prefix)
        self.native = self.car.native
        self.back_emf_scale = 1.0
        self.torque_scale = 1.0
        if self.native:  # the GitHub car: actions map straight to voltage and servo target
            self._drive_ids, self._steer_id = self.car.drive_ids, self.car.steer_id
            self.reset()
            return
        names = contract.EXPECTED_ACTUATORS
        self._drive_ids = np.array([model.actuator(prefix + n).id for n in names[:4]])
        self._steer_id = model.actuator(prefix + names[4]).id
        self._wheel_dofs = np.array([model.jnt_dofadr[model.actuator_trnid[a, 0]] for a in self._drive_ids])
        top_spin = cfg.top_speed_ms / sensors.WHEEL_RADIUS  # rad/s
        self.back_emf = cfg.back_emf_calibration * (cfg.max_torque - cfg.roll_friction) / top_spin
        self.back_emf_scale = 1.0  # sim2real: top speed varies inversely
        self.torque_scale = 1.0  # sim2real: motor strength (launch, not top speed)
        self.reset()

    def reset(self) -> None:
        self.throttle = 0.0
        self.steer = 0.0
        self._target = (0.0, 0.0)

    @property
    def steer_limit_rad(self) -> float:
        """rad of steering command per unit of steering action."""
        return self.car.steer_limit if self.native else self.cfg.max_steer_rad

    def command(self, throttle: float, steer: float, dt: float) -> None:
        """New action (both in [-1, 1]); the applied values slew toward it
        (legacy car) or are applied as they are (GitHub car)."""
        t = float(np.clip(throttle, -1.0, 1.0))
        s = float(np.clip(steer, -1.0, 1.0))
        if self.native:
            self.throttle, self.steer = t, s
            return
        self.throttle += float(np.clip(t - self.throttle, -self.cfg.throttle_slew * dt, self.cfg.throttle_slew * dt))
        self.steer += float(np.clip(s - self.steer, -self.cfg.steer_slew * dt, self.cfg.steer_slew * dt))

    def apply(self, data: mujoco.MjData) -> None:
        """Write wheel torques (from the current wheel spin) and the steering target."""
        if self.native:
            data.ctrl[self._drive_ids] = self.throttle * self.car.throttle_limit
            data.ctrl[self._steer_id] = self.steer * self.car.steer_limit
            return
        c = self.cfg
        spin = data.qvel[self._wheel_dofs]
        tmax = c.max_torque * self.torque_scale
        torque = self.throttle * tmax - self.back_emf * self.back_emf_scale * self.torque_scale * spin
        torque -= np.clip(spin * c.roll_friction_slope, -c.roll_friction, c.roll_friction)
        data.ctrl[self._drive_ids] = np.clip(torque, -tmax, tmax)
        data.ctrl[self._steer_id] = self.steer * c.max_steer_rad


# Car body half extents (m), the chassis collision box: a beam at angle a from
# the car's center leaves the body after min(hx/|cos a|, hy/|sin a|).
BODY_HALF = (0.223, 0.134)
BEAM_EXTENT = np.array([
    min(BODY_HALF[0] / max(abs(np.cos(a)), 1e-9), BODY_HALF[1] / max(abs(np.sin(a)), 1e-9))
    for a in np.deg2rad(np.arange(0, 360, 45))
])


def wall_clearance(ranges: np.ndarray, use: np.ndarray | None = None, extent: np.ndarray | None = None) -> float:
    """Smallest gap (m) between the car's body and what its LiDAR beams hit,
    over the beams in `use` (default: all but the rear beam, which the real
    car does not have); inf if none hit. ranges: the 8 clean LiDAR ranges;
    extent: car_model.CarInfo.beam_extent (default: the legacy car's)."""
    if use is None:
        use = np.ones(8, dtype=bool)
        use[REAR_BEAM] = False
    extent = BEAM_EXTENT if extent is None else extent
    use = use & (ranges > 0)
    return float((ranges[use] - extent[use]).min()) if use.any() else float("inf")


def clearance_cost(clearance: float, margin_m: float, weight: float) -> float:
    """Per-step cost for running closer than margin_m: weight at contact,
    0 at margin_m, linear between."""
    if weight <= 0 or clearance >= margin_m:
        return 0.0
    return weight * (1.0 - max(clearance, 0.0) / margin_m)


def apply_surface_friction(model: mujoco.MjModel, cfg: DriveModelConfig | None) -> None:
    """Set every geom's sliding friction to cfg.surface_friction (see the
    module docstring). Call once per compiled model, before sim2real reads
    its nominal values."""
    if cfg is not None and cfg.surface_friction is not None:
        model.geom_friction[:, 0] = cfg.surface_friction


def mask_rear(lidar: np.ndarray) -> np.ndarray:
    """LiDAR ranges as the real car sees them: nothing straight back."""
    lidar = lidar.copy()
    lidar[..., REAR_BEAM] = -1.0
    return lidar
