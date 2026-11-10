"""
Fitting a spec's values to recorded runs of the real car.

A value that no one can measure on the car directly, such as how long its
servo takes to act on a command, is fitted: the model drives a run that the car
drove, and MuJoCo's least-squares optimizer (mujoco.minimize) finds the
value at which the model's sensors read what the car's did. The runs are in
assets/runs/, a folder per car (assets.RUN_DIR), and the fitted values are
in the car's spec, which names the fit as their source. A test holds the
spec to the fit, so a change to the model that moves a fitted value fails
until the spec has the new one.

    python -m neoracer_mujoco.fit

prints each fit next to the spec's values.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass, replace

import mujoco
import numpy as np
from mujoco import minimize

from neoracer_spec import derive_v2
from neoracer_spec.racecar_neo_v2 import RACECAR_NEO_V2, RacecarNeoV2

from . import assets, sim
from .control.steering import ControlCommand
from .scene import Scene, compile_scene

V2_RUNS = os.path.join(assets.RUN_DIR, "racecar_neo_v2")


@dataclass(frozen=True)
class Run:
    """A recorded run of a car. Its times are in s from the first command
    that moved the car."""

    commands: np.ndarray
    """A row per change of the command: its time, the speed command and the
    steering command, both in [-1, 1], the steering positive to the left."""
    odometry: np.ndarray
    """A row per message of the car's odometry: its time, the encoder's
    speed (m/s) and the yaw rate (rad/s)."""
    imu: np.ndarray
    """A row per IMU sample: its time, the acceleration (m/s^2) and the
    angular velocity (rad/s), each along the car's x (forward), y (left) and
    z (up)."""


def load(folder: str | os.PathLike[str]) -> Run:
    """The run in folder: its commands.csv, odometry.csv and imu.csv."""

    def read(name: str) -> np.ndarray:
        path = os.path.join(folder, name)
        return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)

    return Run(read("commands.csv"), read("odometry.csv"), read("imu.csv"))


def runs(folder: str | os.PathLike[str] = V2_RUNS) -> list[Run]:
    """Every run in folder, sorted by name."""
    names = sorted(os.listdir(folder))
    paths = [os.path.join(folder, name) for name in names]
    return [load(path) for path in paths if os.path.isdir(path)]


@dataclass(frozen=True)
class SteeringStep:
    """The first steering command of a run, given while the car drove
    straight."""

    t: float
    """Its time (s)."""
    steering: float
    """The steering command, in [-1, 1]."""
    speed: float
    """The speed command that the car held before and after it."""
    lead: float
    """How long the car had held that speed (s)."""


def steering_step(run: Run) -> SteeringStep:
    """The run's first steering command. ValueError if the run has none, or
    if the car was not moving when it came."""
    c = run.commands
    steered = np.flatnonzero(c[:, 2] != 0)
    if len(steered) == 0:
        raise ValueError("the run has no steering command")
    i = steered[0]
    speed = c[i, 1]
    if i == 0 or speed == 0 or c[i - 1, 1] != speed:
        raise ValueError("the run's first steering command finds the car not moving")
    held = np.flatnonzero(c[:i, 1] != speed)
    start = c[held[-1] + 1 if len(held) else 0, 0]
    return SteeringStep(
        t=float(c[i, 0]),
        steering=float(c[i, 2]),
        speed=float(speed),
        lead=float(c[i, 0] - start),
    )


@dataclass(frozen=True)
class LatencyFitParams:
    """What the steering latency's fit compares and where it searches."""

    window: tuple[float, float] = (-0.05, 0.25)
    """The times around the steering command whose yaw rates the fit
    compares (s): the car's turn-in."""
    settled: tuple[float, float] = (0.3, 1.0)
    """The times after the steering command when the yaw rate has settled
    (s). The fit divides each yaw rate by its mean over them."""
    bounds: tuple[float, float] = (0.0, 0.2)
    """The bounds of the latency (s)."""
    step: float = 1e-3
    """The step of the optimizer's finite differences (s)."""


LATENCY_FIT = LatencyFitParams()


@dataclass(frozen=True)
class LatencyFit:
    """A fit of the steering's latency to steering steps."""

    latency: float
    """The time from a steering command to the servo acting on it (s)."""
    rms: float
    """The root mean square of the difference left between the model's yaw
    rate and the car's, each as a share of its settled yaw rate."""


def racecar_neo_v2_steering_latency(
    car: RacecarNeoV2 = RACECAR_NEO_V2,
    recorded: Sequence[Run] | None = None,
    params: LatencyFitParams = LATENCY_FIT,
) -> LatencyFit:
    """The steering latency at which the yaw rate of car's model follows
    the car's gyro through the steering step of each recorded run, by
    default the full-lock circles in V2_RUNS. The search starts from the
    lower bound, so the fit does not depend on car's own latency.

    Each yaw rate is taken as a share of its settled value, so the fit sees
    when the car turns in, and not how tight a circle it ends on, which the
    linkage and the tires set.

    The latency only delays the steering step, so the fit steps a model
    with no latency once and shifts its yaw rate in time. A shift is
    continuous, where the model's delay is a whole number of physics steps.

    Limit: the fit takes the servo's free speed from car. Two steering
    steps do not tell a slower servo with less latency from a faster one
    with more: from 5.5 to 8 rad/s, each free speed with its best latency,
    85 to 93 ms, follows the gyro as well as the others."""
    if recorded is None:
        recorded = [run for run in runs() if not is_straight(run)]
    steps = [steering_step(run) for run in recorded]
    measured = [_share(run, step, params) for run, step in zip(recorded, steps)]
    simulated = [_steering_response(car, step, params) for step in steps]

    def residual(x: np.ndarray) -> np.ndarray:
        columns = []
        for latency in x.reshape(-1):
            parts = []
            for (t, yaw_rate), (at, share) in zip(simulated, measured):
                settled = _within(t, params.settled)
                model = np.interp(at - latency, t, yaw_rate / yaw_rate[settled].mean())
                parts.append(model - share)
            columns.append(np.concatenate(parts))
        return np.stack(columns, axis=1)

    lower, upper = (np.array([bound]) for bound in params.bounds)
    x, _ = minimize.least_squares(
        lower,
        residual,
        bounds=[lower, upper],
        eps=params.step,
        verbose=0,
    )
    left = residual(x)
    return LatencyFit(latency=float(x[0]), rms=float(np.sqrt(np.mean(left**2))))


def _within(t: np.ndarray, span: tuple[float, float]) -> np.ndarray:
    return (t > span[0]) & (t < span[1])


def _share(
    run: Run, step: SteeringStep, params: LatencyFitParams
) -> tuple[np.ndarray, np.ndarray]:
    """The times of the run's IMU samples in the fit's window, from the
    steering step, and their yaw rates over the settled yaw rate."""
    t = run.imu[:, 0] - step.t
    yaw_rate = run.imu[:, 6]
    compared = _within(t, params.window)
    return t[compared], yaw_rate[compared] / yaw_rate[_within(t, params.settled)].mean()


def _steering_response(
    car: RacecarNeoV2, step: SteeringStep, params: LatencyFitParams
) -> tuple[np.ndarray, np.ndarray]:
    """The times from a steering step and the gyro's yaw rates of the model
    of car with no steering latency: settled, driven straight for the
    step's lead, then steered until the settled time has passed."""
    s = car.steering
    steering = replace(s, latency=replace(s.latency, value=0.0))
    variant = replace(car, steering=steering)
    scene = _scene(variant)
    model, data = scene.model, sim.settle(scene.model)
    lock = derive_v2.steer_lock(variant)
    gyro = scene.car.sensors.imu_gyro
    dt = model.opt.timestep
    legs = (
        (sim.steps_in(step.lead, dt), 0.0),
        (sim.steps_in(params.settled[1], dt) + 1, step.steering),
    )
    steered = data.time + legs[0][0] * dt
    rows = []
    for steps, steering_command in legs:
        command = ControlCommand(throttle=step.speed, steer=steering_command * lock)
        sim.apply_command(scene.car, data, command)
        for _ in range(steps):
            mujoco.mj_step(model, data)
            # A step leaves the sensors as they read at its start.
            rows.append((data.time - dt - steered, data.bind_scalar(gyro).data[2]))
    a = np.array(rows)
    return a[:, 0], a[:, 1]


def _scene(car: RacecarNeoV2) -> Scene:
    """The model of car alone, without the meshes that only show it."""
    spec = mujoco.MjSpec.from_string(assets.racecar_neo_v2_xml(car))
    spec.modelfiledir = assets.ASSET_DIR
    assets.delete_visuals(spec)
    return compile_scene(spec)


@dataclass(frozen=True)
class SpeedStep:
    """A straight run: a speed command from rest, held, then command 0."""

    command: float
    """The speed command, in [-1, 1]."""
    seconds: float
    """How long the car held it (s)."""


def speed_step(run: Run) -> SpeedStep:
    """The run's speed step. ValueError if the run steers, or is not one
    speed command from rest followed by command 0."""
    c = run.commands
    if np.any(c[:, 2] != 0):
        raise ValueError("the run has a steering command")
    if len(c) != 3 or c[0, 1] != 0 or c[1, 1] == 0 or c[2, 1] != 0:
        raise ValueError("the run is not one speed command from rest, then 0")
    return SpeedStep(command=float(c[1, 1]), seconds=float(c[2, 0] - c[1, 0]))


def is_straight(run: Run) -> bool:
    """Whether the run never steers."""
    return not np.any(run.commands[:, 2] != 0)


@dataclass(frozen=True)
class DriveFitParams:
    """What the drive's fit compares and where it searches."""

    rise: tuple[float, ...] = (0.5, 0.9)
    """The shares of the steady speed whose first crossings after the start
    the fit compares."""
    steady: float = 1.0
    """The time before command 0 over which the speed's mean is the steady
    speed (s)."""
    smooth: int = 3
    """The samples of the running median that the speeds pass through
    before their crossings are read: the car's speed has lone spikes, the
    first at its encoder's first edge."""
    gain: tuple[float, float] = (2.0, 100.0)
    """The bounds of the speed controller's gain (N per m/s)."""
    latency: tuple[float, float] = (0.0, 0.2)
    """The bounds of the latency (s)."""
    step: tuple[float, float] = (0.1, 1e-3)
    """The steps of the optimizer's finite differences, in the gain and in
    the latency."""


DRIVE_FIT = DriveFitParams()


@dataclass(frozen=True)
class DriveFit:
    """A fit of the speed controller to straight runs."""

    gain: float
    """The speed controller's gain (N per m/s)."""
    latency: float
    """The time from a command to the speed controller (s)."""
    rms: float
    """The root mean square of the differences left between the model's
    times and the car's (s)."""


def racecar_neo_v2_drive(
    car: RacecarNeoV2 = RACECAR_NEO_V2,
    recorded: Sequence[Run] | None = None,
    params: DriveFitParams = DRIVE_FIT,
) -> DriveFit:
    """The gain of car's speed controller and its drive's latency at which
    the model's speed rises when the car's does in each recorded straight
    run, by default those in V2_RUNS. The times compared are when the speed
    first reaches each share of the steady speed in params.rise. The gain
    sets how far apart they are, and the latency shifts them all.

    The model's speed is read at the times of the car's odometry and passes
    through the same running median, so what the sampling and the median do
    to a time they do to both.

    The latency only delays the commands, so the fit steps a model with no
    latency and shifts its speed in time. The search starts from the lower
    bounds, so the fit does not depend on car's own two values.

    Limits: the car's speed peaks above its steady speed a second after the
    start and the model's does not, so the fit compares when the speed
    rises and not the speed itself. The car is slower to start than to
    stop: its speed begins to fall about 0.03 s after command 0, well
    before the fitted latency, and the model has the one latency for both."""
    if recorded is None:
        recorded = [run for run in runs() if is_straight(run)]
    steps = [speed_step(run) for run in recorded]
    measured = [
        _rise_times(run.odometry[:, 0], run.odometry[:, 1], step, params)
        for run, step in zip(recorded, steps)
    ]
    simulated: dict[float, list[tuple[np.ndarray, np.ndarray]]] = {}

    def speeds(gain: float) -> list[tuple[np.ndarray, np.ndarray]]:
        if gain not in simulated:
            simulated[gain] = _speed_responses(car, gain, steps)
        return simulated[gain]

    def residual(x: np.ndarray) -> np.ndarray:
        columns = []
        for gain, latency in (x.reshape(2, -1) * scale[:, None]).T:
            parts = []
            for run, step, (t, speed), times in zip(
                recorded, steps, speeds(gain), measured
            ):
                at = run.odometry[:, 0]
                model = np.interp(at - latency, t, speed, left=0.0)
                parts.append(_rise_times(at, model, step, params) - times)
            columns.append(np.concatenate(parts))
        return np.stack(columns, axis=1)

    scale = np.array(params.step)
    lower, upper = np.array([params.gain, params.latency]).T / scale
    x, _ = minimize.least_squares(
        lower, residual, bounds=[lower, upper], eps=1.0, verbose=0
    )
    left = residual(x)
    return DriveFit(
        gain=float(x[0] * scale[0]),
        latency=float(x[1] * scale[1]),
        rms=float(np.sqrt(np.mean(left**2))),
    )


def _rise_times(
    t: np.ndarray, speed: np.ndarray, step: SpeedStep, params: DriveFitParams
) -> np.ndarray:
    """The times from the start at which a speed sampled at t first reaches
    each of params.rise's shares of its steady speed, forward or in
    reverse as the step's command is."""
    half = params.smooth // 2
    padded = np.pad(speed, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    smooth = np.median(windows, axis=-1) * np.sign(step.command)
    held = (t > step.seconds - params.steady) & (t <= step.seconds)
    steady = smooth[held].mean()
    times = []
    for share in params.rise:
        level = share * steady
        i = int(np.argmax((t >= 0) & (smooth >= level)))
        slope = (smooth[i] - smooth[i - 1]) / (t[i] - t[i - 1])
        times.append(t[i - 1] + (level - smooth[i - 1]) / slope)
    return np.array(times)


def _speed_responses(
    car: RacecarNeoV2, gain: float, steps: Sequence[SpeedStep]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """For each speed step, the times from its start and the encoder's
    speeds of the model of car with a speed controller of gain and no drive
    latency: settled, then driven at the step's command for as long as the
    car held it. Steps of one command share the longest one's drive."""
    d = car.drive
    drive = replace(
        d, gain=replace(d.gain, value=gain), latency=replace(d.latency, value=0.0)
    )
    variant = replace(car, drive=drive)
    scene = _scene(variant)
    model = scene.model
    data = mujoco.MjData(model)
    drivetrain = scene.spec.tendon("drivetrain")
    radius = derive_v2.wheel_radius(variant)
    dt = model.opt.timestep
    longest: dict[float, float] = {}
    for step in steps:
        longest[step.command] = max(step.seconds, longest.get(step.command, 0.0))
    driven = {}
    for throttle, seconds in longest.items():
        sim.settle(model, data)
        command = ControlCommand(throttle=throttle, steer=0.0)
        sim.apply_command(scene.car, data, command)
        speeds = [0.0]
        for _ in range(sim.steps_in(seconds, dt)):
            mujoco.mj_step(model, data)
            speeds.append(data.bind_scalar(drivetrain).velocity[0] * radius)
        driven[throttle] = (np.arange(len(speeds)) * dt, np.array(speeds))
    responses = [driven[step.command] for step in steps]
    return responses


def main() -> None:
    steering = racecar_neo_v2_steering_latency()
    drive = racecar_neo_v2_drive()
    spec = RACECAR_NEO_V2
    print("RACECAR Neo V2 steering, from the full-lock circles:")
    print(f"  latency {steering.latency:.4f} s (spec {spec.steering.latency.value}),")
    print(f"  rms {steering.rms:.4f} of the settled yaw rate")
    print("RACECAR Neo V2 drive, from the straight runs:")
    print(f"  gain {drive.gain:.2f} N per m/s (spec {spec.drive.gain.value}),")
    print(f"  latency {drive.latency:.4f} s (spec {spec.drive.latency.value}),")
    print(f"  rms {drive.rms:.4f} s")


if __name__ == "__main__":
    main()
