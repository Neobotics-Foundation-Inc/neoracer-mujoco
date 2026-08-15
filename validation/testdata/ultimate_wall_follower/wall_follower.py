"""
Ultimate Wall Follower - entry point.

Adaptive high-speed wall following for the RACECAR Neo. No SLAM or map: every
command comes from the live LIDAR scan. Run this file to start everything.

Usage:
    cd ~/jupyter_ws/neobotics/labs/ultimate_wall_follower
    python3 wall_follower.py
    # press the START button to engage; hold the RIGHT BUMPER as a deadman
    # while it drives; release it or press BACK to stop.

Tuning:
    Parameters live in config/wall_follower.yaml. `speed` is the primary knob;
    the rest (PD gains, lookahead, gap, blend, speed scaling) have sane defaults.
    Edit the YAML and press START to reload without restarting the process.
    All control logic lives in control.py and is unit-tested in tests/.
"""

import os
import sys
import time

# Nested one level under labs/, so the library is two directories up.
sys.path.insert(0, "../../library")
import racecar_core

import control

rc = racecar_core.create_racecar()

# Parameters live in config/wall_follower.yaml and reload on START. The path is
# resolved against this file so it works regardless of the launch directory.
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "wall_follower.yaml")

# Require the right bumper to be held while driving. The car is fast and the
# hardware is fragile; the deadman keeps a human in the loop. Set False only in
# a safe, open space.
REQUIRE_DEADMAN = True

controller = control.WallFollowController()
_last_time = 0.0
_last_result = None
_last_symbol = None


def _set_turn_indicator(symbol):
    """Publish the turn glyph to the dot matrix only when it changes, so the
    text topic is not flooded at the 60 Hz loop rate."""
    global _last_symbol
    if symbol == _last_symbol:
        return
    rc.display.show_text(symbol if symbol else " ")
    _last_symbol = symbol


def _load_params():
    """Return (Config, speed) from the YAML, falling back to defaults if the
    file is missing or PyYAML is absent."""
    try:
        return control.load_config(CONFIG_PATH)
    except (FileNotFoundError, ImportError) as exc:
        print(f"[wall_follower] config load failed ({exc}); using defaults")
        return control.Config(), control.DEFAULT_SPEED


def start():
    global _last_time, _last_symbol
    cfg, speed = _load_params()
    controller.cfg = cfg
    controller.speed = speed
    controller.reset()
    _last_symbol = None
    _set_turn_indicator("")
    # Unlock the hardware speed range so SPEED maps to the full envelope.
    # No-op on library builds where set_max_speed is not yet wired through.
    rc.drive.set_max_speed(1.0)
    rc.drive.stop()
    _last_time = time.time()
    print(">> Ultimate Wall Follower ready.")
    print(f"   speed={speed}  deadman={'RIGHT BUMPER' if REQUIRE_DEADMAN else 'off'}")
    print(f"   params: {CONFIG_PATH}")


def update():
    global _last_time, _last_result

    now = time.time()
    dt = now - _last_time
    _last_time = now

    if REQUIRE_DEADMAN and not rc.controller.is_down(rc.controller.Button.RB):
        controller.reset()
        _last_result = None
        rc.drive.stop()
        _set_turn_indicator("")
        return

    scan = rc.lidar.get_samples()
    if scan is None or len(scan) == 0:
        rc.drive.stop()
        _set_turn_indicator("")
        return

    result = controller.compute(scan, dt)
    _last_result = result
    rc.drive.set_speed_angle(result.speed, result.angle)
    _set_turn_indicator(control.turn_symbol(result.angle, controller.cfg.turn_deadband))


def update_slow():
    r = _last_result
    if r is None:
        print("[wall_follower] idle (deadman released or no data)")
        return
    print(
        f"[wall_follower] mode={r.mode:4s} speed={r.speed:+.2f} angle={r.angle:+.2f} "
        f"front={r.forward_clear_cm:6.0f}cm blend={r.blend:.2f} err={r.error_cm:+.0f}cm"
    )


if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
