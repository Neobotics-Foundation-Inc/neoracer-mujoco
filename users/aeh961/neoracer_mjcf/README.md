# NeoRacer MuJoCo Simulator

Clean, team-facing MuJoCo baseline for the NeoRacer 1/8-scale RC car.

## Quick start

```sh
# From the repo root:
users/aeh961/.venv/bin/mjpython users/aeh961/neoracer_mjcf/scripts/run.py
```

mjpython is required on macOS for the passive viewer; plain python3 will error.

## Directory layout

```
neoracer_mjcf/
├── README.md              — this file
├── models/
│   └── neoracer.xml       — MuJoCo MJCF model (the source of truth)
├── scripts/
│   ├── run.py             — launch script (mjpython entry point)
│   └── sensor_logger.py   — sensor read/print helpers
├── validation/            — headless validation scripts (future)
├── docs/                  — design notes and parameter log (future)
└── assets/                — meshes, textures (future)
```

## Model summary (`models/neoracer.xml`)

| Property | Value | Source |
|---|---|---|
| Wheelbase | 312 mm | EXTRACTED — amogh_car.xml |
| Track width | 260 mm | EXTRACTED — amogh_car.xml |
| Wheel radius | 69 mm | EXTRACTED — URDF joint geometry |
| Chassis mass | 2.0 kg | ESTIMATED |
| Wheel mass | 0.15 kg each | ESTIMATED |
| Total mass | 2.6 kg | ESTIMATED |
| Steering | Ackermann (4th-order polynomial equality constraints) | |
| Drive | AWD — 4 independent torque motors | |
| Suspension | Coil-over approximation (slide Z joint, k=600 N/m, c=25 Ns/m) | ESTIMATED |
| Sensors | IMU (accel/gyro/quat/vel) · steer · wheel vel · susp pos · 8-beam LiDAR | |

### Actuator interface

| Index | Name | Unit | Direction |
|---|---|---|---|
| ctrl[0] | fl_motor | N·m | + = forward |
| ctrl[1] | fr_motor | N·m | + = forward |
| ctrl[2] | rl_motor | N·m | + = forward |
| ctrl[3] | rr_motor | N·m | + = forward |
| ctrl[4] | steer_servo | rad | + = left turn |

## Known parameter gaps (pending hardware measurement)

The following values come from the CAD-derived URDF in the hardware repository
(`osracer-description/urdf/osracer.urdf`) and are ready to apply once approved:

| Parameter | Current (ESTIMATED) | URDF (EXTRACTED) | Delta |
|---|---|---|---|
| Wheelbase | 312 mm | 288 mm | −24 mm |
| Track width | 260 mm | 235 mm | −25 mm |
| Chassis mass | 2000 g | 1621 g | −23% |
| Wheel mass | 150 g each | ~58 g each | −61% |
| Total mass | 2600 g | 2080 g | −20% |
| IMU site | (0, 0, 0) | (+42, −18, −64) mm | off-centre |
| LiDAR site | (0, 0, +150) mm | (−83, −17, +34) mm | rearward, lower |

## Development history

Built on top of Amogh's `amogh_car.xml` base geometry (see `../neoracer_car/docs/amogh_review/`).
Key additions vs Amogh's file: explicit masses, widened steer range, coil-over suspension,
multi-joint-per-body pattern (required for Ackermann equality constraints to work in MuJoCo),
and the full sensor block.

## Next milestones (recommended order)

1. Apply URDF geometry corrections (wheelbase, track, masses) — geometry only, no controller
2. Expand LiDAR from 8 beams to 36 (10° spacing)
3. Add oval track (port `neoracer_sim_world.py`)
4. Pure-pursuit controller (`controller.py`)
5. Gymnasium wrapper (`neoracer_env.py`)
