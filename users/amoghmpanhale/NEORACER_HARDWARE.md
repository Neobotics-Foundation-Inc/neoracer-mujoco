# NeoRacer Vehicle — Complete Hardware Reference

Exhaustive inventory extracted from the CAD assets in this folder so the design never
needs to be re-parsed. Source of truth: `neoracer-full-vehicle.FCStd` (FreeCAD 1.0,
1006 objects), cross-checked against `.step` (AUTOMOTIVE_DESIGN AP214) and the `.stl`
meshes.

## What this vehicle is

A **small-scale (≈1/8–1/10 RC-sized) autonomous race car** — not a full-size vehicle.
It carries a real autonomy stack (Jetson compute + LIDAR + stereo cameras) on an RC-car
mechanical platform with full independent suspension, steering, and an all-wheel-drive
geared drivetrain. The mechanical parts were designed by a **Hungarian-speaking team**
(part labels are Hungarian; accented characters got mangled into `_X_<hex>` escapes on
export — decode table at the bottom).

## Overall dimensions & units

- **Native units: millimeters, kilograms** (FreeCAD `UnitSystem = Standard (mm, kg, s, °)`;
  STEP confirms `SI_UNIT(.MILLI.,.METRE.)` and `SI_UNIT(.KILO.,.GRAM.)`).
- **STL bounding box** (reliable, whole-vehicle): X 268.5 mm × Y 191.2 mm × Z 446.3 mm.
  - Long axis (Z, ≈446 mm) is vehicle length; X (≈268 mm) is width; Y (≈191 mm) is height.
  - The mesh origin is **not** centered: Z spans −367.5 to +78.8, so re-center before
    use in MuJoCo (the body origin should sit at the CoM/wheelbase center).
- Mesh density: ~2.08 M triangles — visual only, never use raw for collision.

> ⚠️ Component placements in the FreeCAD file are **local assembly coordinates** (relative
> to each sub-assembly's parent), not world coordinates. Treat them as relative hints, not
> absolute positions. Only the STL bounding box gives a trustworthy global scale.

## Drivetrain (geared AWD)

| Part (label) | Hungarian / meaning | Notes |
|---|---|---|
| `M300_ASM` | drive motor | Single motor labeled "M300" |
| `ESC_GOOLRC_60A` | ESC | **GoolRC 60 A** brushless electronic speed controller |
| `DIFI_*` + `DIFI_TENGELY_FRONT_REAR_CENTRAL` | differenciálmű (differential) | **Three differentials: front, rear, AND central** → full AWD with center diff |
| `BOLYG_X_F3KER_X_E9K` | **bolygókerék** = planet gear | 8 total (4 + 4 mirrored) — planetary/bevel diff internals |
| `NAPKER_X_E9K` | **napkerék** = sun gear | 4 total — diff internals |
| `K_X_FAPKER_X_E9K_Z14_HAJT` | **kúpkerék** = bevel gear, Z14 (14 teeth), hajtó (drive) | bevel ring/pinion, 14T |
| `HAJT_X_F3TENGELY`, `HAJT_X_E1S_HATSZ` | **hajtótengely** (driveshaft), **hajtás hátsó** (rear drive) | driveshafts; HÁTSÓ = rear |
| `F_X_E9LTENGELY_00` | **féltengely** = half-shaft / CV axle | 4 (front pair + rear pair, mirrored) |
| `DIFI_H_X_E1Z` | **diffi ház** = differential housing | gearbox case |

## Suspension (independent, coil-over + anti-roll)

| Part (label) | Hungarian / meaning | Notes |
|---|---|---|
| `LENG_X_E9SCSILLAP...` | **lengéscsillapító** = shock absorber / damper | Largest part family in the model — coil-over dampers at all corners |
| `RUG_X_F3TORONY_CSAP` | **rugó** (spring) + **torony** (tower) + **csap** (pin) | Coil springs on shock towers; 4 (2 + 2 mirrored) |
| `LENG_X2_0151_X0_KAR`, `LENGG_..._KAR` | **kar** = control arm / wishbone | Suspension control arms (upper/lower wishbones) |
| `TORZI_X_F3S_R...` | **torziós (rugó)** = torsion bar / **anti-roll bar** | front & rear anti-roll/sway bars |
| `TORZI_..._BILINCS` | **bilincs** = clamp | sway-bar mounting clamps |
| `CSAP_X_E1GY__Z14...` | **csapágy** = ball bearing | Many bearings throughout (hubs, diffs, steering) |

## Steering (servo-driven Ackermann)

| Part (label) | Hungarian / meaning | Notes |
|---|---|---|
| `SERVO_SPT5435LV_ASM` | steering servo | **SPT5435LV** servo (35 kg-class, LV = low-voltage) |
| `SERVO_ARM` | servo horn/arm | output arm off the servo |
| `KORM_X_E1NY...` | **kormány** = steering (linkage family) | full steering linkage / bellcrank set |
| `STEERING_KNUCKLE_JOINT` | steering knuckle | 2 (L/R) — front upright pivot |
| `STEERING_OUTER_LINK` | outer tie-rod link | 2 (L/R) |
| `STEERING_LINK_SHAFT` | steering link shaft / drag link | center link |
| `IR_X_E1NY_X_EDT_X_F3KAR` | **irányító kar** = steering arm | tie-rod / steering arm to knuckle |

→ This is a steerable front axle: **servo → arm → drag link → bellcrank → tie rods →
knuckles**, i.e. classic Ackermann front steering. Maps cleanly to MuJoCo's
"steer body + hinge (axis Z) + position actuator" pattern already used in the experiments.

## Wheels & tires

| Part (label) | Hungarian / meaning | Notes |
|---|---|---|
| `KER_X_E9KANYA_1` | **kerékanya** = wheel nut | 4 (1 + 1 + 2 mirrored) → confirms **4 wheels** |
| `39500` | tire | likely a tire part/model number (sits at the wheel hubs) |
| `A014-2024-05-11`, `A016-L-2024-05-11`, `A017-R-2024-05-11` | dated wheel/rim parts | left/right rim iterations dated May 2024 |
| `BOLYG_..._TENGELY` | planet-gear shaft | (also part of diffs, listed above) |

## Compute, sensors & autonomy stack

| Part (label) | What it is | Notes |
|---|---|---|
| `Jetson Orin Nano` | **main compute** | NVIDIA Jetson Orin Nano — runs the autonomy / inference |
| `RC-LIDAR-00` | **LIDAR** | spinning RC LIDAR, mounted high (top of stack) |
| `Camera:1`, `Camera:2` | **two cameras** | stereo / dual-camera vision |
| `STMINIV2`, `SSTMINI`, `MINISST`, `MINIV2-205` | **custom STM32 board** ("ST Mini V2") | low-level microcontroller (motor/servo/sensor I/O) |
| `LED00`, `Dot Matrix` | status LED + dot-matrix display | UI / status indicators |

## Power system

| Part (label) | What it is | Notes |
|---|---|---|
| `LIPO Battery:1`, `:2` | **two LiPo battery packs** | main power |
| `Lipo-PCB Connection` | battery → board harness | |
| `CONN-TH_XT60PW` | **XT60** power connector | main battery connector |
| `CONN-TH_XT30UPB`, `XT30PB` | **XT30 / XT30U** connectors | secondary power distribution (many instances) |
| `3D_OSRC_PW_V1_2025-10-29` | power board ("PW") | dated Oct 2025 |

## Custom PCBs (designed in EasyEDA)

`EASYEDA_PCB_MODEL_*`, `BOARD_YPGN`, `BOARD_VKX0`, `BOARD_O9SG`, `PCB`, `PCB Pins` —
**three custom boards** (YPGN, VKX0, O9SG). Populated with standard SMD parts:

- **Resistors:** 0603, 0805, 1206 (imperial footprints) — `R0603`, `R0805`, `R1206`, `R2`, `R3`
- **Capacitors:** 0805, 1206, 1812, plus large `CAP-SMD_BD8.0` (8 mm electrolytic) — `C0805`/`C1206`/`C1812`/`C1`/`C6`/`C9`/`C13`
- **Transistors / MOSFETs:** `Q1` = PG-HSOF-8 (Infineon power FET package), `Q2` = SOT-23-3
- **Diodes:** SOD-323, SOT-23-3 (`U4`, `U5`, multiple)
- **Signal connectors:** JST-style 1.25 mm pitch — `CONN-SMD_8P-P1.25` (ZX-MX1.25), `CONN-SMD_4P-P1.25` (XUNPU wafer)
- **Headers:** 2.54 mm through-hole 5-pin — `HDR-TH_5P-P2.54`, `H4`

## Body / chassis / aero

| Part (label) | What it is |
|---|---|
| `Lower-Chassis` | main lower chassis plate / tub |
| `Front Bumper` | front bumper |
| `Rear Spoiler` | rear spoiler (aero) |
| `WING-MOUNT-L`, `WING-MOUNT-R` | rear wing mounts (aero) |
| `Payload (3D-Printed)` | 3D-printed payload module |
| `NAPKER`, `NEORACERV1-Jan18th v3` | top-level vehicle assembly name |

## Fasteners (standard hardware — bulk, low MuJoCo relevance)

Large counts of standard screws/nuts; ignore for simulation but noted for completeness:
`ISO_4035` (M3 thin hex nuts), `CSAVAR`/`CNS_4558` (M3×0.5 screws — *csavar* = screw),
`ANSI_B18_3_4M` (M3 socket-head cap screws), `ASME_ANSI_B18_3_5M` (M3×6),
`BS_4168` (M3/M5 hex socket set screws), `AS_1421` (M4×8 set screws),
`PAN_HEAD_PHILLIP_MC_SCREW`, `ANSI_B27_7` (retaining rings).

## MuJoCo implications (how to use this in sim)

1. **Scale**: import the STL with `scale="0.001 0.001 0.001"` (mm → m). Whole car is
   ~0.45 m long × 0.27 m wide — about 1/4 to 1/3 the size of the current placeholder box
   (`size=[1, 0.5, 0.2]` = 2 m × 1 m car). Consider rescaling the experiment car to match.
2. **Drivetrain → actuators**: it's **AWD with 3 diffs**, so a realistic model drives all
   4 wheels (or applies a single motor torque split via the center diff). The current
   experiments drive via `qvel`/rear motors only — to match hardware, add motor actuators
   to all four wheel hinge joints, or model one motor + a torque-split.
3. **Steering → front Ackermann**: servo + tie-rods + knuckles ⇒ steer the **front two
   wheels** via steer bodies (hinge `axis=[0,0,1]`) + position actuators (the `aeh961`
   `10_mjspec_car_with_joints.py` pattern is the right template; servo is ~35 kg-class).
4. **Suspension**: coil-overs + control arms + anti-roll bars at all 4 corners. A faithful
   model uses prismatic/hinge suspension joints with `<spring>`/damping per corner, rather
   than rigidly mounting wheels to the body (current experiments mount wheels rigidly).
5. **Mass / CoM**: heavy items (two LiPo packs, Jetson, motor, ESC) dominate inertia —
   if accuracy matters, place mass elements at those component locations rather than
   assuming a uniform box. The LIPOs and Jetson sit low/mid; LIDAR sits high (raises CoM).
6. **Sensors for RL/perception**: LIDAR + 2 cameras + IMU-class data are available on the
   real platform — MuJoCo can emulate these (rangefinder sensors, camera renders) if the
   project moves toward sim-to-real perception.

## Hungarian → English decode table

Export mangled accented letters into `_X_<hex>` (á=E1, é=E9, í=ED, ó=F3, ö=F6, ő=F5,
ú=FA, ü=FC, ű=FB). Common roots:

| Mangled / root | Hungarian | English |
|---|---|---|
| `KORM_X_E1NY` | kormány | steering |
| `KER_X_E9K` | kerék | wheel/gear |
| `KER_X_E9KANYA` | kerékanya | wheel nut |
| `LENG_X_E9SCSILLAP_X_EDT_X_F3` | lengéscsillapító | shock absorber / damper |
| `RUG_X_F3` | rugó | spring |
| `TORONY` | torony | tower (shock tower) |
| `HAJT_X_F3TENGELY` | hajtótengely | driveshaft |
| `HAJT_X_E1S` / `HATSZ_X_F3` | hajtás / hátsó | drive / rear |
| `F_X_E9LTENGELY` | féltengely | half-shaft / CV axle |
| `TENGELY` | tengely | shaft / axle |
| `DIFI` | differenciálmű | differential |
| `BOLYG_X_F3KER_X_E9K` | bolygókerék | planet gear |
| `NAPKER` | napkerék | sun gear |
| `K_X_FAPKER_X_E9K` | kúpkerék | bevel gear |
| `CSAP_X_E1GY` | csapágy | (ball) bearing |
| `CSAP` | csap | pin / journal / pivot |
| `CSAVAR` | csavar | screw / bolt |
| `KAR` | kar | arm (control arm) |
| `IR_X_E1NY_X_EDT_X_F3` | irányító | steering / guiding |
| `TORZI_X_F3S` | torziós | torsion (anti-roll bar) |
| `BILINCS` | bilincs | clamp / clip |
| `H_X_E1Z` | ház | housing / case |
| `FEJ` | fej | head |
| `ANYA` | anya | nut |
| `MIR` | (suffix) | mirrored part (L↔R symmetry) |
| `ASM` | (suffix) | assembly |
