# Results

Every evaluation of this project, in order. Unless noted: `--headless`, the
deterministic policy, randomized cars ("rand") and clean cars ("clean"),
fixed seeds so models are compared on identical tracks and cars.

Racing counts **respawns** (wall, rollover or stuck deaths) over 100 races of
2 cars x 3 laps, about 600 car-laps. Procedural counts **laps completed** out
of 100 (or 50) tracks. The car, grip and randomization changed several times,
so compare numbers only within a section.

## 1. Loop track, original car

The original MJCF: four +-1 N.m torque motors, 1.86 kg, 50 mm wheels, grip 1.5.

| Run | Result |
|---|---|
| Plain PPO | ~100% laps, ~3.7 s per lap |
| Sim-to-real, harsh randomization (friction +-30%, motor +-20%, 0-2 step delay) | peaked at ~3.2M steps (5/5 laps randomized, 5.4-6.5 s), then degraded to ~55% laps by 10M |

The degradation came with approx_kl rising to 0.1, clip fraction to 0.46 and
action std collapsing to 0.08: fixed with `target_kl=0.02` and a linear
learning-rate decay. A second bug: `best/` saved the model and its observation
normalizer at different times, so a good checkpoint replayed badly; both are
now saved together.

## 2. Racing, original car (opposite starts, crash penalty 50)

| Run | rand respawns |
|---|---|
| race_opposite3 (warm start from the single-car sim-to-real policy) | 9 / 50 races; 23 / 100 races |
| race_opposite4 (continued, lr 1e-4) | 17 / 50 |
| race_opposite_mem (continued, 3-step memory) | 19 / 50; 45 / 100 |

Clean cars: 0 respawns for all, so every crash came from randomized builds.
race_opposite3/final matched /best (8 vs 9), so /best was not a lucky
checkpoint; the continuations really were worse.

## 3. Procedural tracks, original car (difficulty 1-2, 50 tracks)

| Run | rand | clean |
|---|---|---|
| proc_safe (crash penalties 20, time penalty 0.005, corner weight 0.2) | 24 | 40 |
| proc_mem (+ 3-step memory) | 31 | 45 |

Training curves plateaued: proc_safe at ~70% laps for its last 9M steps.

## 4. Realistic drive model on the older car

A DC-motor throttle (6 m/s at full throttle), steering capped at 18.75 deg,
LiDAR blind straight back, gentler randomization (about half the old ranges,
0-1 step delay). Grip still 1.5.

| Run | Result |
|---|---|
| race_real (racing) | 2 / 100 races, best lap 5.24 s (clean: 0, 5.06 s) |
| race_opposite3 on the same gentle randomization | 0 / 100 |
| proc_real (procedural, 50 tracks) | 35 rand, 46 clean |

race_opposite3 also had 0 under the gentle ranges: the gain came from the
gentler randomization, not the drive model.

## 5. Floor grip 0.8 (typical indoor rubber) instead of 1.5

| Run | Result |
|---|---|
| race_real, before retraining | 807 respawns / 100 races |
| race_grip (retrained) | 19 rand (clean 4), 5.60 s |
| race_grip2 (continued; still improving at the end of race_grip) | 2 rand (clean 0), 6.27 s |
| proc_real, before retraining | 18 / 50 |
| proc_grip3 / proc_grip6 (3- vs 6-step memory), best / final, 100 tracks | 77 / 70 and 73 / 80 |

Corner speed scales with the square root of grip: at 0.8 the car slid out above
~3 m/s where 1.5 held 4.3 m/s. 3- vs 6-step memory: no significant difference.

## 6. Upstream domain randomization (physics.py port)

Wheel friction, chassis mass, motor torque and steering stiffness. Procedural
tests from here on use a fresh, never-seen track every episode; before, 100
episodes reused about 13 tracks.

| Run | Result |
|---|---|
| race_grip2 (baseline under the new randomization) | 3 respawns, 6.20 s |
| race_phys (trained on it) | 5, 6.23 s (clean 0, 5.96 s) |
| race_fast (crash penalty 20 instead of 50) | 3, 6.32 s: not faster |
| proc_grip6 (baseline, 100 fresh tracks) | 87 |
| proc_phys (track bank 300, pool 50, wall clearance cost) | 80 final / 85 best (clean 73) |
| proc_grip6_phys (only the randomization changed) | 75 / 76 (clean 77) |
| s2r_phys (single car, 50 laps) | 50/50, 7.86 s (clean 7.31 s) |

## 7. GitHub car (upstream assets/neoracer.xml)

RS-545 DC motor and SPT5435 servo modeled natively, 3.0 kg, 42.5 mm wheels.
Full throttle settles at ~3.5 m/s.

| Run | rand | clean |
|---|---|---|
| race_gh (from race_phys) | **0 respawns**, 7.53 s | **0**, 7.00 s |
| proc_gh (from proc_grip6) | **96 / 100** | **99 / 100** |
| s2r_gh (from s2r_phys, 50 laps) | **50 / 50**, 9.46 s | |

Part of the jump is that the car is slower and so easier to keep on track,
but that is the real car's pace.

## Lessons

- **Randomization wider than the real car's variation costs more than it
  buys.** The harsh ranges caused most crashes of policies that drove
  perfectly on clean cars.
- **Lap time is set by the car, not the reward.** Lowering the crash penalty
  did not make the racing policy faster; grip and motor limits did.
- **Continuing from a converged model often made it worse** (four times).
  Copying the reward normalizer along with the policy on `--init-from` is the
  fix in place; it has not yet been confirmed by an experiment.
- **The MJCF chassis box tops out exactly at the LiDAR height**, so beams
  skimmed rival cars and missed them. Racing cars carry an invisible,
  massless LiDAR target box from 4 to 12 cm.
- **Evaluate on many episodes with fixed seeds.** 50 episodes leave about
  +-7% of noise; several apparent effects disappeared at 100.
- **More training past a plateau does nothing**: check the TensorBoard curve
  before starting another run.
