# proc_track — procedural racetrack RL environment

Single-agent time-trial env for the NeoRacer in MuJoCo. Every episode generates
a fresh closed-loop racetrack, deterministic from one integer seed. First step
toward GPU-parallel RL (repo goal), so the architecture is MJX-ready even though
this pass is CPU-only.

Per-developer scratch (`users/amoghmpanhale/`) — not part of the shared baseline.

## The architecture invariant (INV-1)

Every episode-dependent quantity lives in mutable fields of a fixed-topology
`MjModel` or in Python-side state. The model is compiled **once** — floor + car +
a fixed pool of 512 wall capsules with placeholder poses. `reset` rewrites
`geom_pos/quat/size` and domain-randomization fields in place; no XML generation,
no recompilation, ever. That is what makes the later MJX port a mechanical array
replace.

`assets/neoracer.xml` is never modified and no new XML is written to `assets/`.

## Run

```sh
# tests (plain python3):
python3 -m pytest users/amoghmpanhale/proc_track/tests/ -v

# viewer autopilot (macOS needs mjpython), streams tracks seed by seed:
mjpython users/amoghmpanhale/proc_track/view.py --seed 7
```

Deps: `scipy>=1.11` (see requirements.txt); numpy/mujoco/pytest come from the
repo root env.

## Layout

| File | What |
|---|---|
| `config.py` | ALL tunables (Gen/Wall/Proj/Env/Drive configs) + `*_for_phase` |
| `generator.py` | centerline, arc-length reparam, width, validate, repair → `Track`, `sample_valid_track` |
| `walls.py` | wall capsule params + `build_model` (compiled once at startup) |
| `projection.py` | stateful forward-biased local projection onto the centerline |
| `env.py` | `ProcTrackEnv` — reset, step, reward, termination, 23-dim Frenet obs |
| `drive.py` | scripted pure-pursuit + speed-PD reference controller |
| `view.py` | mjpython viewer demo |
| `tests/` | generator (V1–V4, T3, T6), walls (T4), projection, env (T2, determinism), drivability (T1, T5), wall-ride (T7) |

Everything is derived from one seed via a single `np.random.default_rng`, consumed
in a fixed order (reset: track seed → domain randomization → spawn). Same seed →
bit-identical Track, walls, and rollout (test T3).

## Curriculum

`gen_config_for_phase(0..3)` / `env_config_for_phase(0..3)` tighten max curvature
(0.4 → 1.0 m⁻¹) and shrink min half-width (0.60 → 0.35 m) across phases 0–3.

## Notes / gotchas

- **Walls need midphase disabled.** Static (worldbody) wall geoms repositioned at
  reset are invisible to MuJoCo's compile-time midphase BVH — they generate zero
  contacts once the model is big enough to trigger midphase. `build_model` sets
  `mjDSBL_MIDPHASE` to force the dynamic broadphase. Do not remove it.
- **Frictionless walls + contact penalty ship together.** Walls are frictionless
  (condim=1, priority=1) so the reward penalizes per-step wall contact (`c_wall`)
  to stop wall-riding (test T7). Removing one without the other breaks the design.
- Meshes in `assets/` are cosmetic; the trained car is a box with wheels.

## Out of scope this pass

MJX/jax backend, PPO training loop, curriculum auto-advance, normal-force wall
penalty, banking/heightfields, obstacles, promotion out of `users/`.
