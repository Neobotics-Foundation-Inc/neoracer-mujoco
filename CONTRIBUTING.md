# Contributing

Thanks for working on the NeoRacer MuJoCo baseline. This file covers how to get set
up, where code belongs, and what your PR has to clear before it merges.

## Setup

Follow the [README setup section](README.md#setup) — conda or uv, then
`pip install -e .`. Two gotchas that bite everyone once:

- **Git LFS**: run `git lfs install` before cloning, or `assets/meshes/*.STL` arrive
  as 132-byte pointer files.
- **macOS viewer**: the passive viewer needs `mjpython -m examples.run`, not
  `python3`. `examples/manual_drive.py` is the exception — it owns a glfw window
  and must run under plain `python3`.

## Workflow

1. Branch off `main`. Name it `<github-username>/<Topic>` — e.g.
   `amoghmpanhale/Procedurally-Gen-Tracks`. That is the convention across every
   branch in the repo.
2. Commit as you go. Keep commits readable; the PR is squash-merged into `main`, so
   the PR title becomes the permanent history entry.
3. Open a PR against `main`. Describe what changed, and — for anything touching
   physics — how you verified it (which validation test, or what you saw in the
   viewer).
4. Direct pushes to `main` are not the workflow. Everything lands through a PR.

## Checks before you push

```sh
ruff check .          # lint (CI runs this with --output-format=github)
ruff format .         # format
python3 -m pytest validation/ -v
```

CI (`.github/workflows/ruff.yml`) runs `ruff check` and `ruff format --check` on
every PR. It does **not** run the validation suite — run it yourself. The suite
takes ~1 minute; `test_track_generation.py` dominates that.

`users/` is excluded from ruff (see `ruff.toml`). Everything else is not.

## Where code goes

The shared, team-facing baseline is `assets/`, `src/neoracer_mujoco/`, `examples/`,
and `validation/`. Nothing else.

| You are adding | It goes in | Notes |
|---|---|---|
| A car model | `assets/*.xml` | Auto-discovered by `cars()`; validation runs on it automatically. |
| A track | `assets/tracks/*.xml` or the generator | `compose(track, car)` attaches a car to either. |
| Reusable library code | `src/neoracer_mujoco/` | Controllers under `control/`. |
| A runnable demo | `examples/` | Hackable scripts, not a stable API. Copy `run.py` as the template. |
| A conformance check | `validation/` | Contract-level, not unit tests of helpers. |
| Personal experiments | `users/<your-username>/` | Your sandbox. Not part of the baseline. |

**`users/<username>/` is scratch space.** Nothing in the shared baseline may import
from it, and it is not reviewed as API. Prototype there freely; promote to
`src/neoracer_mujoco/` when it's ready to be depended on.

## Design principles

These are not style preferences — they are what the repo is for.

**The model is the product.** `assets/neoracer.xml` is the single source of truth.
It must stay drag-and-droppable into other MuJoCo tools, so it carries no
dependency on this Python package.

**Meshes are cosmetic.** STLs are `mass="0"` and exist for humans looking at the
viewer. All real mass and inertia lives in the XML. The car that gets trained is a
box with wheels — do not add vertices to the collision path for looks.

**Write code an average researcher can read.** Compressing five clear lines into one
clever line is frowned upon here. Boring beats clever. Optimize for someone who
opened this repo an hour ago and wants to understand the physics, not the Python.

**Don't build it until it's needed.** No abstractions with one implementation, no
config for a value that never changes.

## The car contract

`src/neoracer_mujoco/contract.py` is the single source of truth for actuator names,
sensor requirements, ctrl layout, and the mass band:

- `ctrl[0:4]` — fl/fr/rl/rr wheel drive torque, N·m, `+` = forward
- `ctrl[4]` — steer servo target angle, rad, ±0.4, `+` = left

Two Ackermann polynomial equality constraints derive the inner/outer front-wheel
angles from that single steer command.

If your change alters the contract, change it in `contract.py` — not in the tests,
not in the examples. Validation reads from there, and so does everything else.

## Adding a new car

Drop the XML into `assets/`. `validation/conftest.py` globs `assets/*.xml` and
parametrizes every conformance test over each car, so the full battery runs against
your model with zero test edits. If it fails, the model is wrong or the contract
changed — fix one of those, don't special-case the test.

## Adding a validation test

`validation/` is a conformance gate, not a unit-test suite. Add a test when there is
a **failure mode** it catches — the car sinks through the floor, an RL agent could
extract free energy, sensors go non-finite under load, the sim is non-deterministic.
One test per failure mode, named after the failure.

Pure-logic checks (unit conversions, geometry math) go in `test_logic.py` and should
not need a physics step.

## Physics changes

MuJoCo is not perfect, and wheel/plane contact plus the soft-body relations are the
fragile parts. If you touch contact parameters, solver settings, suspension, or
friction:

- Say so explicitly in the PR description.
- Run the full validation suite and report the result.
- Watch it in the viewer (`mjpython -m examples.run`) before claiming it works.

A change that makes one controller behave better but quietly degrades contact
stability is a regression, not an improvement.

## Questions

Open an issue or ask in the PR. Better a question than a large diff pointed the
wrong way.
