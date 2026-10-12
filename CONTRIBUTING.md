# Contributing

Thank you for helping with the NeoRacer simulator. This page explains how to make a change and what reviewers look for.
Coding agents read AGENTS.md instead.

## Before you push

- Run `ruff check .` and `ruff format .`. CI runs `ruff check .` and `ruff format --check .`.
- Run `ty check`. CI runs it too.
  ty needs the mujoco stubs in `typings/`.
  Make them with `pybind11-stubgen mujoco mujoco.viewer -o typings` after you install or update mujoco.
- Run `python -m pytest validation`. Put a new test in `validation/test_<module>.py`, never in the module it tests.
- Keep the tests green after each commit.
  If a change alters behavior, change its test in the same commit.
- `users/` holds personal experiments.
  `validation/testdata/` and `examples/ultimate_wall_follower/` hold copies of outside code.
  Do not change these directories unless your change is about them.
- Keep one change in one pull request.

## What reviewers look for

The code follows a few ideas from typed functional programming. Each one removes a kind of bug, or a kind of check that someone must remember to write.

- **Bad values cannot be built.** Parameters and records are frozen dataclasses.
  A fixed set of names is a `Literal`, such as `Side = Literal["left", "right"]`.
  Cases that carry different data are a union of frozen dataclasses. We do not use `Enum`.
- **Input is parsed once, at the edge.** Sensor arrays, files and the command line become typed values where they enter the code.
  After that, the code trusts the types and does not check the values again.
- **Every function answers every input it accepts.** If some input has no answer, the input type or the output type changes, so the case is explicit.
- **Every case is handled.** A `match` over a `Literal` or a union covers each case and ends with `assert_never`, so the type checker shows each place to update when a case is added.
- **Logic is pure, and state is explicit.** Controllers, rewards and termination are functions of their inputs.
  State that lasts from one step to the next is a value that goes in and comes out, not a hidden field.
- **Simple, not simpler.** A plain `float` or array is fine when nothing inspects it. A helper is extracted when a second use needs it, not before.
- **Physical numbers carry their units**, and estimated numbers say where they come from.
  A known limit is marked with `# Limit:`, with when it matters and what to do then.
- **Comments say what the code does.** The commit message gives the history and the reasons.

If you are not sure how to apply one of these, open the pull request and ask. A reviewer will help.
