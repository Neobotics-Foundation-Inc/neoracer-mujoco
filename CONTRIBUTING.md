# Contributing

Thank you for helping with the NeoRacer simulator. This page explains how to make a change.
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
