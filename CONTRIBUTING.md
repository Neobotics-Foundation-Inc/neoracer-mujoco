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
- `users/` is the place to experiment. Make your own folder, `users/<your-github-name>/`, and put anything there: scripts, notes, XML, your own environment files.
  ruff, ty, the tests and CI do not check `users/`, and nobody else changes your folder.
  Do not change the folder of another person.
  Code outside `users/` never imports from it. When an experiment is ready to share, copy it to `src/` or `examples/` in its own pull request and make it follow the rules there.
- `validation/testdata/` and `examples/ultimate_wall_follower/` hold copies of outside code.
  Do not change these directories unless your change is about them.
- Keep one change in one pull request.

## You are responsible for your change

You are responsible for every change you contribute, including code that an AI tool such as Claude or Copilot wrote for you.
Read and understand all of it before you open a pull request, and be ready to explain any line in review.
The commit is yours, so do not list an AI tool as a co-author: leave out `Co-authored-by:` lines and "Generated with" footers that name one.
The `.claude/settings.json` in this repo stops Claude Code from adding them.
