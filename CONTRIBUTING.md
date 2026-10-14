# Contributing

Thank you for helping with the NeoRacer simulator. This page explains how to make a change.
Coding agents read AGENTS.md instead.

## Before you push

- Run `ruff check .` and `ruff format .`. CI runs `ruff check .` and `ruff format --check .`.
- Run `ty check`. CI runs it too.
  ty needs the mujoco stubs in `typings/`.
  Make them with `pybind11-stubgen mujoco mujoco.viewer mujoco.rollout -o typings` after you install or update mujoco.
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
- `CITATION.cff` lists the five people with the most commits. If your commits change who they are, run `python3 .github/citation.py > CITATION.cff`.
  If you have committed under more than one name or email, add a line to `.mailmap` first, so your commits count as one person's.

## You are responsible for your change

You are responsible for every change you contribute, including code that an AI tool such as Claude or Copilot wrote for you.
Read and understand all of it before you open a pull request, and be ready to explain any line in review.

You may credit a tool in your own commits, but the commit that lands on main credits no AI tool.
A pull request is squash-merged with its title and description as the commit message.
The AI credit workflow removes the lines that credit an AI tool, such as `Co-authored-by: Claude` or "Generated with Claude Code", from the description.
