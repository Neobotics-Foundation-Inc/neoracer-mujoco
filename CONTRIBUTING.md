# Contributing

This file describes how we write code and text in this repository.
Read it before you open a pull request.

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
  Do not change these directories unless the task names them.
- Keep one change in one pull request.

## Code style

Read AGENTS.md. It is the technical style guide, for people and for coding agents.
Review a pull request against its rules.

## Written English

Write comments, docstrings, commit messages and pull request descriptions in simple technical English
(close to [ASD-STE100](https://www.asd-ste100.org)):

- Write short sentences: 20 words or fewer for an instruction, 25 or fewer for a description.
- Put one topic in each sentence.
- Use the active voice and simple tenses: "The sweep deletes the key", not "The key is deleted".
- Write an instruction in the imperative, and put the condition first: "If the lock is held, stop."
- Use one word for one thing. When you choose a term, use it everywhere.
- Keep "the" and "a". Do not remove words to make a sentence short.
- Use plain words. Do not use idioms or figures of speech.
- Give each sentence a concrete subject, a plain verb and an explicit object. Say who does what to what.
- Only people and programs act. Data, files and documents do not "promise", "say", "plan" or "know".
- Do not write riddles or slogans. State the fact plainly.
- If you use a technical term, define it in one plain sentence where it first appears, or link to its definition.
  Do not put two technical terms in one short phrase.
- Write for a reader who does not know the code yet, not as notes to yourself.

Examples (compressed → plain):

- "The sensor parse keeps what it checked." → "The sensor parser returns a typed reading, so later code does not check the values again."
- "The rollout is a fold over step." → "The rollout calls `step` in a loop and passes each new state to the next call."
- "What the contract promises." → "What `contract.py` requires of every car model."
- A comment says what the code does. It does not give the history or the motivation; the commit message does that.
- A docstring describes the contract (inputs, outputs, units) in as few lines as it can.
