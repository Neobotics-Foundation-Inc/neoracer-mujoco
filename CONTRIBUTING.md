# Contributing

This file describes how we write code and text in this repository.
Read it before you open a pull request.

## Before you push

- Run `ruff check .` and `ruff format .`. CI runs both.
- Run `pytest validation`. Put a new test in `validation/test_<module>.py`, never in the module it tests.
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
- A comment says what the code does. It does not give the history or the motivation; the commit message does that.
- A docstring describes the contract (inputs, outputs, units) in as few lines as it can.
