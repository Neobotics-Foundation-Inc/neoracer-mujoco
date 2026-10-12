# Contributing

This file describes how we write code and text in this repository.
Read it before you open a pull request.

## Before you push

- Run `ruff check .` and `ruff format .`. CI runs both.
- Run `pytest validation`. Put a new test in `validation/test_<module>.py`, never in the module it tests.
- Keep one change in one pull request.

## Model the data so that bad values cannot exist

- Put parameters in a frozen dataclass with defaults: `params = WallFollowingParams()`, not module constants or loose arguments.
- Use a frozen dataclass for a record, not a `dict` or a tuple.
- Use an `Enum` or a `Literal` for a fixed set of choices, not a string or a `bool` flag.
  Two flags that cannot both be true are one `Enum`.
- Parse input once, at the edge (a file, a sensor, the command line), into a typed value.
  After that, the code trusts the type and does not check it again.
- Match every case of an `Enum`. End the `match` with `case _: assert_never(x)`, so a new case fails type checking.
- Make a function return a value for every input it accepts.
  If an input is not valid, change the input type. Do not add a check deep inside the function.
- Do not wrap a value in a class if no code looks inside it. A plain `float` or `list` is correct then.

## Structure

- Keep one source of truth for an interface. `contract.py` is the example: code, examples and tests all read it.
- Extract code that you use twice into a module (for example `math.py`).
  Before you write a utility, check if mujoco, numpy or gymnasium already has one.
- Inline a function that is one or two lines and has one caller.
- Put imports at the top of the file. Import a module by its name (`from . import sensors`, not `as _sensors`).
- Load package files with `importlib.resources`, not with paths built from `__file__`.
- Prefer a library's functional API when it has one (for example gymnasium's `FuncEnv`): state goes in, state comes out.
- Give every physical constant its unit, and its source if it is estimated:
  `WHEEL_RADIUS = 0.050  # m, ESTIMATED from the STL bounding box`.
- Mark a known limit with `# ponytail:` and say what to do when the limit matters.
- Keep a helper small until a second use needs more.

## Names

- Use the name of the real part: "motor encoder", not "wheel encoder", if the car has a motor encoder.
- A name says what a thing does. A function that does not use `async` is not called `async`; call it `non_blocking`.

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
