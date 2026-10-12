# AGENTS.md

This file is the technical style guide for this repository.
It applies to people and to coding agents.
CONTRIBUTING.md has the workflow and the rules for written English.

## Before you push

Follow "Before you push" in CONTRIBUTING.md.
It gives the checks to run and the directories that you must not change.

The code requires Python 3.15. Use a feature from Python 3.13 to 3.15 where it removes a check,
a class of bugs or a copy of code (rule 9), not only because it is new.

The rules below come from typed functional programming.
Each rule removes a class of bugs or a class of checks.
When two rules disagree, the simpler code wins (rule 9).

## 1. Make invalid states unrepresentable

Choose types whose values are exactly the valid states.
Then no code has to check for an invalid state, because it cannot exist.

- Put parameters in a frozen dataclass with defaults: `params = WallFollowingParams()`.
  Do not use module constants or loose arguments for them.
- Use a frozen dataclass for a record, not a `dict` or a tuple of unnamed fields.
  If an outside API requires a dict with fixed keys (gymnasium's `info`), use a closed `TypedDict`:
  `class EnvInfo(TypedDict, closed=True)`. The type checker then checks each key and value.
- A constant mapping is a `frozendict`, not a `dict`.
- A fixed set of names is a `Literal`:

  ```python
  type Side = Literal["left", "right"]


  @dataclass(frozen=True)
  class WallFollowingParams:
      follow_side: Side = "right"
  ```

  Do not write `follow_side: str` and then check the value in `__post_init__`.
- Cases that carry different data are a union of frozen dataclasses:

  ```python
  @dataclass(frozen=True)
  class Hit:
      distance: float  # m


  @dataclass(frozen=True)
  class NoHit: ...


  @dataclass(frozen=True)
  class Corrupt: ...


  type Beam = Hit | NoHit | Corrupt
  ```

  Do not return `tuple[float, bool]` where the float is a placeholder when the bool is true.
  Do not use one record with fields that are valid only in some cases.
- Two `bool` fields that cannot both be true are one `Literal` or one union.
- Use `X | None` only when "no value" is a real case.
  Do not use `None` to mean "use the default". Write the default:
  `params: WallFollowingParams = WallFollowingParams()`. A frozen dataclass is a safe default.
- Do not give a field a default that is not a valid value.
  An empty array as the default for a 3-axis reading makes a record that is not a reading.
  If a test needs a partial value, the test builds it with a helper.
- Use a tuple for a value of fixed length: `tuple[float, float, float]` for a position.
- Do not use `Enum`.
  An enum adds member values, iteration and identity rules that this code does not use.
- Use `NewType` only when two values of the same type get mixed up in practice.

## 2. Parse, do not validate

A validator checks a value and then forgets what it found.
A parser checks a value and returns a type that keeps the result.

- Parse input once, at the edge, into a typed value.
  The edges are files, the MuJoCo model, the command line and raw sensor arrays.
- A parser returns the typed value or raises an error with a clear message.
  `def parse_side(text: str) -> Side`, not `def check_side(text: str) -> None`.
- Read names from a compiled model once, into a frozen record of indices or slices.
  Do not call `mj_name2id` or `getattr(obj, name, None)` inside the control loop.
- After the edge, the code trusts the type. Do not check the same property again.
- Some invariants have no Python type, for example "a positive float".
  Check such an invariant once, where the value is made (`__post_init__` of its params).
  This check is the parser for that value. Do not check it again in the code that uses it.

## 3. Total functions

A function returns a value for every input that its type accepts.

- If an input has no answer, make the input type smaller,
  or make the output a union that names the case (`Hit | NoHit`).
  Do not raise an error from deep inside the code.
- Do not `assert` what the types already show.
- Do not put `try`/`except` around code that cannot fail.
  Catch an exception only at the edge, and only the exception that you expect.

## 4. Match every case

```python
def distance(beam: Beam, max_range: float) -> float:
    match beam:
        case Hit(distance=d):
            return min(d, max_range)
        case NoHit() | Corrupt():
            return max_range
        case _:
            assert_never(beam)
```

- End each `match` over a `Literal` or a union with `case _: assert_never(x)`.
  A new case then fails type checking at each place that must handle it.
- Do not use chains of `isinstance`, or `dict.get(kind, default)`, over our own unions.
  A default hides a case that you forgot.

## 5. Defunctionalize

Show a choice of behavior as data, and interpret the data in one function.
Do not pass closures, callbacks or strategy objects for a closed set of behaviors.

```python
@dataclass(frozen=True)
class Progress:
    weight: float  # reward per rad of progress around the loop


@dataclass(frozen=True)
class ContactPenalty:
    weight: float  # reward per step in contact with a wall


type RewardTerm = Progress | ContactPenalty


def reward(term: RewardTerm, step: StepInfo) -> float:
    match term:
        case Progress(weight=w):
            return w * step.delta_theta
        case ContactPenalty(weight=w):
            return -w * step.in_contact
        case _:
            assert_never(term)
```

- Data can be compared, printed, saved in a config file, hashed and tested. A lambda cannot.
- A new behavior is a new case. The type checker then shows each place to update.
- Use a callable argument only when the set of behaviors is open,
  for example a policy that a user gives to a rollout function.
  Then give it a type: `Callable[[Observation], Action]` or a `Protocol`.

## 6. Pure core, effects at the edge

- Physics steps, the viewer, files, time and random numbers are effects.
  Keep them in thin outer functions: `examples/`, `sim.py`, the environment's `step`.
- Controllers, signal computation, reward and termination are pure functions of their inputs.
- State that lives from one step to the next is an explicit value:

  ```python
  def step(
      params: Params, state: State, readings: SensorReadings, dt: float
  ) -> tuple[State, ControlCommand]: ...
  ```

  Do not hide it in `self._prev_*` fields.
  A rollout is then a loop that calls `step` again and again and passes each new state to the next call.
  Gymnasium's `FuncEnv` uses the same shape.
- Pass a `np.random.Generator` as an argument. Do not use the global `np.random` functions.
- Do not change an argument.
  If a function must write to a MuJoCo `MjData`, its name says so (`apply_command(data, command)`), and it returns `None`.

## 7. Generic functions

Use the type parameter syntax (Python 3.12 and later):

```python
type Pair[T] = tuple[T, T]  # (left, right)


def mirror[T](pair: Pair[T]) -> Pair[T]:
    left, right = pair
    return right, left


@dataclass(frozen=True)
class Stamped[T]:
    value: T
    time: float  # s
```

- Make a function generic when its body does not depend on the element type.
  One generic function replaces a copy for each type.
- Use `type X = ...` for an alias. Do not use `TypeVar`, `Generic[T]` or `TypeAlias`.
- If the body needs an operation on `T`, bound `T` with a `Protocol` or a tuple of types.

## 8. Composition, not inheritance

- Use a `Protocol` for an interface, not an abstract base class.
- Subclass only a framework base that requires it, for example gymnasium's `Env`.
- If a class has one method and no state, make it a function.
- Build large behavior from small functions that each do one thing.

## 9. Simple, not simpler

- Wrap a value only if code inspects it or constrains it.
  A plain `float`, `tuple` or `np.ndarray` is correct when no code looks inside.
- Do not add an abstraction for one use. Extract code when a second use needs it.
- Before you write a utility, look in mujoco, numpy and gymnasium.
- Inline a function that is one or two lines and has one caller.
- Do not add a parameter that no caller sets.
- Delete dead code. Git keeps the history.

## 10. Structure

- Keep one source of truth for an interface.
  `contract.py` is the example: code, examples and tests all read it.
- Put imports at the top of the file.
  Import a module by its name (`from . import sensors`, not `as _sensors`).
- Load package files with `importlib.resources`, not with paths built from `__file__`.
- Put a test in `validation/test_<module>.py`, never in the module that it tests.

## 11. Numbers, units and frames

- Give each physical quantity its unit in a comment, and its source if it is estimated:
  `WHEEL_RADIUS = 0.050  # m, ESTIMATED from the STL bounding box`.
- State the frame and the sign convention once per module: `+X forward, +Y left, +Z up`.
- Use numpy for vector math. Give array types a name that says the shape:
  `type Vec3 = npt.NDArray[np.float64]  # shape (3,)`.
- Mark a known limit with `# Limit:`. Say when it matters and what to do then.

## 12. Tests state laws

- Prefer a test that states a law to a test with one example value.
  Examples of laws: a mirrored scene gives a mirrored command; a parse then a print gives the input; a second reset changes nothing.
- Test pure functions directly, without the simulator.
  Use the simulator for the physics conformance tests.

## 13. Names and comments

- Use the name of the real part: "motor encoder", not "wheel encoder", if the car has a motor encoder.
- A name says what a thing does. A function that does not use `async` is not called `async`.
- A comment says what the code does.
  It does not give history ("issue #7", "reimplemented from PR #29"). The commit message does that.
- Write comments and docstrings in the simple technical English of CONTRIBUTING.md.
