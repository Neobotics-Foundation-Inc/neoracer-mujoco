"""case() names the case that a loop in a test was on when an assert failed."""

from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def case(name: object) -> Generator[None]:
    """Add the case's name to an AssertionError raised in the block, so a test
    that loops over cases says which one failed:

        for track in TRACKS:
            with case(track):
                assert_car_rests(...)
    """
    try:
        yield
    except AssertionError as error:
        error.add_note(f"case: {name}")
        raise
