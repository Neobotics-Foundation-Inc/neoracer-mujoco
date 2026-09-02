"""
Shared reporting helpers for the vehicle-space wall follower trial harnesses.
Experimental; not part of `pytest validation/`. Reporting/formatting only --
does not compute or alter any trial metric.
"""

from __future__ import annotations

from typing import Protocol


class _HasOutcome(Protocol):
    outcome: str


def outcome_header(results: list[_HasOutcome]) -> str:
    """success/stuck/rollover counts line shared by closed_loop_trials.py's
    and loop_trials.py's summarize()."""
    n_success = sum(r.outcome == "success" for r in results)
    n_stuck = sum(r.outcome == "stuck" for r in results)
    n_rollover = sum(r.outcome == "rollover" for r in results)
    return f"success={n_success}/{len(results)}  stuck={n_stuck}  rollover={n_rollover}"
