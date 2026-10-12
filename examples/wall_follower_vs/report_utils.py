"""
The trial outcome type and the summary line that the trial scripts print.
"""

from collections.abc import Sequence
from typing import Literal

# "stuck": the car made too little progress in the stuck window.
# "timeout": the trial reached its maximum time without success.
type Outcome = Literal["success", "stuck", "rollover", "timeout"]


def outcome_header(outcomes: Sequence[Outcome]) -> str:
    """One line with the count of each outcome."""
    n = outcomes.count
    return (
        f"success={n('success')}/{len(outcomes)}  stuck={n('stuck')}  "
        f"rollover={n('rollover')}  timeout={n('timeout')}"
    )
