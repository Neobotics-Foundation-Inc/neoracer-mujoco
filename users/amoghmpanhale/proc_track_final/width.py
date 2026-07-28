"""Stage 3: how wide the corridor is at each sample around the loop.

Width is a few slow sine waves summed onto a base width, so the track breathes
between narrow and wide without ever stepping. It is drawn independently of the
centerline shape - a sample's width does not depend on where that sample is.
"""

import numpy as np

from .settings import TrackSettings


def draw_width_waves(random):
    """Draw the random ingredients of the width profile: a handful of sine waves,
    each with a random phase and a random amplitude that shrinks as the wave gets
    faster, so the width drifts slowly and never flickers.

    Returns (phases, amplitudes). Drawn once per attempt and reused across
    repairs, so a repaired track keeps the width it was validated against.
    """
    wave_count = int(random.integers(2, 6))
    phases = random.uniform(0, 2 * np.pi, wave_count)
    amplitudes = random.uniform(0.03, 0.12, wave_count) / np.arange(1, wave_count + 1)
    return phases, amplitudes


def half_width_from_waves(waves, settings: TrackSettings):
    """Sum the width waves onto the base half-width and clamp to the allowed
    band, giving one half-width per sample.

    Wave `w` completes exactly w+1 cycles over the loop, so the profile joins
    itself at the seam with no discontinuity.
    """
    phases, amplitudes = waves
    angle_around_loop = np.linspace(0, 2 * np.pi, settings.samples, endpoint=False)
    half_width = settings.half_width_base + sum(
        amplitude * np.sin((wave + 1) * angle_around_loop + phase)
        for wave, (amplitude, phase) in enumerate(zip(amplitudes, phases))
    )
    return np.clip(half_width, settings.half_width_min, settings.half_width_max)
