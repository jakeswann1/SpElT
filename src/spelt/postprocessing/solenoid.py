"""
Solenoid valve event extraction for Bonsai-recorded behavioural trials.

Extracts solenoid activation times from a Bonsai CSV and maps them into the
LFP time domain using the TTL sync signal (one pulse per camera frame recorded
in OpenEphys).  The returned timestamps are in the same reference frame as
``ephys.lfp_data[trial_idx]["timestamps_relative"]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Bonsai CSV columns carrying the solenoid boolean signals
SOLENOID_COLUMNS = ["Value.Item6.Item1", "Value.Item6.Item2"]

# Bonsai CSV column carrying the per-row frame counter (0-indexed)
FRAME_COLUMN = "Value.Item1"


def get_solenoid_times(ephys_obj, trial_idx: int, csv_path: str | Path) -> np.ndarray:
    """
    Extract solenoid valve activation times in LFP-relative seconds.

    Reads rising edges in the solenoid boolean columns
    (``Value.Item6.Item1``, ``Value.Item6.Item2``) from a Bonsai CSV file and
    maps each activation frame to the corresponding LFP-domain timestamp via
    the TTL sync signal stored in ``ephys_obj.sync_data[trial_idx]``.

    TTL data is loaded automatically if not already present on the ephys object.

    Parameters
    ----------
    ephys_obj : spelt.ephys.ephys
        Ephys object for the session.  Must have the sorting analyser loaded
        (via ``load_ephys``) so that TTL loading can determine the recording
        start time.
    trial_idx : int
        Index of the trial to process.
    csv_path : str or Path
        Path to the Bonsai behavioural CSV file for this trial.

    Returns
    -------
    np.ndarray
        Sorted array of solenoid activation times in seconds, referenced to
        the LFP trial start (i.e. directly comparable to
        ``ephys_obj.lfp_data[trial_idx]["timestamps_relative"]``).

    Notes
    -----
    The TTL array and the CSV frame counter are assumed to be 1-to-1.  When
    the TTL array is longer than the CSV (extra pulses at the end), the extra
    entries are silently ignored.  Frame indices beyond the TTL array length
    are clamped to the last TTL entry.
    """
    # Ensure TTL sync data is available
    if (
        ephys_obj.sync_data[trial_idx] is None
        or ephys_obj.sync_data[trial_idx].get("ttl_timestamps") is None
    ):
        ephys_obj.load_ttl([trial_idx], output_flag=False)

    ttl_timestamps = ephys_obj.sync_data[trial_idx]["ttl_timestamps"]

    df = pd.read_csv(csv_path)

    frame_indices = df[FRAME_COLUMN].values.astype(int)
    frame_indices_clamped = np.clip(frame_indices, 0, len(ttl_timestamps) - 1)
    frame_times = ttl_timestamps[frame_indices_clamped]

    present_cols = [c for c in SOLENOID_COLUMNS if c in df.columns]
    if not present_cols:
        raise ValueError(
            f"No solenoid columns found in {csv_path}. " f"Expected: {SOLENOID_COLUMNS}"
        )

    activation_times: list[float] = []
    for col in present_cols:
        vals = df[col].values.astype(bool)
        rising = np.where(np.diff(vals.astype(np.int8)) == 1)[0] + 1
        activation_times.extend(frame_times[rising].tolist())

    return np.sort(np.asarray(activation_times, dtype=float))
