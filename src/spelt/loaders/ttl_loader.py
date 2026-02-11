"""TTL/sync data loading functions."""

from pathlib import Path

import numpy as np


def load_ttl_data(
    trial_path: Path, recording_start_time: float, recording_type: str
) -> dict:
    """
    Load TTL/sync data from OpenEphys recording.

    Parameters
    ----------
    trial_path : Path
        Path to trial directory containing OpenEphys data
    recording_start_time : float
        Recording start time for the segment (from SpikeInterface)

    Returns
    -------
    dict
        TTL data with key:
        - 'ttl_timestamps': Array of TTL pulse times (relative to recording start)

    Raises
    ------
    Exception
        If TTL loading fails
    """
    from spelt.np2_utils.load_ephys import load_np2_ttl

    try:
        # Load TTL events from OpenEphys
        ttl_timestamps = load_np2_ttl(trial_path, recording_type)

        # Rescale timestamps relative to recording start
        if ttl_timestamps[0] - recording_start_time < 0:
            print(
                f"Warning: Recording start time {recording_start_time} is later than "
                f"the first TTL pulse {ttl_timestamps[0]}. "
                f"Setting first TTL pulse to 0."
            )
            ttl_timestamps = ttl_timestamps - ttl_timestamps[0]
        else:
            ttl_timestamps -= recording_start_time

        return {"ttl_timestamps": ttl_timestamps}

    except Exception as e:
        print(f"Error loading TTL data: {str(e)}")
        return {"ttl_timestamps": None}


def get_ttl_frequency(ttl_timestamps: np.ndarray, skip_first: int = 2) -> float:
    """
    Calculate average TTL pulse frequency.

    Parameters
    ----------
    ttl_timestamps : np.ndarray
        Array of TTL pulse times
    skip_first : int
        Number of initial pulses to skip (default: 2)

    Returns
    -------
    float
        Average frequency in Hz
    """
    if ttl_timestamps is None or len(ttl_timestamps) < skip_first + 2:
        return None

    intervals = np.diff(ttl_timestamps[skip_first:])
    return 1.0 / np.mean(intervals)
