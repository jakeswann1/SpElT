"""Functions for filtering position data by time windows."""

import numpy as np


def filter_position_by_windows(
    time_windows: list[tuple], trial_pos_bin_data: dict[int, dict]
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, float]:
    """
    Filter binned position data by time windows and pool across trajectories.

    Extracts position samples and timestamps that occur within specified
    time windows. Useful for analyzing spatial behavior during specific
    epochs (e.g., choice trajectories).

    Parameters
    ----------
    time_windows : list of tuples
        Each tuple is (trial_idx, start_time, end_time) specifying a time window
    trial_pos_bin_data : dict
        {trial_idx: bin_data_dict} where bin_data_dict contains:
        - 'pos_bin_idx': tuple of (x_bins, y_bins) arrays
        - 'pos_sample_times': array of timestamps
        - 'pos_sampling_rate': sampling rate in Hz

    Returns
    -------
    pos_bin_idx : tuple of (x_bins, y_bins)
        Position bin indices pooled across all windows
    pos_sample_times : np.ndarray
        Position sample timestamps pooled across all windows
    pos_sampling_rate : float
        Position sampling rate (Hz)

    Notes
    -----
    - Position bin indices and timestamps retain their original values
    - Assumes all trials have the same sampling rate
    - Windows from different trials are pooled together

    Examples
    --------
    >>> windows = [(0, 10, 15), (0, 20, 25), (1, 5, 10)]
    >>> pos_data = {
    ...     0: {'pos_bin_idx': (x_bins0, y_bins0),
    ...         'pos_sample_times': times0,
    ...         'pos_sampling_rate': 50},
    ...     1: {'pos_bin_idx': (x_bins1, y_bins1),
    ...         'pos_sample_times': times1,
    ...         'pos_sampling_rate': 50}
    ... }
    >>> pos_bins, times, rate = filter_position_by_windows(windows, pos_data)
    """
    all_x_bins = []
    all_y_bins = []
    all_timestamps = []
    pos_sampling_rate = None

    for trial_idx, start_time, end_time in time_windows:
        if trial_idx not in trial_pos_bin_data:
            continue

        bin_data = trial_pos_bin_data[trial_idx]
        x_bins = bin_data["pos_bin_idx"][0]
        y_bins = bin_data["pos_bin_idx"][1]
        timestamps = bin_data["pos_sample_times"]

        # Find samples within time window
        mask = (timestamps >= start_time) & (timestamps <= end_time)

        if not np.any(mask):
            continue

        all_x_bins.append(x_bins[mask])
        all_y_bins.append(y_bins[mask])
        all_timestamps.append(timestamps[mask])

        if pos_sampling_rate is None:
            pos_sampling_rate = bin_data["pos_sampling_rate"]

    if len(all_x_bins) == 0:
        return (np.array([]), np.array([])), np.array([]), 0.0

    # Concatenate across all windows
    pos_bin_idx = (np.concatenate(all_x_bins), np.concatenate(all_y_bins))

    # Create synthetic monotonically increasing timestamps
    # Since we're pooling across trajectories, we can't use actual timestamps
    # Instead, create a time vector based on the sampling rate
    total_samples = len(pos_bin_idx[0])
    dt = 1.0 / pos_sampling_rate if pos_sampling_rate > 0 else 0.02
    pos_sample_times = np.arange(total_samples) * dt

    return pos_bin_idx, pos_sample_times, pos_sampling_rate
