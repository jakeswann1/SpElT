"""Functions for collapsing 2D position data to 1D along X axis."""

import numpy as np


def collapse_position_bins_to_x(
    pos_bin_idx: np.ndarray | tuple[np.ndarray, np.ndarray],
    pos_sample_times: np.ndarray,
    pos_sampling_rate: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Collapse 2D position bin indices to 1D X-only indices.

    Takes position data with (y, x) bin indices and collapses to just x indices,
    effectively ignoring vertical position and focusing on horizontal position
    (the behaviorally relevant dimension in T-maze).

    Parameters
    ----------
    pos_bin_idx : np.ndarray or tuple of (x_bins, y_bins)
        Position bin indices in one of two formats:
        - 2D array with shape (n_samples, 2) with columns [y_idx, x_idx]
        - Tuple of (x_bins, y_bins) arrays
    pos_sample_times : np.ndarray
        Timestamps for each position sample
    pos_sampling_rate : float
        Position sampling rate in Hz

    Returns
    -------
    pos_x_idx : np.ndarray
        1D position bin indices (just X), shape (n_samples,)
    pos_sample_times : np.ndarray
        Unchanged timestamps (returned for consistency)
    pos_sampling_rate : float
        Unchanged sampling rate (returned for consistency)

    Notes
    -----
    This collapses the Y dimension, treating all Y positions at the same X
    as equivalent. This is appropriate for T-maze splitter cell analysis
    where horizontal position is the key variable and vertical wobbling
    within corridors should be averaged out.
    """
    # Handle tuple format (x_bins, y_bins)
    if isinstance(pos_bin_idx, tuple):
        x_bins, y_bins = pos_bin_idx
        return x_bins, pos_sample_times, pos_sampling_rate

    # Handle numpy array format
    if pos_bin_idx.ndim == 1:
        # Already 1D
        return pos_bin_idx, pos_sample_times, pos_sampling_rate

    if pos_bin_idx.shape[1] != 2:
        raise ValueError(
            f"Expected pos_bin_idx with shape (n_samples, 2), got {pos_bin_idx.shape}"
        )

    # Extract just the X indices (second column)
    pos_x_idx = pos_bin_idx[:, 1]

    return pos_x_idx, pos_sample_times, pos_sampling_rate


def make_1d_rate_maps(
    spike_data: dict[int, np.ndarray],
    pos_sample_times: np.ndarray,
    pos_x_idx: np.ndarray,
    pos_sampling_rate: float,
    max_x_bins: int | None = None,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """
    Generate 1D rate maps (firing rate vs X position).

    Parameters
    ----------
    spike_data : dict
        {unit_id: spike_times}
    pos_sample_times : np.ndarray
        Timestamps for position samples
    pos_x_idx : np.ndarray
        X bin index for each position sample
    pos_sampling_rate : float
        Position sampling rate in Hz
    max_x_bins : int, optional
        Maximum number of X bins. If None, uses max(pos_x_idx) + 1

    Returns
    -------
    rate_maps : dict
        {unit_id: 1D rate map (n_x_bins,)}
    occupancy_map : np.ndarray
        1D occupancy time map (n_x_bins,) in seconds

    Notes
    -----
    Generates unsmoothed rate maps by:
    1. Counting spikes per X bin
    2. Computing occupancy time per X bin
    3. Computing rate = spikes / time
    """
    # Determine number of X bins
    if max_x_bins is None:
        max_x_bins = int(np.nanmax(pos_x_idx)) + 1

    # Compute occupancy time per X bin (in seconds)
    occupancy = np.zeros(max_x_bins)
    dt = 1.0 / pos_sampling_rate  # Time per sample

    for x_idx in pos_x_idx:
        if not np.isnan(x_idx):
            occupancy[int(x_idx)] += dt

    # Compute spike counts per X bin for each unit (vectorized)
    rate_maps = {}
    time_tolerance = 1.0 / pos_sampling_rate

    for unit_id, spike_times in spike_data.items():
        if len(spike_times) == 0:
            rate_map = np.full(max_x_bins, np.nan)
            visited = occupancy > 0
            rate_map[visited] = 0.0
            rate_maps[unit_id] = rate_map
            continue

        # Use searchsorted to find insertion points for all spikes at once
        # This gives us the index where each spike would be inserted to maintain order
        insert_idx = np.searchsorted(pos_sample_times, spike_times)

        # Clip to valid range
        insert_idx = np.clip(insert_idx, 0, len(pos_sample_times) - 1)

        # Check distances to nearest position samples
        time_diffs = np.abs(pos_sample_times[insert_idx] - spike_times)

        # Also check previous index in case it's closer
        prev_idx = np.clip(insert_idx - 1, 0, len(pos_sample_times) - 1)
        prev_diffs = np.abs(pos_sample_times[prev_idx] - spike_times)

        # Use the closest index (prefer earlier index in case of ties, matching argmin)
        use_prev = prev_diffs <= time_diffs
        closest_idx = np.where(use_prev, prev_idx, insert_idx)
        min_diffs = np.where(use_prev, prev_diffs, time_diffs)

        # Filter by time tolerance and valid X indices
        valid_mask = (min_diffs < time_tolerance) & (~np.isnan(pos_x_idx[closest_idx]))
        valid_x_idx = pos_x_idx[closest_idx[valid_mask]]

        # Count spikes per bin using bincount
        spike_counts = np.bincount(valid_x_idx.astype(int), minlength=max_x_bins)[
            :max_x_bins
        ]

        # Compute rate map (Hz)
        rate_map = np.full(max_x_bins, np.nan)
        visited = occupancy > 0
        rate_map[visited] = spike_counts[visited] / occupancy[visited]

        rate_maps[unit_id] = rate_map

    return rate_maps, occupancy
