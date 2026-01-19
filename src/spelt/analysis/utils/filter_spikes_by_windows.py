"""Functions for filtering spikes by time windows."""

import numpy as np


def filter_spikes_by_windows(
    time_windows: list[tuple],
    trial_spike_trains: dict[int, dict],
    unit_ids: list | None = None,
    align_to_synthetic_time: bool = False,
) -> dict[int, np.ndarray]:
    """
    Filter spikes by time windows and pool across trajectories.

    This function extracts spikes that occur within specified time windows
    and pools them together. Useful for analyzing neural activity during
    specific behavioral epochs (e.g., choice trajectories in T-maze).

    Parameters
    ----------
    time_windows : list of tuples
        Each tuple is (trial_idx, start_time, end_time) specifying a time window
        from which to extract spikes
    trial_spike_trains : dict
        Nested dict mapping {trial_idx: {unit_id: spike_times}}
        where spike_times is a numpy array of spike timestamps in seconds
    unit_ids : list, optional
        List of unit IDs to process. If None, processes all units found
        in the first trial.
    align_to_synthetic_time : bool, optional
        If True, align spike times to synthetic monotonically increasing time
        coordinates matching the behavior of filter_position_by_windows.
        Use this when pooling spikes across multiple trajectory windows for
        rate map generation. Default: False (preserves original timestamps).

    Returns
    -------
    pooled_spikes : dict
        {unit_id: concatenated_spike_times} where spike times from all
        matching time windows are pooled together

    Examples
    --------
    >>> # Get spikes during left-choice trajectories
    >>> left_windows = [(0, 10.5, 15.2), (0, 20.1, 24.8), (1, 5.0, 9.5)]
    >>> trial_spikes = {
    ...     0: {42: np.array([11.0, 12.5, 21.0, 22.0])},
    ...     1: {42: np.array([6.0, 7.5, 8.0])}
    ... }
    >>> pooled = filter_spikes_by_windows(left_windows, trial_spikes, [42])
    >>> pooled[42]  # Spikes from all three windows concatenated
    array([11.0, 12.5, 21.0, 22.0, 6.0, 7.5, 8.0])

    Notes
    -----
    - When align_to_synthetic_time=False: Spike times retain original timestamps
    - When align_to_synthetic_time=True: Spike times are aligned to synthetic
      time coordinates [0, dt, 2*dt, ...] matching filter_position_by_windows
    - Windows from different trials are pooled together
    - Empty arrays are returned for units with no spikes in any window
    """
    # Determine which units to process
    if unit_ids is None:
        # Get units from first trial
        first_trial = list(trial_spike_trains.keys())[0]
        unit_ids = list(trial_spike_trains[first_trial].keys())

    # Initialize containers for each unit
    spikes_by_unit = {unit_id: [] for unit_id in unit_ids}

    # Track cumulative time for synthetic alignment
    cumulative_synthetic_time = 0.0

    # Extract spikes from each time window
    for trial_idx, start_time, end_time in time_windows:
        if trial_idx not in trial_spike_trains:
            continue

        trial_spikes = trial_spike_trains[trial_idx]

        for unit_id in unit_ids:
            if unit_id not in trial_spikes:
                continue

            # Get spikes in this time window
            unit_spike_times = trial_spikes[unit_id]
            spikes_in_window = unit_spike_times[
                (unit_spike_times >= start_time) & (unit_spike_times <= end_time)
            ]

            if len(spikes_in_window) > 0:
                if align_to_synthetic_time:
                    # Convert to relative time within window
                    relative_spikes = spikes_in_window - start_time
                    # Offset by cumulative synthetic time
                    aligned_spikes = relative_spikes + cumulative_synthetic_time
                    spikes_by_unit[unit_id].append(aligned_spikes)
                else:
                    spikes_by_unit[unit_id].append(spikes_in_window)

        # Update cumulative time for next window
        if align_to_synthetic_time:
            cumulative_synthetic_time += end_time - start_time

    # Concatenate spikes across all windows
    pooled_spikes = {
        unit_id: (np.concatenate(spike_list) if len(spike_list) > 0 else np.array([]))
        for unit_id, spike_list in spikes_by_unit.items()
    }

    return pooled_spikes


def filter_spikes_by_windows_per_trial(
    time_windows: list[tuple],
    trial_spike_trains: dict[int, dict],
    unit_ids: list | None = None,
) -> dict[int, dict[int, list[np.ndarray]]]:
    """
    Filter spikes by time windows, keeping trials separate.

    Similar to filter_spikes_by_windows but returns spikes organized
    by trial, allowing for per-trial analyses.

    Parameters
    ----------
    time_windows : list of tuples
        Each tuple is (trial_idx, start_time, end_time)
    trial_spike_trains : dict
        {trial_idx: {unit_id: spike_times}}
    unit_ids : list, optional
        List of unit IDs to process

    Returns
    -------
    trial_windowed_spikes : dict
        {trial_idx: {unit_id: [window1_spikes, window2_spikes, ...]}}
        where each window's spikes are kept separate as a list element

    Examples
    --------
    >>> # Useful for analyzing trial-by-trial variability
    >>> windows = [(0, 10, 15), (0, 20, 25), (1, 5, 10)]
    >>> trial_spikes = {
    ...     0: {42: np.array([11.0, 21.0])},
    ...     1: {42: np.array([6.0, 7.0])}
    ... }
    >>> result = filter_spikes_by_windows_per_trial(windows, trial_spikes, [42])
    >>> result[0][42]  # Two separate windows from trial 0
    [array([11.0]), array([21.0])]
    >>> result[1][42]  # One window from trial 1
    [array([6.0, 7.0])]
    """
    # Determine which units to process
    if unit_ids is None:
        first_trial = list(trial_spike_trains.keys())[0]
        unit_ids = list(trial_spike_trains[first_trial].keys())

    # Initialize nested structure: {trial: {unit: []}}
    trial_windowed_spikes = {}

    for trial_idx, start_time, end_time in time_windows:
        if trial_idx not in trial_spike_trains:
            continue

        if trial_idx not in trial_windowed_spikes:
            trial_windowed_spikes[trial_idx] = {unit_id: [] for unit_id in unit_ids}

        trial_spikes = trial_spike_trains[trial_idx]

        for unit_id in unit_ids:
            if unit_id not in trial_spikes:
                trial_windowed_spikes[trial_idx][unit_id].append(np.array([]))
                continue

            # Get spikes in this time window
            unit_spike_times = trial_spikes[unit_id]
            spikes_in_window = unit_spike_times[
                (unit_spike_times >= start_time) & (unit_spike_times <= end_time)
            ]

            trial_windowed_spikes[trial_idx][unit_id].append(spikes_in_window)

    return trial_windowed_spikes


def filter_spikes_by_windows_separate(
    left_windows: list[tuple],
    right_windows: list[tuple],
    trial_spike_trains: dict[int, dict],
    unit_ids: list | None = None,
    align_to_synthetic_time: bool = False,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Filter spikes by separate left and right time windows.

    Convenience function for splitter cell analysis where you have
    separate windows for left and right choice trajectories.

    Parameters
    ----------
    left_windows : list of tuples
        Time windows for left-choice trajectories
    right_windows : list of tuples
        Time windows for right-choice trajectories
    trial_spike_trains : dict
        {trial_idx: {unit_id: spike_times}}
    unit_ids : list, optional
        List of unit IDs to process
    align_to_synthetic_time : bool, optional
        If True, align spike times to synthetic monotonically increasing time
        coordinates. Use this when pooling for rate map generation. Default: False.

    Returns
    -------
    left_spikes : dict
        {unit_id: pooled_spike_times} for left-choice trajectories
    right_spikes : dict
        {unit_id: pooled_spike_times} for right-choice trajectories

    Examples
    --------
    >>> left_wins = [(0, 10, 15), (0, 20, 25)]
    >>> right_wins = [(0, 30, 35), (1, 5, 10)]
    >>> trial_spikes = {...}
    >>> left_spikes, right_spikes = filter_spikes_by_windows_separate(
    ...     left_wins, right_wins, trial_spikes
    ... )
    """
    left_spikes = filter_spikes_by_windows(
        left_windows, trial_spike_trains, unit_ids, align_to_synthetic_time
    )
    right_spikes = filter_spikes_by_windows(
        right_windows, trial_spike_trains, unit_ids, align_to_synthetic_time
    )

    return left_spikes, right_spikes


def filter_spikes_by_sectors(
    spike_data: dict[int, np.ndarray],
    spike_times_already_aligned: bool,
    pos_bin_idx: tuple[np.ndarray, np.ndarray],
    pos_sample_times: np.ndarray,
    pos_sampling_rate: float,
    sectors: list[int],
    pos_header: dict,
    bin_size: float = 2.5,
) -> dict[int, np.ndarray]:
    """
    Filter spike times to only those occurring when animal was in specified sectors.

    Matches each spike to the nearest position sample and filters out spikes
    that occurred when the animal was outside the target sectors.

    Parameters
    ----------
    spike_data : dict
        {unit_id: spike_times} - spike times for each unit
    spike_times_already_aligned : bool
        If True, spike times are in synthetic time coordinates matching pos_sample_times
        If False, spike times are in original recording timestamps.
    pos_bin_idx : tuple of (x_bins, y_bins)
        Position bin indices (already time-filtered)
    pos_sample_times : np.ndarray
        Timestamps for position samples (synthetic if align_to_synthetic_time was used)
    pos_sampling_rate : float
        Position sampling rate in Hz
    sectors : list of int
        List of sector numbers to include
    pos_header : dict
        Position header with spatial boundaries
    bin_size : float
        Spatial bin size in cm (default: 2.5)

    Returns
    -------
    filtered_spike_data : dict
        {unit_id: filtered_spike_times} - only spikes in target sectors

    Notes
    -----
    - Uses same spike-to-position matching logic as make_1d_rate_maps()
    - Time tolerance = 1.0 / pos_sampling_rate
    - Spikes without valid position match are excluded

    Examples
    --------
    >>> # Filter spikes to only center stem
    >>> filtered_spikes = filter_spikes_by_sectors(
    ...     spike_data={42: np.array([1.0, 2.0, 3.0])},
    ...     spike_times_already_aligned=True,
    ...     pos_bin_idx=(x_bins, y_bins),
    ...     pos_sample_times=times,
    ...     pos_sampling_rate=50.0,
    ...     sectors=[6, 7],
    ...     pos_header=header,
    ...     bin_size=2.5
    ... )
    """
    from spelt.analysis.t_maze.assign_sectors import bin_indices_to_sectors

    x_bins, y_bins = pos_bin_idx

    # Handle empty position data
    if len(x_bins) == 0:
        return {unit_id: np.array([]) for unit_id in spike_data.keys()}

    # Convert bin indices to sector numbers for all position samples
    sector_numbers = bin_indices_to_sectors(x_bins, y_bins, pos_header, bin_size)

    # Create mask for target sectors
    in_target_sectors = np.isin(sector_numbers, sectors) & ~np.isnan(sector_numbers)

    # Time tolerance for spike-position matching
    time_tolerance = 1.0 / pos_sampling_rate

    # Filter spikes for each unit
    filtered_spike_data = {}

    for unit_id, spike_times in spike_data.items():
        if len(spike_times) == 0:
            filtered_spike_data[unit_id] = np.array([])
            continue

        # Find nearest position sample for each spike
        # Same logic as make_1d_rate_maps()
        insert_idx = np.searchsorted(pos_sample_times, spike_times)
        insert_idx = np.clip(insert_idx, 0, len(pos_sample_times) - 1)

        # Check distances to nearest position samples
        time_diffs = np.abs(pos_sample_times[insert_idx] - spike_times)

        # Also check previous index
        prev_idx = np.clip(insert_idx - 1, 0, len(pos_sample_times) - 1)
        prev_diffs = np.abs(pos_sample_times[prev_idx] - spike_times)

        # Use closest index
        use_prev = prev_diffs <= time_diffs
        closest_idx = np.where(use_prev, prev_idx, insert_idx)
        min_diffs = np.where(use_prev, prev_diffs, time_diffs)

        # Valid spikes: within time tolerance AND in target sectors
        valid_mask = (min_diffs < time_tolerance) & in_target_sectors[closest_idx]

        filtered_spike_data[unit_id] = spike_times[valid_mask]

    return filtered_spike_data
