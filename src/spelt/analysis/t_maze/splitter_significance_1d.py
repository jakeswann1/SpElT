"""Significance testing for splitter cells using 1D (X-only) analysis."""

import numpy as np
from joblib import Parallel, delayed

from spelt.analysis.t_maze.collapse_position_to_1d import (
    collapse_position_bins_to_x,
    make_1d_rate_maps,
)
from spelt.analysis.utils import (
    filter_position_by_windows,
    filter_spikes_by_windows_separate,
)


def get_sector_x_bins(
    pos_header: dict, bin_size: float, sectors: list[int]
) -> tuple[int, int]:
    """
    Get the X bin range that corresponds to specified sectors.

    Parameters
    ----------
    pos_header : dict
        Position header with min/max x/y boundaries
    bin_size : float
        Spatial bin size in cm
    sectors : list of int
        List of sector numbers (1-12)

    Returns
    -------
    x_min_bin : int
        Minimum X bin index
    x_max_bin : int
        Maximum X bin index (inclusive)
    """
    # Sector dimensions
    n_cols = 4

    # Get spatial extent
    x_extent = pos_header["max_x"] - pos_header["min_x"]
    sector_width = x_extent / n_cols

    # Convert bin size from cm to pixels
    scaled_ppm = pos_header.get("scaled_ppm", 400)
    bin_length = bin_size * scaled_ppm / 100  # cm -> m -> pixels

    # Find X range of sectors
    sector_cols = [(s - 1) % n_cols for s in sectors]  # 0-indexed columns
    min_col = min(sector_cols)
    max_col = max(sector_cols)

    # Convert to spatial coordinates
    x_min = pos_header["min_x"] + min_col * sector_width
    x_max = pos_header["min_x"] + (max_col + 1) * sector_width

    # Convert to bin indices
    x_min_bin = int(np.floor((x_min - pos_header["min_x"]) / bin_length))
    x_max_bin = int(np.ceil((x_max - pos_header["min_x"]) / bin_length)) - 1

    return x_min_bin, x_max_bin


def check_minimum_activity(
    left_windows: list[tuple],
    right_windows: list[tuple],
    trial_spike_trains: dict[int, dict],
    unit_id: int,
    min_choices: int = 5,
) -> bool:
    """
    Check if a unit has minimum activity on choice trajectories.

    Parameters
    ----------
    left_windows : list of tuples
        Left-choice trajectory windows
    right_windows : list of tuples
        Right-choice trajectory windows
    trial_spike_trains : dict
        {trial_idx: {unit_id: spike_times}}
    unit_id : int
        Unit ID to check
    min_choices : int
        Minimum number of choices where unit must fire

    Returns
    -------
    bool
        True if unit has sufficient activity
    """
    choices_with_spikes = 0

    # Check left-choice trajectories
    for trial_idx, start_time, end_time in left_windows:
        if trial_idx not in trial_spike_trains:
            continue
        if unit_id not in trial_spike_trains[trial_idx]:
            continue

        spikes = trial_spike_trains[trial_idx][unit_id]
        spikes_in_window = spikes[(spikes >= start_time) & (spikes <= end_time)]
        if len(spikes_in_window) > 0:
            choices_with_spikes += 1

    # Check right-choice trajectories
    for trial_idx, start_time, end_time in right_windows:
        if trial_idx not in trial_spike_trains:
            continue
        if unit_id not in trial_spike_trains[trial_idx]:
            continue

        spikes = trial_spike_trains[trial_idx][unit_id]
        spikes_in_window = spikes[(spikes >= start_time) & (spikes <= end_time)]
        if len(spikes_in_window) > 0:
            choices_with_spikes += 1

    return choices_with_spikes >= min_choices


def precompute_window_position_bins(
    windows: list[tuple], pos_bin_data: dict[int, dict]
) -> tuple[list[np.ndarray], float]:
    """
    Pre-compute filtered position bins (X indices only) for each window.

    This extracts and collapses position data for each window individually,
    but does NOT create synthetic timestamps yet. Timestamps must be created
    after concatenating windows to ensure monotonicity.

    Parameters
    ----------
    windows : list of tuples
        List of (trial_idx, start_time, end_time) tuples
    pos_bin_data : dict
        {trial_idx: bin_data_dict}

    Returns
    -------
    x_bins_per_window : list of np.ndarray
        List of X position bin indices for each window
    pos_sampling_rate : float
        Position sampling rate (Hz)
    """
    x_bins_per_window = []
    pos_sampling_rate = None

    for trial_idx, start_time, end_time in windows:
        if trial_idx not in pos_bin_data:
            x_bins_per_window.append(np.array([]))
            continue

        bin_data = pos_bin_data[trial_idx]
        pos_bin_idx = bin_data["pos_bin_idx"]
        timestamps = bin_data["pos_sample_times"]

        # Extract X bins (handle both tuple and array formats)
        if isinstance(pos_bin_idx, tuple):
            # Tuple format: (x_bins, y_bins)
            x_bins = pos_bin_idx[0]
        elif isinstance(pos_bin_idx, np.ndarray):
            if pos_bin_idx.ndim == 1:
                # Already 1D
                x_bins = pos_bin_idx
            elif pos_bin_idx.shape[1] == 2:
                # 2D array format: columns are [y_idx, x_idx]
                x_bins = pos_bin_idx[:, 1]  # Column 1 is X!
            else:
                raise ValueError(f"Unexpected pos_bin_idx shape: {pos_bin_idx.shape}")
        else:
            raise ValueError(f"Unexpected pos_bin_idx type: {type(pos_bin_idx)}")

        # Find samples within time window
        mask = (timestamps >= start_time) & (timestamps <= end_time)

        if not np.any(mask):
            x_bins_per_window.append(np.array([]))
            continue

        # Store just the X bins for this window
        x_bins_per_window.append(x_bins[mask])

        if pos_sampling_rate is None:
            pos_sampling_rate = bin_data["pos_sampling_rate"]

    if pos_sampling_rate is None:
        pos_sampling_rate = 50.0  # Default fallback

    return x_bins_per_window, pos_sampling_rate


def compute_shuffle_iteration(
    all_windows: list[tuple],
    n_left: int,
    trial_spike_trains: dict[int, dict],
    precomputed_x_bins: list[np.ndarray],
    pos_sampling_rate: float,
    unit_ids: list,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """
    Compute one shuffle iteration with 1D processing using pre-computed position bins.

    Parameters
    ----------
    all_windows : list of tuples
        All trajectory windows (left + right)
    n_left : int
        Number of left trajectory windows
    trial_spike_trains : dict
        {trial_idx: {unit_id: spike_times}}
    precomputed_x_bins : list of np.ndarray
        Pre-computed X position bins for each window in all_windows
    pos_sampling_rate : float
        Position sampling rate (Hz)
    unit_ids : list
        List of unit IDs to process

    Returns
    -------
    tuple
        (left_spike_counts, right_spike_counts, left_occupancy, right_occupancy)
        where spike_counts are {unit_id: 1D array} and occupancy are 1D arrays
    """
    # Randomly shuffle trajectory assignments
    shuffled_indices = np.random.permutation(len(all_windows))
    shuffled_left_indices = shuffled_indices[:n_left]
    shuffled_right_indices = shuffled_indices[n_left:]

    shuffled_left_windows = [all_windows[i] for i in shuffled_left_indices]
    shuffled_right_windows = [all_windows[i] for i in shuffled_right_indices]

    # Filter spikes
    left_spikes, right_spikes = filter_spikes_by_windows_separate(
        shuffled_left_windows,
        shuffled_right_windows,
        trial_spike_trains,
        unit_ids,
        align_to_synthetic_time=True,
    )

    # Concatenate pre-computed position bins based on shuffled assignments
    left_x_bins_list = [precomputed_x_bins[i] for i in shuffled_left_indices]
    right_x_bins_list = [precomputed_x_bins[i] for i in shuffled_right_indices]

    # Combine position bins and create synthetic times (after concatenation)
    if left_x_bins_list and any(len(b) > 0 for b in left_x_bins_list):
        left_x_idx = np.concatenate([b for b in left_x_bins_list if len(b) > 0])
        # Create synthetic monotonic times after concatenation
        dt = 1.0 / pos_sampling_rate if pos_sampling_rate > 0 else 0.02
        left_times = np.arange(len(left_x_idx)) * dt
        left_rate = pos_sampling_rate
    else:
        left_x_idx = np.array([])
        left_times = np.array([])
        left_rate = 0.0

    # Combine position bins for right trajectories
    if right_x_bins_list and any(len(b) > 0 for b in right_x_bins_list):
        right_x_idx = np.concatenate([b for b in right_x_bins_list if len(b) > 0])
        # Create synthetic monotonic times after concatenation
        dt = 1.0 / pos_sampling_rate if pos_sampling_rate > 0 else 0.02
        right_times = np.arange(len(right_x_idx)) * dt
        right_rate = pos_sampling_rate
    else:
        right_x_idx = np.array([])
        right_times = np.array([])
        right_rate = 0.0

    # Generate 1D rate maps
    left_rate_maps, left_occupancy = make_1d_rate_maps(
        left_spikes, left_times, left_x_idx, left_rate
    )
    right_rate_maps, right_occupancy = make_1d_rate_maps(
        right_spikes, right_times, right_x_idx, right_rate
    )

    # Compute spike counts
    left_spike_counts = {}
    right_spike_counts = {}

    for unit_id in unit_ids:
        if unit_id in left_rate_maps:
            rate_map = left_rate_maps[unit_id]
            left_spike_counts[unit_id] = np.where(
                np.isnan(rate_map), np.nan, rate_map * left_occupancy
            )

        if unit_id in right_rate_maps:
            rate_map = right_rate_maps[unit_id]
            right_spike_counts[unit_id] = np.where(
                np.isnan(rate_map), np.nan, rate_map * right_occupancy
            )

    return left_spike_counts, right_spike_counts, left_occupancy, right_occupancy


def splitter_significance_1d(
    left_windows: list[tuple],
    right_windows: list[tuple],
    trial_spike_trains: dict[int, dict],
    pos_bin_data: dict[int, dict],
    unit_ids: list,
    pos_header: dict,
    bin_size: float = 2.5,
    correlation_sectors: list[int] | None = None,
    n_shuffles: int = 1000,
    min_significant_bins: int = 3,
    min_activity_choices: int = 5,
    n_jobs: int = -1,
) -> dict:
    """
    Test splitter cell significance using 1D (X-only) analysis.

    This version collapses position data to 1D (just X) before generating rate maps,
    making the analysis more efficient and conceptually cleaner.

    Parameters
    ----------
    left_windows : list of tuples
        Left-choice trajectory windows (trial_idx, start_time, end_time)
    right_windows : list of tuples
        Right-choice trajectory windows
    trial_spike_trains : dict
        {trial_idx: {unit_id: spike_times}}
    pos_bin_data : dict
        {trial_idx: bin_data_dict}
    unit_ids : list
        List of unit IDs to test
    pos_header : dict
        Position header with spatial boundaries
    bin_size : float
        Spatial bin size in cm
    correlation_sectors : list of int
        Sectors to analyze (e.g., [6, 7] for choice point)
    n_shuffles : int
        Number of shuffle iterations (default: 1000)
    min_significant_bins : int
        Minimum number of significant X bins to be classified as splitter
    min_activity_choices : int
        Minimum number of choices where unit must fire
    n_jobs : int
        Number of parallel jobs (-1 for all cores)

    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'is_splitter': {unit_id: bool}
        - 'n_significant_bins': {unit_id: int}
        - 'p_values_per_bin': {unit_id: np.ndarray} - 1D p-values per X bin
        - 'has_minimum_activity': {unit_id: bool}
        - 'left_rate_profile': {unit_id: np.ndarray} - 1D left rate profile
        - 'right_rate_profile': {unit_id: np.ndarray} - 1D right rate profile
        - 'diff_profile': {unit_id: np.ndarray} - 1D difference profile
        - 'significant_bins_mask': {unit_id: np.ndarray}
            1D boolean mask of significant bins
    """
    if correlation_sectors is None:
        correlation_sectors = [6, 7]

    print(f"  Computing splitter significance (sectors {correlation_sectors})...")
    print(f"    Running {n_shuffles} shuffles with {n_jobs} parallel jobs...")
    print("    Using 1D (X-only) analysis...")

    # Step 1: Generate real 1D rate maps
    left_spikes, right_spikes = filter_spikes_by_windows_separate(
        left_windows,
        right_windows,
        trial_spike_trains,
        unit_ids,
        align_to_synthetic_time=True,
    )

    left_pos_bin_idx, left_pos_times, left_pos_rate = filter_position_by_windows(
        left_windows, pos_bin_data
    )
    right_pos_bin_idx, right_pos_times, right_pos_rate = filter_position_by_windows(
        right_windows, pos_bin_data
    )

    # Collapse to 1D
    left_x_idx, left_times, left_rate = collapse_position_bins_to_x(
        left_pos_bin_idx, left_pos_times, left_pos_rate
    )
    right_x_idx, right_times, right_rate = collapse_position_bins_to_x(
        right_pos_bin_idx, right_pos_times, right_pos_rate
    )

    # Generate 1D rate maps
    left_rate_maps_real, left_occupancy_real = make_1d_rate_maps(
        left_spikes, left_times, left_x_idx, left_rate
    )
    right_rate_maps_real, right_occupancy_real = make_1d_rate_maps(
        right_spikes, right_times, right_x_idx, right_rate
    )

    # Compute spike counts
    left_spike_counts_real = {}
    right_spike_counts_real = {}

    for unit_id in unit_ids:
        if unit_id in left_rate_maps_real:
            rate_map = left_rate_maps_real[unit_id]
            left_spike_counts_real[unit_id] = np.where(
                np.isnan(rate_map), np.nan, rate_map * left_occupancy_real
            )

        if unit_id in right_rate_maps_real:
            rate_map = right_rate_maps_real[unit_id]
            right_spike_counts_real[unit_id] = np.where(
                np.isnan(rate_map), np.nan, rate_map * right_occupancy_real
            )

    # Determine X bin range for correlation sectors
    x_min_bin, x_max_bin = get_sector_x_bins(pos_header, bin_size, correlation_sectors)

    # Step 2: Pre-compute position bins for all windows (optimization)
    print("    Pre-computing position bins for all windows...")
    all_windows = left_windows + right_windows
    n_left = len(left_windows)

    precomputed_x_bins, pos_rate = precompute_window_position_bins(
        all_windows, pos_bin_data
    )

    # Step 3: Run shuffles in parallel
    print("    Running shuffles...")
    shuffle_results = Parallel(n_jobs=n_jobs)(
        delayed(compute_shuffle_iteration)(
            all_windows,
            n_left,
            trial_spike_trains,
            precomputed_x_bins,
            pos_rate,
            unit_ids,
        )
        for _ in range(n_shuffles)
    )

    # Step 4: Calculate significance for each unit
    print("    Calculating X-bin-wise significance...")

    results = {
        "is_splitter": {},
        "n_significant_bins": {},
        "p_values_per_bin": {},
        "has_minimum_activity": {},
        "left_rate_profile": {},
        "right_rate_profile": {},
        "diff_profile": {},
        "significant_bins_mask": {},
    }

    for unit_id in unit_ids:
        # Check minimum activity
        has_activity = check_minimum_activity(
            left_windows,
            right_windows,
            trial_spike_trains,
            unit_id,
            min_activity_choices,
        )
        results["has_minimum_activity"][unit_id] = has_activity

        if not has_activity:
            results["is_splitter"][unit_id] = False
            results["n_significant_bins"][unit_id] = 0
            results["p_values_per_bin"][unit_id] = None
            results["left_rate_profile"][unit_id] = None
            results["right_rate_profile"][unit_id] = None
            results["diff_profile"][unit_id] = None
            results["significant_bins_mask"][unit_id] = None
            continue

        if unit_id not in left_rate_maps_real or unit_id not in right_rate_maps_real:
            results["is_splitter"][unit_id] = False
            results["n_significant_bins"][unit_id] = 0
            results["p_values_per_bin"][unit_id] = None
            results["left_rate_profile"][unit_id] = None
            results["right_rate_profile"][unit_id] = None
            results["diff_profile"][unit_id] = None
            results["significant_bins_mask"][unit_id] = None
            continue

        # Get full rate profiles
        left_profile = left_rate_maps_real[unit_id]
        right_profile = right_rate_maps_real[unit_id]

        # Extract sector bins only
        left_sector = left_profile[x_min_bin : x_max_bin + 1]
        right_sector = right_profile[x_min_bin : x_max_bin + 1]
        diff_sector = left_sector - right_sector

        # Store only the sector-specific profiles (not full profiles)
        results["left_rate_profile"][unit_id] = left_sector
        results["right_rate_profile"][unit_id] = right_sector
        results["diff_profile"][unit_id] = diff_sector

        # Dual occupancy: bins with data in both trajectories
        dual_occ = (~np.isnan(left_sector)) & (~np.isnan(right_sector))

        # Collect shuffled profiles for this unit
        shuffled_diffs_sector = []

        for (
            shuffle_left_sp,
            shuffle_right_sp,
            shuffle_left_occ,
            shuffle_right_occ,
        ) in shuffle_results:
            if unit_id not in shuffle_left_sp or unit_id not in shuffle_right_sp:
                continue

            # Compute shuffle rates
            left_sp = shuffle_left_sp[unit_id]
            right_sp = shuffle_right_sp[unit_id]

            shuffle_left_rate = np.where(
                shuffle_left_occ > 0, left_sp / shuffle_left_occ, np.nan
            )
            shuffle_right_rate = np.where(
                shuffle_right_occ > 0, right_sp / shuffle_right_occ, np.nan
            )

            # Extract sector and compute difference
            shuffle_left_sector = shuffle_left_rate[x_min_bin : x_max_bin + 1]
            shuffle_right_sector = shuffle_right_rate[x_min_bin : x_max_bin + 1]
            shuffle_diff_sector = shuffle_left_sector - shuffle_right_sector

            shuffled_diffs_sector.append(shuffle_diff_sector)

        if len(shuffled_diffs_sector) == 0:
            results["is_splitter"][unit_id] = False
            results["n_significant_bins"][unit_id] = 0
            results["p_values_per_bin"][unit_id] = None
            results["significant_bins_mask"][unit_id] = None
            continue

        # Stack shuffles: shape (n_shuffles, n_sector_bins)
        shuffled_diffs_array = np.stack(shuffled_diffs_sector, axis=0)

        # Calculate p-values per X bin
        real_abs = np.abs(diff_sector)
        shuffled_abs = np.abs(shuffled_diffs_array)

        n_exceed = np.sum(shuffled_abs >= real_abs[np.newaxis, :], axis=0)
        p_values = (n_exceed + 1) / (n_shuffles + 1)

        # Significant bins: p < 0.05 AND has dual occupancy
        significant_bins = (p_values < 0.05) & dual_occ
        n_sig_bins = np.sum(significant_bins)

        results["p_values_per_bin"][unit_id] = p_values
        results["n_significant_bins"][unit_id] = n_sig_bins
        results["is_splitter"][unit_id] = n_sig_bins >= min_significant_bins
        results["significant_bins_mask"][unit_id] = significant_bins

    n_splitters = sum(results["is_splitter"].values())
    print(f"    Found {n_splitters}/{len(unit_ids)} splitter cells")
    print(
        f"    (at least {min_significant_bins} X bins with p < 0.05 "
        f"in sectors {correlation_sectors})"
    )

    return results
