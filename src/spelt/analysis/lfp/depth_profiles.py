"""
Depth-resolved LFP analysis functions.

This module provides functions for computing depth profiles of LFP properties
across linear probe recordings. These functions work with any frequency band
and are not specific to particular brain regions or oscillations.
"""

from typing import Optional

import numpy as np

from .frequency_power import compute_band_power


def compute_depth_power_profile(
    lfp_data_by_trial: dict[int, dict],
    trials_to_include: list[int],
    freq_min: float,
    freq_max: float,
    reference_depth_idx: Optional[int] = None,
    fs: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute normalized band power profile across recording depths.

    Concatenates LFP from multiple trials and computes power in specified
    frequency band for each recording depth. Optionally normalizes power
    to a reference depth (e.g., pyramidal layer, reference electrode).

    Parameters
    ----------
    lfp_data_by_trial : dict[int, dict]
        Dictionary mapping trial indices to LFP data dicts containing:
        - 'data': (n_samples, n_channels) LFP array
        - 'timestamps': time values for each sample
        - 'channel_depths_normalised': depth of each channel (μm)
        - 'sampling_rate': sampling rate in Hz
    trials_to_include : list[int]
        Trial indices to include in concatenation
    freq_min : float
        Minimum frequency of band (Hz)
    freq_max : float
        Maximum frequency of band (Hz)
    reference_depth_idx : int, optional
        Channel index for power normalization. If None, no normalization.
        Typically set to channel at depth=0 (e.g., pyramidal layer).
    fs : float, optional
        Sampling rate override. If None, uses rate from lfp_data.

    Returns
    -------
    depths_sorted : np.ndarray
        Channel depths sorted from superficial to deep (μm)
    power_sorted : np.ndarray
        Raw band power at each depth (μV²)
    normalized_power_sorted : np.ndarray
        Power normalized to reference_depth_idx (dimensionless, 1.0 at reference)
        If reference_depth_idx is None, this equals power_sorted.

    Raises
    ------
    KeyError
        If required keys missing from lfp_data_by_trial
    ValueError
        If trials_to_include is empty or no valid data found

    Notes
    -----
    This function works with any frequency band by specifying freq_min and freq_max:
    - Theta (4-12 Hz): freq_min=4, freq_max=12
    - Gamma (30-80 Hz): freq_min=30, freq_max=80
    - Ripple (125-250 Hz): freq_min=125, freq_max=250

    The normalization to a reference depth allows comparison across sessions
    where absolute power may vary due to electrode impedance or recording quality.

    Examples
    --------
    >>> # Theta (4-12 Hz) power profile
    >>> depths, power, norm_power = compute_depth_power_profile(
    ...     lfp_data, trials=[0,1,2], freq_min=4, freq_max=12,
    ...     reference_depth_idx=10  # Pyramidal layer at index 10
    ... )
    >>> plt.plot(depths, norm_power)
    >>> plt.xlabel('Depth (μm)')
    >>> plt.ylabel('Normalized theta power')
    >>>
    >>> # Gamma (30-80 Hz) power profile without normalization
    >>> depths, power, _ = compute_depth_power_profile(
    ...     lfp_data, trials=[0,1,2], freq_min=30, freq_max=80
    ... )
    >>> plt.plot(depths, power)
    """
    if not trials_to_include:
        raise ValueError("trials_to_include cannot be empty")

    # Concatenate LFP data across trials
    lfp_segments = []
    for trial_idx in trials_to_include:
        if trial_idx not in lfp_data_by_trial:
            available = sorted(lfp_data_by_trial.keys())
            raise KeyError(
                f"Trial {trial_idx} (type: {type(trial_idx)}) not found in "
                f"lfp_data_by_trial. Available trials: {available} "
                f"(types: {[type(k) for k in available]})"
            )

        trial_data = lfp_data_by_trial[trial_idx]
        lfp_segments.append(trial_data["data"])  # (n_samples, n_channels)

    # Concatenate along time axis
    lfp_concatenated = np.concatenate(
        lfp_segments, axis=0
    )  # (total_samples, n_channels)

    # Get sampling rate
    if fs is None:
        fs = lfp_data_by_trial[trials_to_include[0]]["sampling_rate"]

    # Get depths from first trial (assume consistent across trials)
    depths = lfp_data_by_trial[trials_to_include[0]]["channel_depths_normalised"]

    # Compute band power for each channel
    band_powers = compute_band_power(
        lfp_data=lfp_concatenated,
        fs=fs,
        freq_min=freq_min,
        freq_max=freq_max,
        method="integral",  # Trapezoidal rule integration
    )

    # Normalize to reference depth
    # If not provided, auto-detect depth closest to 0
    if reference_depth_idx is None:
        reference_depth_idx = np.argmin(np.abs(depths))

    power_at_reference = band_powers[reference_depth_idx]
    if power_at_reference == 0:
        raise ValueError(
            f"Power at reference depth (index {reference_depth_idx}) is zero. "
            "Cannot normalize."
        )
    normalized_power = band_powers / power_at_reference

    # Sort by depth (superficial to deep)
    sort_idx = np.argsort(depths)
    depths_sorted = depths[sort_idx]
    power_sorted = band_powers[sort_idx]
    normalized_power_sorted = normalized_power[sort_idx]

    return depths_sorted, power_sorted, normalized_power_sorted


def compute_conditional_depth_power_profile(
    lfp_data_by_trial: dict[int, dict],
    trials_to_include: list[int],
    time_windows_by_trial: dict[int, list[tuple[float, float]]],
    freq_min: float,
    freq_max: float,
    reference_depth_idx: Optional[int] = None,
    fs: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute band power profile from LFP within specified time windows.

    Similar to compute_depth_power_profile() but filters LFP to only include
    data within specified time windows (e.g., during specific behavioral states,
    maze sectors, task phases, or experimental conditions).

    Parameters
    ----------
    lfp_data_by_trial : dict[int, dict]
        Dictionary mapping trial indices to LFP data dicts
        (same format as compute_depth_power_profile)
    trials_to_include : list[int]
        Trial indices to process
    time_windows_by_trial : dict[int, list[tuple[float, float]]]
        Dictionary mapping trial_idx to list of (start_time, end_time)
        windows in seconds.
        Only LFP samples within these windows are included in power computation.
    freq_min : float
        Minimum frequency of band (Hz)
    freq_max : float
        Maximum frequency of band (Hz)
    reference_depth_idx : int, optional
        Channel index for power normalization
    fs : float, optional
        Sampling rate override

    Returns
    -------
    depths_sorted, power_sorted, normalized_power_sorted : tuple
        Same as compute_depth_power_profile()

    Raises
    ------
    KeyError
        If trial not found in lfp_data_by_trial or time_windows_by_trial
    ValueError
        If no valid data within time windows

    Notes
    -----
    This function enables state-dependent or condition-specific power profiling:
    - Behavioral states: run vs. rest, exploration vs. goal-directed
    - Spatial zones: specific maze sectors, reward zones, decision points
    - Task phases: sample vs. choice, delay periods
    - Experimental manipulations: optogenetic stimulation periods

    Examples
    --------
    >>> # Power during specific maze sectors
    >>> time_windows = {
    ...     0: [(10.5, 15.2), (20.1, 25.3)],  # Trial 0: two visits to sector
    ...     1: [(5.0, 10.5)]                   # Trial 1: one visit
    ... }
    >>> depths, power, norm = compute_conditional_depth_power_profile(
    ...     lfp_data, [0,1], time_windows, freq_min=4, freq_max=12
    ... )
    >>>
    >>> # Power during running (speed > threshold)
    >>> # (time_windows extracted from behavioral data)
    >>> depths, power, norm = compute_conditional_depth_power_profile(
    ...     lfp_data, trials, running_windows, freq_min=30, freq_max=80
    ... )
    """
    if not trials_to_include:
        raise ValueError("trials_to_include cannot be empty")

    # Extract LFP segments within time windows
    lfp_segments = []

    for trial_idx in trials_to_include:
        if trial_idx not in lfp_data_by_trial:
            available = sorted(lfp_data_by_trial.keys())
            raise KeyError(
                f"Trial {trial_idx} (type: {type(trial_idx)}) not found in "
                f"lfp_data_by_trial. Available trials: {available} "
                f"(types: {[type(k) for k in available]})"
            )
        if trial_idx not in time_windows_by_trial:
            available = sorted(time_windows_by_trial.keys())
            raise KeyError(
                f"Trial {trial_idx} (type: {type(trial_idx)}) not found in "
                f"time_windows_by_trial. Available trials: {available} "
                f"(types: {[type(k) for k in available]})"
            )

        trial_data = lfp_data_by_trial[trial_idx]
        lfp_data = trial_data["data"]  # (n_samples, n_channels)
        # Prefer relative timestamps if available (for consistency with position)
        timestamps = trial_data.get("timestamps_relative", trial_data["timestamps"])
        time_windows = time_windows_by_trial[trial_idx]

        # Extract segments within time windows
        for start_time, end_time in time_windows:
            # Find samples within time window
            mask = (timestamps >= start_time) & (timestamps <= end_time)

            if np.any(mask):
                lfp_segment = lfp_data[mask, :]
                lfp_segments.append(lfp_segment)

    if not lfp_segments:
        raise ValueError("No valid LFP data found within specified time windows")

    # Concatenate all segments
    lfp_concatenated = np.concatenate(
        lfp_segments, axis=0
    )  # (total_samples, n_channels)

    # Get sampling rate
    if fs is None:
        fs = lfp_data_by_trial[trials_to_include[0]]["sampling_rate"]

    # Get depths from first trial
    depths = lfp_data_by_trial[trials_to_include[0]]["channel_depths_normalised"]

    # Compute band power
    band_powers = compute_band_power(
        lfp_data=lfp_concatenated,
        fs=fs,
        freq_min=freq_min,
        freq_max=freq_max,
        method="integral",
    )

    # Normalize to reference depth
    # If not provided, auto-detect depth closest to 0
    if reference_depth_idx is None:
        reference_depth_idx = np.argmin(np.abs(depths))

    power_at_reference = band_powers[reference_depth_idx]
    if power_at_reference == 0:
        raise ValueError(
            f"Power at reference depth (index {reference_depth_idx}) is zero. "
            "Cannot normalize."
        )
    normalized_power = band_powers / power_at_reference

    # Sort by depth
    sort_idx = np.argsort(depths)
    depths_sorted = depths[sort_idx]
    power_sorted = band_powers[sort_idx]
    normalized_power_sorted = normalized_power[sort_idx]

    return depths_sorted, power_sorted, normalized_power_sorted
