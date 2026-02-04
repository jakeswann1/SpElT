"""
Generic event-locked LFP analysis utilities.

Provides low-level functions for extracting and pooling LFP windows around events.
These work with any LFP data and event times, enabling analysis of various neural
events such as ripples, spindles, gamma bursts, or theta phase crossings.

Functions:
    extract_event_locked_windows: Extract LFP segments around event times
    pool_windows_across_trials: Pool and average windows from multiple trials
"""

import numpy as np


def extract_event_locked_windows(
    lfp_data: np.ndarray,
    timestamps: np.ndarray,
    event_times: np.ndarray,
    time_window: tuple[float, float],
    sampling_rate: float,
) -> tuple[list[np.ndarray], int]:
    """
    Extract LFP windows around event times.

    For each event time, extracts an LFP segment from [event_time + time_window[0]]
    to [event_time + time_window[1]]. Events too close to recording boundaries or
    with invalid data are excluded.

    Parameters
    ----------
    lfp_data : np.ndarray
        LFP data with shape (n_samples, n_channels)
    timestamps : np.ndarray
        Timestamps in seconds with shape (n_samples,)
    event_times : np.ndarray
        Event timestamps in seconds with shape (n_events,)
    time_window : tuple[float, float]
        Time window around events as (pre_time, post_time) in seconds.
        Example: (-0.1, 0.1) for ±100ms window
    sampling_rate : float
        Sampling frequency in Hz

    Returns
    -------
    valid_windows : list[np.ndarray]
        List of valid LFP windows, each with shape (window_samples, n_channels)
    n_excluded : int
        Number of events excluded due to boundary or validity issues

    Notes
    -----
    Events are excluded if:
    - Window extends beyond recording boundaries
    - Window contains NaN values
    - Window length doesn't match expected length

    Examples
    --------
    >>> # Extract ±100ms windows around ripple peaks
    >>> windows, n_excluded = extract_event_locked_windows(
    ...     lfp_data=lfp,
    ...     timestamps=timestamps,
    ...     event_times=ripple_peaks,
    ...     time_window=(-0.1, 0.1),
    ...     sampling_rate=1000
    ... )
    >>> print(f"Extracted {len(windows)} windows, excluded {n_excluded}")
    """
    # Validate inputs
    if lfp_data.ndim != 2:
        raise ValueError(f"lfp_data must be 2D, got shape {lfp_data.shape}")
    if len(timestamps) != lfp_data.shape[0]:
        raise ValueError(
            f"timestamps length ({len(timestamps)}) must match "
            f"lfp_data samples ({lfp_data.shape[0]})"
        )
    if time_window[0] >= time_window[1]:
        raise ValueError(
            f"time_window[0] ({time_window[0]}) must be < "
            f"time_window[1] ({time_window[1]})"
        )

    # Convert time window to samples
    pre_samples = int(time_window[0] * sampling_rate)  # Negative value
    post_samples = int(time_window[1] * sampling_rate)  # Positive value
    window_length = post_samples - pre_samples

    # Extract windows
    valid_windows = []
    n_excluded = 0

    for event_time in event_times:
        # Find nearest timestamp index
        event_idx = np.argmin(np.abs(timestamps - event_time))

        # Calculate window boundaries
        start_idx = event_idx + pre_samples
        end_idx = event_idx + post_samples

        # Check boundaries
        if start_idx < 0 or end_idx > len(timestamps):
            n_excluded += 1
            continue

        # Extract window
        window = lfp_data[start_idx:end_idx, :]

        # Validate window
        if window.shape[0] != window_length:
            n_excluded += 1
            continue

        if np.any(np.isnan(window)):
            n_excluded += 1
            continue

        valid_windows.append(window)

    return valid_windows, n_excluded


def pool_windows_across_trials(
    windows_by_trial: dict[int, list[np.ndarray]],
) -> tuple[np.ndarray, dict]:
    """
    Pool and average windows from multiple trials.

    Combines LFP windows from multiple trials into a single averaged window.
    All windows must have the same shape.

    Parameters
    ----------
    windows_by_trial : dict[int, list[np.ndarray]]
        Dictionary mapping trial_idx to list of windows for that trial.
        Each window has shape (window_samples, n_channels)

    Returns
    -------
    lfp_mean : np.ndarray
        Mean LFP across all windows, shape (window_samples, n_channels)
    statistics : dict
        Dictionary containing:
        - 'n_windows_total': Total number of windows pooled
        - 'n_trials': Number of trials
        - 'windows_per_trial': Dict mapping trial_idx to number of windows
        - 'lfp_std': Standard deviation across windows
        - 'lfp_sem': Standard error of the mean

    Raises
    ------
    ValueError
        If no valid windows provided or windows have inconsistent shapes

    Examples
    --------
    >>> # Pool ripple windows from multiple trials
    >>> windows_by_trial = {
    ...     0: [window1, window2, window3],
    ...     2: [window4, window5]
    ... }
    >>> lfp_mean, stats = pool_windows_across_trials(windows_by_trial)
    >>> print(
    ...     f"Pooled {stats['n_windows_total']} windows "
    ...     f"from {stats['n_trials']} trials"
    ... )
    """
    # Collect all windows
    all_windows = []
    windows_per_trial = {}

    for trial_idx, windows in windows_by_trial.items():
        all_windows.extend(windows)
        windows_per_trial[trial_idx] = len(windows)

    if len(all_windows) == 0:
        raise ValueError("No valid windows provided")

    # Check shape consistency
    expected_shape = all_windows[0].shape
    for i, window in enumerate(all_windows):
        if window.shape != expected_shape:
            raise ValueError(
                f"Window {i} has shape {window.shape}, expected {expected_shape}"
            )

    # Stack and compute statistics
    windows_array = np.stack(all_windows, axis=0)  # (n_windows, n_samples, n_channels)

    lfp_mean = np.mean(windows_array, axis=0)
    lfp_std = np.std(windows_array, axis=0)
    lfp_sem = lfp_std / np.sqrt(len(all_windows))

    statistics = {
        "n_windows_total": len(all_windows),
        "n_trials": len(windows_by_trial),
        "windows_per_trial": windows_per_trial,
        "lfp_std": lfp_std,
        "lfp_sem": lfp_sem,
    }

    return lfp_mean, statistics
