"""
Event-locked current source density (CSD) analysis.

This module provides functions for computing event-locked CSD from multi-trial
LFP recordings. Works with any event type (ripples, phase crossings, spikes, etc.)
and any probe configuration.
"""

import numpy as np

from .csd import _compute_csd
from .event_locked import extract_event_locked_windows, pool_windows_across_trials


def compute_event_locked_csd(
    lfp_data_by_trial: dict[int, dict],
    event_times_by_trial: dict[int, np.ndarray],
    trials_to_include: list[int],
    time_window: tuple[float, float] = (-0.1, 0.1),
    spat_sm: float = 0.5,
    temp_sm: float = 0.0,
) -> dict:
    """
    Compute event-locked CSD from multi-trial LFP data.

    Extracts LFP windows around events, pools across trials, computes CSD,
    and returns time-depth matrix of current source density.

    Parameters
    ----------
    lfp_data_by_trial : dict[int, dict]
        Dictionary mapping trial indices to LFP data dicts containing:
        - 'data': (n_samples, n_channels) LFP array
        - 'timestamps': time values for each sample
        - 'channel_depths_normalised': depth of each channel (μm)
        - 'sampling_rate': sampling rate in Hz
    event_times_by_trial : dict[int, np.ndarray]
        Dictionary mapping trial indices to event times (seconds).
        Events can be any neural events: ripples, phase crossings, spikes, etc.
    trials_to_include : list[int]
        Trial indices to analyze
    time_window : tuple[float, float], optional
        (pre, post) window around events in seconds.
        Default: (-0.1, 0.1) for ±100ms window
    spat_sm : float, optional
        Spatial smoothing kernel width for CSD (Savitzky-Golay filter).
        Default: 0.5. Set to 0 for no smoothing.
    temp_sm : float, optional
        Temporal smoothing kernel width for CSD (Savitzky-Golay filter).
        Default: 0.0 (no smoothing). Increase for noisy data.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'csd_matrix': (n_channels-2, n_timepoints) CSD array
        - 'time_axis': Time points (seconds relative to event, t=0 at event)
        - 'depths': Channel depths (μm) for CSD (n_channels-2,)
        - 'n_events': Total number of events pooled across trials
        - 'mean_lfp': Mean event-locked LFP (optional, for overlay)
        - 'sem_lfp': SEM of event-locked LFP (optional)

    Raises
    ------
    KeyError
        If trial not found in lfp_data_by_trial or event_times_by_trial
    ValueError
        If no valid events found or window extraction fails

    Notes
    -----
    CSD computation:
    - Uses second spatial derivative: CSD_n = -(LFP_{n-1} - 2*LFP_n + LFP_{n+1})
    - Loses 2 channels (top and bottom) due to derivative
    - Negative values indicate current sinks (inward current, typically depolarization)
    - Positive values indicate current sources (outward current)

    Event-locking process:
    1. Extract LFP windows around each event (both trials)
    2. Pool windows across trials (compute mean and SEM)
    3. Compute CSD on pooled mean LFP
    4. Return CSD matrix with time and depth axes

    This function works with any event type by accepting generic event_times:
    - Ripple peaks (125-250 Hz oscillations)
    - Theta phase crossings (specific phase values)
    - Gamma bursts (30-80 Hz oscillations)
    - Sharp-wave events
    - Spike times from single units
    - Behavioral events (rewards, choices, etc.)

    Examples
    --------
    >>> # Ripple-locked CSD
    >>> results = compute_event_locked_csd(
    ...     lfp_data=lfp_data,
    ...     event_times_by_trial=ripple_times,
    ...     trials_to_include=[0, 1, 2],
    ...     time_window=(-0.1, 0.1)  # ±100ms
    ... )
    >>> csd = results['csd_matrix']
    >>> time = results['time_axis']
    >>> depths = results['depths']
    >>>
    >>> # Theta phase-locked CSD (descending zero crossing)
    >>> results = compute_event_locked_csd(
    ...     lfp_data=lfp_data,
    ...     event_times_by_trial=phase_crossing_times,
    ...     trials_to_include=[0, 1, 2],
    ...     time_window=(-0.05, 0.05)  # ±50ms
    ... )
    >>>
    >>> # Spike-triggered CSD
    >>> results = compute_event_locked_csd(
    ...     lfp_data=lfp_data,
    ...     event_times_by_trial=spike_times,
    ...     trials_to_include=[0],
    ...     time_window=(-0.01, 0.01),  # ±10ms
    ...     spat_sm=0.3  # Less smoothing for sharper features
    ... )
    """
    if not trials_to_include:
        raise ValueError("trials_to_include cannot be empty")

    # Extract event-locked windows for each trial
    all_windows = []
    total_events = 0

    for trial_idx in trials_to_include:
        if trial_idx not in lfp_data_by_trial:
            raise KeyError(f"Trial {trial_idx} not found in lfp_data_by_trial")
        if trial_idx not in event_times_by_trial:
            raise KeyError(f"Trial {trial_idx} not found in event_times_by_trial")

        trial_data = lfp_data_by_trial[trial_idx]
        event_times = event_times_by_trial[trial_idx]

        # Skip if no events in this trial
        if len(event_times) == 0:
            continue

        # Extract windows
        windows, n_excluded = extract_event_locked_windows(
            lfp_data=trial_data["data"],
            timestamps=trial_data["timestamps"],
            event_times=event_times,
            time_window=time_window,
            sampling_rate=trial_data["sampling_rate"],
        )

        all_windows.extend(windows)
        total_events += len(windows)

    if not all_windows:
        raise ValueError("No valid event-locked windows extracted from any trial")

    # Pool windows across all trials
    pooled_results = pool_windows_across_trials(all_windows)

    mean_lfp = pooled_results["mean"]  # (n_timepoints, n_channels)
    sem_lfp = pooled_results["sem"]

    # Compute CSD from pooled mean LFP
    csd_matrix = _compute_csd(
        lfp_fragment=mean_lfp,
        spat_sm=int(spat_sm),
        temp_sm=int(temp_sm),
        do_detrend=False,
    )
    # csd_matrix shape: (n_timepoints, n_channels-2)

    # Transpose for standard convention (channels × time)
    csd_matrix = csd_matrix.T  # (n_channels-2, n_timepoints)

    # Create time axis
    first_trial = trials_to_include[0]
    fs = lfp_data_by_trial[first_trial]["sampling_rate"]
    n_timepoints = csd_matrix.shape[1]
    time_axis = np.linspace(time_window[0], time_window[1], n_timepoints)

    # Get channel depths (CSD loses top and bottom channels)
    depths_full = lfp_data_by_trial[first_trial]["channel_depths_normalised"]
    depths_csd = depths_full[1:-1]  # Remove first and last channel

    # Package results
    results = {
        "csd_matrix": csd_matrix,
        "time_axis": time_axis,
        "depths": depths_csd,
        "n_events": total_events,
        "mean_lfp": mean_lfp.T,  # (n_channels, n_timepoints)
        "sem_lfp": sem_lfp.T,  # (n_channels, n_timepoints)
        "time_window": time_window,
        "sampling_rate": fs,
    }

    return results
