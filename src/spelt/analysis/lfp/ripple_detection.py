"""
Ripple detection for LFP signals.

This module provides functions for detecting high-frequency ripple oscillations
(typically 125-250 Hz) in local field potential (LFP) recordings using envelope-based
thresholding with multiple quality filters.
"""

import numpy as np
from scipy.signal import hilbert

from .filtering import bandpass_filter_lfp


def _compute_ripple_envelope(
    lfp_signal: np.ndarray, fs: float, freq_band: tuple[float, float]
) -> np.ndarray:
    """
    Compute instantaneous amplitude envelope using Hilbert transform.

    Parameters
    ----------
    lfp_signal : np.ndarray
        Raw LFP signal, shape (n_samples,)
    fs : float
        Sampling frequency in Hz
    freq_band : tuple[float, float]
        Frequency band for filtering (freq_min, freq_max) in Hz

    Returns
    -------
    np.ndarray
        Instantaneous amplitude envelope, shape (n_samples,)
    """
    # Bandpass filter to target frequency band
    filtered = bandpass_filter_lfp(lfp_signal, fs, freq_band[0], freq_band[1])

    # Compute analytic signal via Hilbert transform
    analytic_signal = hilbert(filtered)

    # Envelope is the magnitude of the analytic signal
    envelope = np.abs(analytic_signal)

    return envelope


def _find_candidate_events(
    envelope: np.ndarray, threshold: float
) -> list[tuple[int, int]]:
    """
    Find continuous segments where envelope exceeds threshold.

    Parameters
    ----------
    envelope : np.ndarray
        Envelope signal, shape (n_samples,)
    threshold : float
        Threshold value

    Returns
    -------
    list[tuple[int, int]]
        List of (onset_idx, offset_idx) tuples
    """
    # Find where envelope exceeds threshold
    above_threshold = envelope > threshold

    # Find transitions (onsets and offsets)
    transitions = np.diff(above_threshold.astype(int))

    # Onsets: transitions from False to True (value = 1)
    onsets = np.where(transitions == 1)[0] + 1

    # Offsets: transitions from True to False (value = -1)
    offsets = np.where(transitions == -1)[0] + 1

    # Handle edge cases
    if len(above_threshold) > 0:
        if above_threshold[0]:
            onsets = np.concatenate([[0], onsets])
        if above_threshold[-1]:
            offsets = np.concatenate([offsets, [len(envelope)]])

    # Pair onsets and offsets
    events = list(zip(onsets, offsets))

    return events


def _filter_by_duration(
    events: list[tuple[int, int]], fs: float, min_duration_ms: float
) -> list[tuple[int, int]]:
    """
    Filter out events shorter than minimum duration.

    Parameters
    ----------
    events : list[tuple[int, int]]
        List of (onset_idx, offset_idx) tuples
    fs : float
        Sampling frequency in Hz
    min_duration_ms : float
        Minimum duration in milliseconds

    Returns
    -------
    list[tuple[int, int]]
        Filtered events
    """
    # Convert minimum duration to samples
    min_duration_samples = int(min_duration_ms * fs / 1000)

    # Filter events by duration
    filtered_events = []
    for onset, offset in events:
        duration_samples = offset - onset
        if duration_samples >= min_duration_samples:
            filtered_events.append((onset, offset))

    return filtered_events


def _filter_by_power(
    events: list[tuple[int, int]], filtered_signal: np.ndarray, power_threshold: float
) -> list[tuple[int, int]]:
    """
    Filter events by mean power criterion.

    Events must have mean power (signal^2) at least power_threshold times
    the overall mean power of the filtered signal.

    Parameters
    ----------
    events : list[tuple[int, int]]
        List of (onset_idx, offset_idx) tuples
    filtered_signal : np.ndarray
        Bandpass filtered signal, shape (n_samples,)
    power_threshold : float
        Multiplier for mean power

    Returns
    -------
    list[tuple[int, int]]
        Filtered events
    """
    # Compute overall mean power
    mean_power = np.mean(filtered_signal**2)
    threshold = power_threshold * mean_power

    # Filter events by power
    filtered_events = []
    for onset, offset in events:
        event_power = np.mean(filtered_signal[onset:offset] ** 2)
        if event_power >= threshold:
            filtered_events.append((onset, offset))

    return filtered_events


def _filter_by_supra_ripple_ratio(
    events: list[tuple[int, int]],
    ripple_envelope: np.ndarray,
    supra_ripple_envelope: np.ndarray,
    ratio_threshold: float,
) -> list[tuple[int, int]]:
    """
    Filter events by ripple/supra-ripple envelope ratio.

    This filter eliminates high-frequency noise by requiring the ripple band
    envelope to be at least ratio_threshold times the supra-ripple envelope.

    Parameters
    ----------
    events : list[tuple[int, int]]
        List of (onset_idx, offset_idx) tuples
    ripple_envelope : np.ndarray
        Ripple band envelope, shape (n_samples,)
    supra_ripple_envelope : np.ndarray
        Supra-ripple band envelope, shape (n_samples,)
    ratio_threshold : float
        Required ratio threshold

    Returns
    -------
    list[tuple[int, int]]
        Filtered events
    """
    filtered_events = []
    for onset, offset in events:
        # Compute mean envelopes within event window
        ripple_mean = np.mean(ripple_envelope[onset:offset])
        supra_ripple_mean = np.mean(supra_ripple_envelope[onset:offset])

        # Check if ripple envelope is at least ratio_threshold times supra-ripple
        if supra_ripple_mean > 0 and ripple_mean >= ratio_threshold * supra_ripple_mean:
            filtered_events.append((onset, offset))

    return filtered_events


def _find_event_peaks(
    events: list[tuple[int, int]], envelope: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find peak index within each event.

    Parameters
    ----------
    events : list[tuple[int, int]]
        List of (onset_idx, offset_idx) tuples
    envelope : np.ndarray
        Envelope signal, shape (n_samples,)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (peak_indices, onsets, offsets, peak_amplitudes)
    """
    if len(events) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    peak_indices = []
    onsets = []
    offsets = []
    peak_amplitudes = []

    for onset, offset in events:
        # Find peak within event window
        event_envelope = envelope[onset:offset]
        local_peak_idx = np.argmax(event_envelope)
        global_peak_idx = onset + local_peak_idx

        peak_indices.append(global_peak_idx)
        onsets.append(onset)
        offsets.append(offset)
        peak_amplitudes.append(envelope[global_peak_idx])

    return (
        np.array(peak_indices),
        np.array(onsets),
        np.array(offsets),
        np.array(peak_amplitudes),
    )


def _merge_close_events(
    peak_indices: np.ndarray,
    onsets: np.ndarray,
    offsets: np.ndarray,
    peak_amplitudes: np.ndarray,
    fs: float,
    merge_threshold_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge ripples within merge_threshold, keeping first timestamp only.

    Parameters
    ----------
    peak_indices : np.ndarray
        Peak sample indices
    onsets : np.ndarray
        Onset sample indices
    offsets : np.ndarray
        Offset sample indices
    peak_amplitudes : np.ndarray
        Peak amplitudes
    fs : float
        Sampling frequency in Hz
    merge_threshold_ms : float
        Merge threshold in milliseconds

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (merged_peaks, merged_onsets, merged_offsets, merged_amplitudes)
    """
    if len(peak_indices) == 0:
        return peak_indices, onsets, offsets, peak_amplitudes

    # Sort by peak indices
    sort_idx = np.argsort(peak_indices)
    sorted_peaks = peak_indices[sort_idx]
    sorted_onsets = onsets[sort_idx]
    sorted_offsets = offsets[sort_idx]
    sorted_amplitudes = peak_amplitudes[sort_idx]

    # Convert merge threshold to samples
    merge_threshold_samples = int(merge_threshold_ms * fs / 1000)

    # Find time differences between consecutive peaks
    peak_diffs = np.diff(sorted_peaks)

    # Find peaks that should be kept (first of each group)
    keep_mask = np.ones(len(sorted_peaks), dtype=bool)

    # Mark peaks that are too close to previous peak
    keep_mask[1:] = peak_diffs > merge_threshold_samples

    return (
        sorted_peaks[keep_mask],
        sorted_onsets[keep_mask],
        sorted_offsets[keep_mask],
        sorted_amplitudes[keep_mask],
    )


def detect_ripples(
    lfp_signal: np.ndarray,
    fs: float,
    ripple_band: tuple[float, float] = (125, 250),
    supra_ripple_band: tuple[float, float] = (250, 450),
    envelope_threshold: float = 5.0,
    power_threshold: float = 1.5,
    supra_ripple_ratio: float = 2.0,
    min_duration_ms: float = 16.0,
    merge_threshold_ms: float = 300.0,
) -> dict:
    """
    Detect ripple events in LFP signal using envelope-based thresholding.

    Algorithm:
    1. Bandpass filter to ripple band (default 125-250 Hz)
    2. Compute Hilbert envelope for ripple band
    3. Compute Hilbert envelope for supra-ripple band (default 250-450 Hz)
    4. Find candidate events where envelope > envelope_threshold * median envelope
    5. Filter by power: mean power during event > power_threshold * mean power
    6. Filter by supra-ripple ratio:
        ripple envelope > supra_ripple_ratio * supra-ripple envelope
    7. Exclude events shorter than min_duration_ms
    8. Find peak (max envelope) within each event
    9. Merge events within merge_threshold_ms (keep first only)

    Parameters
    ----------
    lfp_signal : np.ndarray
        LFP signal, shape (n_samples,) for single channel
    fs : float
        Sampling frequency in Hz
    ripple_band : tuple[float, float], optional
        Ripple frequency band (freq_min, freq_max) in Hz. Default: (125, 250)
    supra_ripple_band : tuple[float, float], optional
        Supra-ripple frequency band for noise rejection. Default: (250, 450)
    envelope_threshold : float, optional
        Multiplier for median envelope. Default: 5.0
    power_threshold : float, optional
        Multiplier for mean power. Default: 1.5
    supra_ripple_ratio : float, optional
        Required ratio of ripple/supra-ripple envelope. Default: 2.0
    min_duration_ms : float, optional
        Minimum ripple duration in milliseconds. Default: 16.0
    merge_threshold_ms : float, optional
        Merge ripples within this time window in milliseconds. Default: 300.0

    Returns
    -------
    dict
        Dictionary with keys:
        - 'ripple_peaks': np.ndarray of peak indices (sample indices)
        - 'ripple_onsets': np.ndarray of onset indices
        - 'ripple_offsets': np.ndarray of offset indices
        - 'ripple_durations': np.ndarray of durations in ms
        - 'ripple_peak_amplitudes': np.ndarray of peak envelope amplitudes
        - 'n_events': int, number of detected ripples
        - 'ripple_envelope': np.ndarray, full ripple-band envelope
        - 'supra_ripple_envelope': np.ndarray, full supra-ripple envelope

    Examples
    --------
    >>> # Detect ripples in LFP signal
    >>> results = detect_ripples(lfp_signal, fs=1000)
    >>> print(f"Detected {results['n_events']} ripples")
    >>> peak_times = results['ripple_peaks'] / fs  # Convert to seconds
    """
    # Step 1 & 2: Compute ripple band envelope
    ripple_envelope = _compute_ripple_envelope(lfp_signal, fs, ripple_band)

    # Step 3: Compute supra-ripple band envelope
    supra_ripple_envelope = _compute_ripple_envelope(lfp_signal, fs, supra_ripple_band)

    # Step 4: Find candidate events using envelope threshold
    envelope_median = np.median(ripple_envelope)
    threshold = envelope_threshold * envelope_median
    candidate_events = _find_candidate_events(ripple_envelope, threshold)

    # Step 5: Filter by duration
    events = _filter_by_duration(candidate_events, fs, min_duration_ms)

    # Step 6: Filter by power
    filtered_signal = bandpass_filter_lfp(
        lfp_signal, fs, ripple_band[0], ripple_band[1]
    )
    events = _filter_by_power(events, filtered_signal, power_threshold)

    # Step 7: Filter by supra-ripple ratio
    events = _filter_by_supra_ripple_ratio(
        events, ripple_envelope, supra_ripple_envelope, supra_ripple_ratio
    )

    # Step 8: Find peaks within events
    peak_indices, onsets, offsets, peak_amplitudes = _find_event_peaks(
        events, ripple_envelope
    )

    # Step 9: Merge close events
    merged_peaks, merged_onsets, merged_offsets, merged_amplitudes = (
        _merge_close_events(
            peak_indices, onsets, offsets, peak_amplitudes, fs, merge_threshold_ms
        )
    )

    # Compute durations in milliseconds
    durations_ms = (merged_offsets - merged_onsets) * 1000 / fs

    return {
        "ripple_peaks": merged_peaks,
        "ripple_onsets": merged_onsets,
        "ripple_offsets": merged_offsets,
        "ripple_durations": durations_ms,
        "ripple_peak_amplitudes": merged_amplitudes,
        "n_events": len(merged_peaks),
        "ripple_envelope": ripple_envelope,
        "supra_ripple_envelope": supra_ripple_envelope,
    }
