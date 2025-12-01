"""
LFP filtering utilities for preprocessing local field potential data.

This module provides flexible bandpass filtering functions that can be used
throughout spelt for consistent LFP preprocessing.
"""

import numpy as np
from scipy.signal import butter, filtfilt, firwin


def bandpass_filter_lfp(
    lfp_data,
    fs,
    freq_min=None,
    freq_max=None,
    peak_freq=None,
    filt_half_bandwidth=None,
    filter_type="butter",
    order=4,
):
    """
    Apply a bandpass filter to LFP data with flexible frequency specification.

    Can specify frequencies either as:
    1. Direct range: freq_min and freq_max
    2. Center frequency with bandwidth: peak_freq and filt_half_bandwidth

    Parameters:
    -----------
    lfp_data : np.ndarray
        LFP data, shape (n_samples, n_channels) or (n_samples,)
    fs : float
        Sampling frequency in Hz
    freq_min : float, optional
        Lower cutoff frequency in Hz. Mutually exclusive with peak_freq.
    freq_max : float, optional
        Upper cutoff frequency in Hz. Mutually exclusive with peak_freq.
    peak_freq : float, optional
        Center frequency in Hz. Mutually exclusive with freq_min/freq_max.
    filt_half_bandwidth : float, optional
        Half-bandwidth around peak_freq in Hz. Required if peak_freq is provided.
        Default is 2 Hz.
    filter_type : str, optional
        Type of filter: 'butter' for Butterworth or 'fir' for FIR (default: 'butter')
    order : int, optional
        Filter order for Butterworth filter (default: 4)

    Returns:
    --------
    np.ndarray
        Filtered LFP data with same shape as input

    Raises:
    -------
    ValueError
        If neither freq_min/freq_max nor peak_freq is provided,
        or if both are provided, or if frequencies are out of valid range.

    Example:
    --------
    >>> # Method 1: Direct frequency range
    >>> lfp_ripple = bandpass_filter_lfp(
    ...     lfp_data, fs=1000, freq_min=150, freq_max=250
    ... )
    >>>
    >>> # Method 2: Center frequency with bandwidth
    >>> lfp_theta = bandpass_filter_lfp(
    ...     lfp_data, fs=1000, peak_freq=8, filt_half_bandwidth=2
    ... )  # Filters to 6-10 Hz
    """
    # Validate input parameters
    if (freq_min is None or freq_max is None) and peak_freq is None:
        raise ValueError(
            "Must provide either (freq_min, freq_max) or "
            "(peak_freq, filt_half_bandwidth)"
        )

    if (freq_min is not None or freq_max is not None) and peak_freq is not None:
        raise ValueError(
            "Cannot provide both (freq_min, freq_max) and peak_freq. "
            "Use one method or the other."
        )

    # Convert peak_freq + bandwidth to freq_min/freq_max
    if peak_freq is not None:
        if filt_half_bandwidth is None:
            filt_half_bandwidth = 2  # Default 2 Hz half-bandwidth
        freq_min = peak_freq - filt_half_bandwidth
        freq_max = peak_freq + filt_half_bandwidth

    # Ensure LFP is 2D
    original_shape = lfp_data.shape
    if lfp_data.ndim == 1:
        lfp_data = lfp_data[:, np.newaxis]

    n_samples, n_channels = lfp_data.shape

    # Validate frequency range
    nyquist = fs / 2
    if freq_min <= 0 or freq_max >= nyquist:
        raise ValueError(
            f"Frequencies must be between 0 and Nyquist ({nyquist} Hz). "
            f"Got: {freq_min}-{freq_max} Hz"
        )

    # Design and apply filter
    if filter_type == "butter":
        filtered_data = _apply_butterworth_filter(
            lfp_data, fs, freq_min, freq_max, order
        )
    elif filter_type == "fir":
        filtered_data = _apply_fir_filter(lfp_data, fs, freq_min, freq_max, n_samples)
    else:
        raise ValueError(f"Unknown filter_type '{filter_type}'. Use 'butter' or 'fir'")

    # Restore original shape
    if len(original_shape) == 1:
        filtered_data = filtered_data.squeeze()

    return filtered_data


def _apply_butterworth_filter(lfp_data, fs, freq_min, freq_max, order):
    """Apply Butterworth bandpass filter to LFP data."""
    nyquist = fs / 2
    low = freq_min / nyquist
    high = freq_max / nyquist
    b, a = butter(order, [low, high], btype="band")

    n_channels = lfp_data.shape[1]
    filtered_data = np.zeros_like(lfp_data)
    for ch_idx in range(n_channels):
        filtered_data[:, ch_idx] = filtfilt(b, a, lfp_data[:, ch_idx])

    return filtered_data


def _apply_fir_filter(lfp_data, fs, freq_min, freq_max, n_samples):
    """Apply FIR bandpass filter to LFP data."""
    # Design FIR filter (same approach as get_signal_phase)
    filter_order = min(int(fs / 2), 1001)
    if filter_order % 2 == 0:
        filter_order += 1  # Ensure odd order for zero phase filtering

    filter_taps = firwin(
        filter_order, [freq_min, freq_max], pass_zero=False, window="blackman", fs=fs
    )

    # Calculate appropriate padding for filtfilt
    pad_length = min(3 * (len(filter_taps) - 1), n_samples - 1, 1000)

    # Apply filter to each channel
    n_channels = lfp_data.shape[1]
    filtered_data = np.zeros_like(lfp_data)
    for ch_idx in range(n_channels):
        filtered_data[:, ch_idx] = filtfilt(
            filter_taps, 1, lfp_data[:, ch_idx], padlen=pad_length
        )

    return filtered_data


def get_filter_frequencies(
    peak_freq=None, filt_half_bandwidth=2, freq_min=None, freq_max=None
):
    """
    Convert between different frequency specifications.

    Helper function to convert between peak_freq + bandwidth and freq_min/freq_max.

    Parameters:
    -----------
    peak_freq : float, optional
        Center frequency in Hz
    filt_half_bandwidth : float, optional
        Half-bandwidth around peak_freq in Hz (default: 2)
    freq_min : float, optional
        Lower cutoff frequency in Hz
    freq_max : float, optional
        Upper cutoff frequency in Hz

    Returns:
    --------
    tuple[float, float]
        (freq_min, freq_max) tuple

    Example:
    --------
    >>> # Convert from peak frequency to range
    >>> freq_min, freq_max = get_filter_frequencies(peak_freq=8, filt_half_bandwidth=2)
    >>> print(freq_min, freq_max)  # 6.0, 10.0
    >>>
    >>> # Pass through existing range
    >>> freq_min, freq_max = get_filter_frequencies(freq_min=150, freq_max=250)
    >>> print(freq_min, freq_max)  # 150, 250
    """
    if peak_freq is not None:
        if freq_min is not None or freq_max is not None:
            raise ValueError("Cannot provide both peak_freq and freq_min/freq_max")
        freq_min = peak_freq - filt_half_bandwidth
        freq_max = peak_freq + filt_half_bandwidth
    elif freq_min is None or freq_max is None:
        raise ValueError("Must provide either peak_freq or both freq_min and freq_max")

    return freq_min, freq_max
