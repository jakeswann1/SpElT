from typing import Union

import numpy as np
from scipy.signal import filtfilt, firwin, hilbert


def get_signal_phase(
    lfp: np.ndarray,
    sampling_rate: float,
    peak_freq: float | np.ndarray,
    clip_value: float = 0,
    filt_half_bandwidth: float = 2,
    power_thresh: float = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Efficiently calculates the signal phase timeseries from an LFP signal.
    Vectorized implementation that handles both 1D and 2D inputs.
    Supports channel-specific peak frequencies.

    Parameters:
    -----------
    lfp : np.ndarray
        Local Field Potential time series.
        Can be 1D (samples,) or 2D (samples, channels).
    sampling_rate : float
        The sampling rate of the LFP.
    peak_freq : float or np.ndarray
        The central frequency around which the LFP is filtered.
        Can be a single value for all channels or an array with one value per channel.
    clip_value : float, optional
        The value above which the LFP is considered clipped. Default is 0.
    filt_half_bandwidth : float, optional
        Half bandwidth for filtering. Default is 2 Hz.
    power_thresh : float, optional
        Threshold (percentile) for minimum power per cycle. Default is 5.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        - signal_phase: Signal phase (in radians) timeseries,
        with low power cycles set to NaN. Same shape as input lfp.
        - cycle_numbers: Cycle number for each sample, with bad cycles set to NaN.
          Same shape as input lfp.
    """
    # Handle both 1D and 2D inputs
    input_is_1d = lfp.ndim == 1
    if input_is_1d:
        lfp = lfp.reshape(-1, 1)

    n_samples, n_channels = lfp.shape

    # Handle peak_freq as single value or array
    if isinstance(peak_freq, (int, float)):
        peak_freqs = np.full(n_channels, peak_freq)
    else:
        peak_freqs = np.asarray(peak_freq)
        if len(peak_freqs) != n_channels:
            raise ValueError(
                f"""Length of peak_freq ({len(peak_freqs)})
                must match number of channels ({n_channels})"""
            )

    # Pre-allocate output arrays
    signal_phase_out = np.zeros((n_samples, n_channels))
    cycle_numbers_out = np.zeros((n_samples, n_channels), dtype=float)

    # Process each channel with its specific peak frequency
    for c in range(n_channels):
        # Get filter parameters for this channel
        low_freq = peak_freqs[c] - filt_half_bandwidth
        high_freq = peak_freqs[c] + filt_half_bandwidth

        # Optimize filter design
        filter_order = min(int(sampling_rate / 2), 1001)
        if filter_order % 2 == 0:
            filter_order += 1  # Ensure odd order for zero phase filtering

        # Design bandpass filter for this channel
        filter_taps = firwin(
            filter_order,
            [low_freq, high_freq],
            pass_zero=False,
            window="blackman",
            fs=sampling_rate,
        )

        # Calculate appropriate padding for filtfilt
        pad_length = min(3 * (len(filter_taps) - 1), n_samples - 1, 1000)

        # Filter LFP for this channel
        filtered_lfp = filtfilt(filter_taps, 1, lfp[:, c], padlen=pad_length)

        # Hilbert transform
        analytic_signal = hilbert(filtered_lfp)

        # Extract phase
        signal_phase = np.angle(analytic_signal) % (2 * np.pi)

        # Identify phase transitions efficiently
        phase_diff = np.diff(signal_phase)
        phase_transitions = np.zeros(n_samples, dtype=bool)
        phase_transitions[1:] = phase_diff < -np.pi

        # Get cycle numbers efficiently
        cycle_numbers = np.cumsum(phase_transitions)

        # Calculate cycle statistics in one pass
        max_cycle = cycle_numbers[-1] + 1
        bincount_cycles = np.bincount(cycle_numbers, minlength=max_cycle)

        # Power calculation
        power = filtered_lfp**2
        power_per_cycle = np.bincount(cycle_numbers, weights=power, minlength=max_cycle)

        # Avoid division by zero
        mask = bincount_cycles > 0
        power_per_cycle[mask] /= bincount_cycles[mask]

        # Identify bad cycles - these are separate O(n) operations
        bad_cycles_set = set()

        # 1. Clipped cycles
        if clip_value > 0:
            clipped_indices = np.where(np.abs(lfp[:, c]) > clip_value)[0]
            if len(clipped_indices) > 0:
                clipped_cycles = np.unique(cycle_numbers[clipped_indices])
                clipped_cycles = clipped_cycles[clipped_cycles < max_cycle]
                bad_cycles_set.update(clipped_cycles.astype(int))

        # 2. Bad power cycles
        power_threshold = np.nanpercentile(power_per_cycle, power_thresh)
        bad_power_cycles = np.where(power_per_cycle < power_threshold)[0]
        bad_power_cycles = bad_power_cycles[bad_power_cycles < max_cycle]
        bad_cycles_set.update(bad_power_cycles.astype(int))

        # 3. Bad length cycles
        cycle_length_lim = np.ceil(
            1 / np.array([high_freq, low_freq]) * sampling_rate
        ).astype(int)
        bad_length_mask = (bincount_cycles < cycle_length_lim[0]) | (
            bincount_cycles > cycle_length_lim[1]
        )
        bad_length_cycles = np.where(bad_length_mask)[0]
        bad_length_cycles = bad_length_cycles[bad_length_cycles < max_cycle]
        bad_cycles_set.update(bad_length_cycles.astype(int))

        # Create mask for bad samples efficiently
        is_bad_cycle = np.zeros(max_cycle, dtype=bool)
        if bad_cycles_set:
            bad_cycles_array = np.array(list(bad_cycles_set), dtype=int)
            is_bad_cycle[bad_cycles_array] = True
        bad_sample_mask = is_bad_cycle[cycle_numbers]

        # Create output arrays for this channel
        channel_phase = signal_phase.copy()
        channel_phase[bad_sample_mask] = np.nan
        signal_phase_out[:, c] = channel_phase

        channel_cycles = cycle_numbers.astype(float)
        channel_cycles[bad_sample_mask] = np.nan
        cycle_numbers_out[:, c] = channel_cycles

    # Return with proper dimensionality
    if input_is_1d:
        return signal_phase_out.squeeze(), cycle_numbers_out.squeeze()
    else:
        return signal_phase_out, cycle_numbers_out


def get_spike_phase(
    lfp: np.ndarray,
    spike_times: np.ndarray,
    sampling_rate: float,
    peak_freq: Union[float, np.ndarray],
    clip_value: float = 0,
    filt_half_bandwidth: float = 2,
    power_thresh: float = 5,
    channel_idx: int = 0,
):
    """
    Calculate the signal phase of spikes based on the LFP.
    Works with both 1D and 2D LFP arrays.

    Parameters:
    - lfp: Local Field Potential time series (1D or 2D).
    - spike_times: Times (in seconds) at which the spikes occurred.
    - sampling_rate: The sampling rate of the LFP.
    - peak_freq: Central frequency or array of central frequencies.
    - clip_value: The value above which the LFP is considered clipped. Default is 0.
    - filt_half_bandwidth: Half bandwidth for filtering. Default is 2 Hz.
    - power_thresh: Threshold (percentile) for minimum power per cycle. Default is 5.
    - channel_idx: For 2D LFP, which channel to use (default 0).

    Returns:
    - spike_phases: phase (in radians) of spikes.
    """
    # Get phase for all channels
    phase, _ = get_signal_phase(
        lfp,
        sampling_rate,
        peak_freq,
        clip_value=clip_value,
        filt_half_bandwidth=filt_half_bandwidth,
        power_thresh=power_thresh,
    )

    # Handle dimensionality for indexing
    if phase.ndim == 1:
        selected_phase = phase
    else:
        if channel_idx >= phase.shape[1]:
            raise ValueError(
                f"""channel_idx {channel_idx} is out of bounds
                for LFP with {phase.shape[1]} channels"""
            )
        selected_phase = phase[:, channel_idx]

    # Convert spike times to indices
    spike_indices = np.floor(spike_times * sampling_rate).astype(np.int64)

    # Create output array with NaNs
    spike_phases = np.full_like(spike_times, np.nan, dtype=float)

    # Find valid indices
    valid_mask = (spike_indices >= 0) & (spike_indices < len(selected_phase))

    # Extract phases for valid spikes
    if np.any(valid_mask):
        valid_indices = spike_indices[valid_mask]
        spike_phases[valid_mask] = selected_phase[valid_indices]

    return spike_phases
