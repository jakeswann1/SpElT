import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert

from .filtering import bandpass_filter_lfp


def get_signal_phase(
    lfp: np.ndarray,
    sampling_rate: float,
    peak_freq: float | np.ndarray | None = None,
    clip_value: float = 0,
    filt_half_bandwidth: float = 2,
    power_thresh: float = 5,
    cycle_start: str = "peak",
    return_inst_freq: bool = False,
    bandpass_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Efficiently calculates the signal phase timeseries from an LFP signal.
    Vectorized implementation that handles both 1D and 2D inputs.
    Supports channel-specific peak frequencies or direct bandpass filtering.

    Pi radians is the signal trough. 0 (and 2pi) radians is the signal peak.
    pi/2 is the descending zero crossing, and 3pi/2 is the ascending zero crossing.

    Parameters:
    -----------
    lfp : np.ndarray
        Local Field Potential time series.
        Can be 1D (samples,) or 2D (samples, channels).
    sampling_rate : float
        The sampling rate of the LFP.
    peak_freq : float or np.ndarray, optional
        The central frequency around which the LFP is filtered.
        Can be a single value for all channels or an array with one value per channel.
        Mutually exclusive with bandpass_range. Default is None.
    clip_value : float, optional
        The value above which the LFP is considered clipped. Default is 0.
    filt_half_bandwidth : float, optional
        Half bandwidth for filtering (only used with peak_freq). Default is 2 Hz.
    power_thresh : float, optional
        Threshold (percentile) for minimum power per cycle. Default is 5.
    cycle_start : str, optional
        Where to start counting cycles from.
        Either 'peak' (0 radians) or 'trough' (pi radians). Default is 'peak'.
    return_inst_freq : bool, optional
        If True, also return instantaneous frequency. Default is False.
    bandpass_range : tuple[float, float], optional
        Direct bandpass filter range as (low_freq, high_freq).
        Mutually exclusive with peak_freq. Default is None.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray] or Tuple[np.ndarray, np.ndarray, np.ndarray]
        - signal_phase: Signal phase (in radians) timeseries,
        with low power cycles set to NaN. Same shape as input lfp.
        - cycle_numbers: Cycle number for each sample, with bad cycles set to NaN.
          Same shape as input lfp.
        - inst_freq: Instantaneous frequency timeseries (if return_inst_freq=True).
          Same shape as input lfp.
    """
    if cycle_start not in ["peak", "trough"]:
        raise ValueError("cycle_start must be either 'peak' or 'trough'")

    # Validate filtering parameters
    if peak_freq is None and bandpass_range is None:
        raise ValueError("Either peak_freq or bandpass_range must be provided")
    if peak_freq is not None and bandpass_range is not None:
        raise ValueError("peak_freq and bandpass_range are mutually exclusive")

    # Handle both 1D and 2D inputs
    input_is_1d = lfp.ndim == 1
    if input_is_1d:
        lfp = lfp.reshape(-1, 1)

    n_samples, n_channels = lfp.shape

    # Handle filtering mode
    if peak_freq is not None:
        # Original peak_freq mode
        if isinstance(peak_freq, (int, float)):
            peak_freqs = np.full(n_channels, peak_freq)
        else:
            peak_freqs = np.asarray(peak_freq)
            if len(peak_freqs) != n_channels:
                raise ValueError(
                    f"""Length of peak_freq ({len(peak_freqs)})
                    must match number of channels ({n_channels})"""
                )
        filter_ranges = [
            (pf - filt_half_bandwidth, pf + filt_half_bandwidth) for pf in peak_freqs
        ]
    else:
        # Direct bandpass mode
        low_freq, high_freq = bandpass_range
        filter_ranges = [(low_freq, high_freq)] * n_channels

    # Pre-allocate output arrays
    signal_phase_out = np.zeros((n_samples, n_channels))
    cycle_numbers_out = np.zeros((n_samples, n_channels), dtype=float)
    if return_inst_freq:
        inst_freq_out = np.zeros((n_samples, n_channels))

    # Process each channel with its specific filter range
    for c in range(n_channels):
        # Get filter parameters for this channel
        low_freq, high_freq = filter_ranges[c]

        # Filter LFP for this channel using the shared filtering function
        filtered_lfp = bandpass_filter_lfp(
            lfp[:, c],
            fs=sampling_rate,
            freq_min=low_freq,
            freq_max=high_freq,
            filter_type="fir",
        )

        # Hilbert transform
        analytic_signal = hilbert(filtered_lfp)

        # Extract phase
        signal_phase = np.angle(analytic_signal) % (2 * np.pi)

        # Calculate instantaneous frequency if requested
        if return_inst_freq:
            # Unwrap phase for frequency calculation
            unwrapped_phase = np.unwrap(np.angle(analytic_signal))

            smoothed_phase = gaussian_filter1d(unwrapped_phase, sigma=2)
            inst_freq = np.gradient(smoothed_phase) * sampling_rate / (2 * np.pi)
            # Limit to filter range
            inst_freq[(inst_freq < low_freq) | (inst_freq > high_freq)] = np.nan

        # Identify phase transitions efficiently
        phase_diff = np.diff(signal_phase)
        phase_transitions = np.zeros(n_samples, dtype=bool)

        # Set transition point based on cycle_start parameter
        if cycle_start == "peak":
            phase_transitions[1:] = phase_diff < -np.pi
        else:  # cycle_start == "trough"
            phase_transitions[1:] = np.abs(phase_diff) > np.pi

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

        if return_inst_freq:
            channel_inst_freq = inst_freq.copy()
            channel_inst_freq[bad_sample_mask] = np.nan
            inst_freq_out[:, c] = channel_inst_freq

    # Return with proper dimensionality
    if input_is_1d:
        if return_inst_freq:
            return (
                signal_phase_out.squeeze(),
                cycle_numbers_out.squeeze(),
                inst_freq_out.squeeze(),
            )
        else:
            return signal_phase_out.squeeze(), cycle_numbers_out.squeeze()
    else:
        if return_inst_freq:
            return signal_phase_out, cycle_numbers_out, inst_freq_out
        else:
            return signal_phase_out, cycle_numbers_out


def get_spike_phase(
    lfp: np.ndarray,
    spike_times: np.ndarray,
    sampling_rate: float,
    peak_freq: float | np.ndarray | None = None,
    clip_value: float = 0,
    filt_half_bandwidth: float = 2,
    power_thresh: float = 5,
    channel_idx: int = 0,
    cycle_start: str = "peak",
    bandpass_range: tuple[float, float] | None = None,
    return_inst_freq: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Calculate the signal phase of spikes based on the LFP.
    Works with both 1D and 2D LFP arrays.

    Parameters:
    -----------
    lfp : np.ndarray
        Local Field Potential time series (1D or 2D).
    spike_times : np.ndarray
        Times (in seconds) at which the spikes occurred.
    sampling_rate : float
        The sampling rate of the LFP.
    peak_freq : float or np.ndarray, optional
        Central frequency or array of central frequencies.
        Mutually exclusive with bandpass_range. Default is None.
    clip_value : float, optional
        The value above which the LFP is considered clipped. Default is 0.
    filt_half_bandwidth : float, optional
        Half bandwidth for filtering (only used with peak_freq). Default is 2 Hz.
    power_thresh : float, optional
        Threshold (percentile) for minimum power per cycle. Default is 5.
    channel_idx : int, optional
        For 2D LFP, which channel to use (default 0).
    cycle_start : str, optional
        Where to start counting cycles from.
        Either 'peak' (0 radians) or 'trough' (pi radians). Default is 'peak'.
    bandpass_range : tuple[float, float], optional
        Direct bandpass filter range as (low_freq, high_freq).
        Mutually exclusive with peak_freq. Default is None.
    return_inst_freq : bool, optional
        If True, also return instantaneous frequency at spike times. Default is False.

    Returns:
    --------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        spike_phases: phase (in radians) of spikes.
        spike_inst_freq: instantaneous frequency (in Hz) at spike times
            (if return_inst_freq=True).
    """
    # Get phase (and optionally inst_freq) for all channels
    if return_inst_freq:
        phase, _, inst_freq = get_signal_phase(
            lfp,
            sampling_rate,
            peak_freq=peak_freq,
            clip_value=clip_value,
            filt_half_bandwidth=filt_half_bandwidth,
            power_thresh=power_thresh,
            cycle_start=cycle_start,
            bandpass_range=bandpass_range,
            return_inst_freq=True,
        )
    else:
        phase, _ = get_signal_phase(
            lfp,
            sampling_rate,
            peak_freq=peak_freq,
            clip_value=clip_value,
            filt_half_bandwidth=filt_half_bandwidth,
            power_thresh=power_thresh,
            cycle_start=cycle_start,
            bandpass_range=bandpass_range,
        )

    # Handle dimensionality for indexing
    if phase.ndim == 1:
        selected_phase = phase
        if return_inst_freq:
            selected_inst_freq = inst_freq
    else:
        if channel_idx >= phase.shape[1]:
            raise ValueError(
                f"""channel_idx {channel_idx} is out of bounds
                for LFP with {phase.shape[1]} channels"""
            )
        selected_phase = phase[:, channel_idx]
        if return_inst_freq:
            selected_inst_freq = inst_freq[:, channel_idx]

    # Convert spike times to indices
    spike_indices = np.floor(spike_times * sampling_rate).astype(np.int64)

    # Create output arrays with NaNs
    spike_phases = np.full_like(spike_times, np.nan, dtype=float)
    if return_inst_freq:
        spike_inst_freq = np.full_like(spike_times, np.nan, dtype=float)

    # Find valid indices
    valid_mask = (spike_indices >= 0) & (spike_indices < len(selected_phase))

    # Extract phases and frequencies for valid spikes
    if np.any(valid_mask):
        valid_indices = spike_indices[valid_mask]
        spike_phases[valid_mask] = selected_phase[valid_indices]
        if return_inst_freq:
            spike_inst_freq[valid_mask] = selected_inst_freq[valid_indices]

    if return_inst_freq:
        return spike_phases, spike_inst_freq
    else:
        return spike_phases


## Sample code to display sine wave - phase relationship
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
# from spelt.analysis.get_signal_phase import get_signal_phase

# # Generate synthetic sine signal
# fs = 1000  # Hz
# t = np.linspace(0, 1, fs, endpoint=False)
# freq = 8  # Hz
# signal = np.sin(2 * np.pi * freq * t)

# # Call the function
# signal_phase, _ = get_signal_phase(signal, sampling_rate=fs, peak_freq=freq)

# # Find peak and trough indices
# peaks, _ = find_peaks(signal)
# troughs, _ = find_peaks(-signal)

# # Plotting
# plt.figure(figsize=(10, 4))
# plt.plot(t, signal, label="Signal")
# plt.plot(t, signal_phase, label="Phase (cycles)", alpha=0.7)
# plt.scatter(t[peaks], signal[peaks], color="red", label="Peaks", zorder=5)
# plt.scatter(t[troughs], signal[troughs], color="blue", label="Troughs", zorder=5)
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude / Phase (cycles)")
# plt.title("Signal and Instantaneous Phase")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Print average phase at peaks and troughs
# print(f"Average phase at peaks: {np.mean(signal_phase[peaks]):.2f} rad")
# print(f"Average phase at troughs: {np.mean(signal_phase[troughs]):.2f} rad")
