import numpy as np
from scipy.signal import firwin, filtfilt, hilbert

def get_theta_phase(lfp, sampling_rate, peak_freq, clip_value = 0, filt_half_bandwidth=2, power_thresh=5):
    """
    Simplified function to calculate the theta phase timeseries from an LFP signal.

    Parameters:
    - lfp: Local Field Potential time series.
    - sampling_rate: The sampling rate of the LFP.
    - peak_freq: The central frequency around which the LFP is filtered.
    - clip_value: The value above which the LFP is considered clipped. Default is 0.
    - filt_half_bandwidth: Half bandwidth for filtering. Default is 2 Hz.
    - power_thresh: Threshold (percentile) for minimum power per cycle. Default is 5.

    Returns:
    - theta_phase: Theta phase (in radians) timeseries, with low power cycles set to NaN.
    """

    # Filter the LFP around the peak frequency
    low_freq = peak_freq - filt_half_bandwidth
    high_freq = peak_freq + filt_half_bandwidth
    filter_taps = firwin(round(sampling_rate) + 1, [low_freq, high_freq], pass_zero=False, window='blackman', fs=sampling_rate)
    pad_length = min(3 * (len(filter_taps) - 1), len(lfp) - 1)
    filtered_lfp = filtfilt(filter_taps, 1, lfp, padlen=pad_length)

    # Extract instantaneous phase using the Hilbert transform
    analytic_signal = hilbert(filtered_lfp)
    theta_phase = np.angle(analytic_signal)
    # Wrap into the range 0 - 2pi, so that oscillation starts at the peak (convention).
    theta_phase %= 2 * np.pi

    # Identify and handle phase transitions and slips
    phase_transitions = np.diff(theta_phase) < -np.pi
    phase_transitions = np.hstack(([False], phase_transitions, [False]))
    cycle_numbers = np.cumsum(phase_transitions[:-1])

    ### Remove bad data
    # Remove any cycles with clipped values above or below 32000
    clipped_cycles = np.where(np.abs(lfp) > clip_value)[0]

    # Calculate bincount
    bincount_cycle_numbers = np.bincount(cycle_numbers)

    # Calculate power per cycle and identify bad power cycles
    power = filtered_lfp ** 2
    power_per_cycle = np.bincount(cycle_numbers, power) / bincount_cycle_numbers
    power_threshold = np.nanpercentile(power_per_cycle, power_thresh)
    bad_power_cycles = power_per_cycle < power_threshold

    # Calculate cycle lengths and identify bad length cycles
    cycle_lengths = bincount_cycle_numbers
    pass_band = np.array([peak_freq - filt_half_bandwidth, peak_freq + filt_half_bandwidth])
    cycle_length_lim = np.ceil(1 / pass_band * sampling_rate).astype(int)
    bad_length_cycles = (cycle_lengths < cycle_length_lim[1]) | (cycle_lengths > cycle_length_lim[0])

    # Combine bad power and length cycles
    bad_cycles = np.union1d(bad_power_cycles, bad_length_cycles)
    bad_cycles = np.union1d(bad_cycles, clipped_cycles)

    # Mark theta phase and cycle numbers corresponding to bad cycles as NaN
    bad_cycle_indices = np.isin(cycle_numbers, bad_cycles)
    theta_phase[bad_cycle_indices] = np.nan
    cycle_numbers = cycle_numbers.astype(float) # Convert to float to allow incluion of NaN
    cycle_numbers[bad_cycle_indices] = np.nan

    return theta_phase, cycle_numbers

def get_spike_theta_phase(lfp, spike_times, sampling_rate, peak_freq, clip_value = 0, filt_half_bandwidth=2, power_thresh=5):
    """
    Calculate the theta phase of spikes based on the LFP.

    Parameters:
    - lfp: Local Field Potential time series.
    - spike_times: Times (in seconds) at which the spikes occurred.
    - sampling_rate: The sampling rate of the LFP.
    - peak_freq: The central frequency around which the LFP is filtered.
    - filt_half_bandwidth: Half bandwidth for filtering. Default is 2 Hz.
    - power_thresh: Threshold (percentile) for minimum power per cycle. Default is 5.

    Returns:
    - spike_phases: Theta phase (in radians) of spikes.
    """

    theta_phase, _ = get_theta_phase(lfp, sampling_rate, peak_freq, clip_value = clip_value, filt_half_bandwidth = filt_half_bandwidth, power_thresh = power_thresh)

    # MAP SPIKE TIMES TO PHASE
    spike_indices = (spike_times * sampling_rate).astype(int)

    # Initialize spike_phases array
    spike_phases = np.zeros_like(spike_times)

    # Handle spike times within the range of the LFP data
    in_bounds_indices = spike_indices < len(theta_phase)
    spike_phases[in_bounds_indices] = theta_phase[spike_indices[in_bounds_indices]]

    # Set the phase as np.nan for out-of-bounds spike times
    out_of_bounds_indices = ~in_bounds_indices
    spike_phases[out_of_bounds_indices] = np.nan

    return spike_phases
