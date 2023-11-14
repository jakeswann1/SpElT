import numpy as np
from scipy.signal import firwin, filtfilt, hilbert

def get_theta_phase(lfp, spike_times, sampling_rate, peak_freq, filt_half_bandwidth=2, power_thresh=5):
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

    # 1. FILTER THE LFP AROUND THE PEAK FREQUENCY
    low_freq = peak_freq - filt_half_bandwidth
    high_freq = peak_freq + filt_half_bandwidth
    filter_taps = firwin(round(sampling_rate) + 1, [low_freq, high_freq], pass_zero=False, window='blackman', fs=sampling_rate)
    pad_length = min(3 * (len(filter_taps) - 1), len(lfp) - 1)
    filtered_lfp = filtfilt(filter_taps, 1, lfp, padlen=pad_length)

    # 2. EXTRACT INSTANTANEOUS PHASE USING HILBERT TRANSFORM
    analytic_signal = hilbert(filtered_lfp)
    lfp_phase = np.angle(analytic_signal)
    lfp_phase = np.mod(lfp_phase, 2 * np.pi)

    # 3. IDENTIFY AND HANDLE PHASE TRANSITIONS AND SLIPS
    phase_transitions = np.diff(lfp_phase) < -np.pi
    phase_transitions = np.hstack(([True], phase_transitions, [True]))
    phase_slips = np.hstack(([False], np.diff(np.unwrap(lfp_phase)) < 0, [False]))
    phase_transitions[phase_slips] = False

    # 4. CALCULATE POWER AND POWER PER CYCLE
    cycle_numbers = np.cumsum(phase_transitions[:-1])
    power = filtered_lfp ** 2
    power_per_cycle = np.bincount(cycle_numbers, power) / np.bincount(cycle_numbers)
    
    # Calculate cycle length
    cycle_length = np.bincount(cycle_numbers)
    
    # 5. HANDLE BAD DATA
    # Power threshold
    power_threshold = np.nanpercentile(power_per_cycle, power_thresh)
    bad_power_cycles = np.where(power_per_cycle < power_threshold)[0]
    bad_power_indices = np.isin(cycle_numbers, bad_power_cycles)

    # Length threshold
    min_cycle_length = np.ceil(sampling_rate / (peak_freq + filt_half_bandwidth))
    max_cycle_length = np.ceil(sampling_rate / (peak_freq - filt_half_bandwidth))
    bad_length_cycles = np.where((cycle_length < min_cycle_length) | (cycle_length > max_cycle_length))[0]
    bad_length_indices = np.isin(cycle_numbers, bad_length_cycles)
    
    # Remove bad data from phase information
    lfp_phase[bad_length_indices | bad_power_indices] = np.nan
    
    # 6. MAP SPIKE TIMES TO PHASE
    spike_indices = (spike_times * sampling_rate).astype(int)

    # Initialize spike_phases array
    spike_phases = np.zeros_like(spike_times)

    # Handle spike times within the range of the LFP data
    in_bounds_indices = spike_indices < len(lfp_phase)
    spike_phases[in_bounds_indices] = lfp_phase[spike_indices[in_bounds_indices]]

    # Set the phase as np.nan for out-of-bounds spike times
    out_of_bounds_indices = ~in_bounds_indices
    spike_phases[out_of_bounds_indices] = np.nan

    return spike_phases
