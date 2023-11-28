import numpy as np
from scipy.signal import firwin, filtfilt, hilbert

def get_theta_phase(lfp, sampling_rate, peak_freq, filt_half_bandwidth=2, power_thresh=5):
    """
    Simplified function to calculate the theta phase timeseries from an LFP signal.

    Parameters:
    - lfp: Local Field Potential time series.
    - sampling_rate: The sampling rate of the LFP.
    - peak_freq: The central frequency around which the LFP is filtered.
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
    theta_phase = np.mod(theta_phase, 2 * np.pi)  # Adjust the phase here

    # Identify and handle phase transitions and slips
    phase_transitions = np.diff(theta_phase) < -np.pi
    phase_transitions = np.hstack(([False], phase_transitions, [False]))
    cycle_numbers = np.cumsum(phase_transitions[:-1])

    # Calculate power per cycle
    power = filtered_lfp ** 2
    power_per_cycle = np.bincount(cycle_numbers, power) / np.bincount(cycle_numbers)

    # Handle low power cycles
    power_threshold = np.nanpercentile(power_per_cycle, power_thresh)
    bad_power_cycles = np.where(power_per_cycle < power_threshold)[0]
    bad_power_indices = np.isin(cycle_numbers, bad_power_cycles)

    # Set low power cycles to NaN
    theta_phase[bad_power_indices] = np.nan

    return theta_phase, cycle_numbers