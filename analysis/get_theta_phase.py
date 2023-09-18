import numpy as np
from scipy.signal import firwin, filtfilt, hilbert

def get_theta_phase(lfp, spike_times, sampling_rate, peakFreq, filtHalfBandWidth=2, powerThresh=5):
    """
    Calculate the theta phase of spikes based on the LFP using the corrected bad data handling logic.

    Parameters:
    - lfp: Local Field Potential time series.
    - spike_times: Times (in seconds) at which the spikes occurred.
    - sampling_rate: The sampling rate of the LFP.
    - peakFreq: The central frequency around which the LFP is filtered.
    - filtHalfBandWidth: Half bandwidth for filtering. Default is 3 Hz.
    - powerThresh: Threshold (percentile) for minimum power per cycle. Default is 5.

    Returns:
    - spike_phases: Theta phase (in radians) of spikes.
    """

    # 1. FILTER THE LFP AROUND THE PEAK FREQUENCY
    low = (peakFreq - filtHalfBandWidth)
    high = (peakFreq + filtHalfBandWidth)
    taps = firwin(round(sampling_rate) + 1, [low, high], pass_zero=False, window='blackman', fs=sampling_rate)
    padlen = min(3 * (len(taps) - 1), len(lfp) - 1)
    theta_lfp = filtfilt(taps, 1, lfp, padlen=padlen)

    # 2. EXTRACT INSTANTANEOUS PHASE USING HILBERT TRANSFORM
    analytic_signal = hilbert(theta_lfp)
    eegPhase = np.angle(analytic_signal)
    eegPhase = np.mod(eegPhase, 2 * np.pi)

    # 3. IDENTIFY AND HANDLE PHASE TRANSITIONS AND SLIPS
    phaseTrans = np.diff(eegPhase) < -np.pi
    phaseTrans = np.hstack(([True], phaseTrans, [True]))
    phaseSlips = np.hstack(([False], np.diff(np.unwrap(eegPhase)) < 0, [False]))
    phaseTrans[phaseSlips] = False

    # 4. CALCULATE EEG POWER AND POWER PER CYCLE
    cycleN = np.cumsum(phaseTrans[:-1])
    power = theta_lfp**2
    powerPerCycle = np.bincount(cycleN, power) / np.bincount(cycleN)
    
    # Calculate cycle length
    cycleLength = np.bincount(cycleN)
    
    # 5. HANDLE BAD DATA
    # Power threshold
    thresh = np.nanpercentile(powerPerCycle, powerThresh)
    badPowerCycle = np.where(powerPerCycle < thresh)[0]
    badPowerInd = np.isin(cycleN, badPowerCycle)

    # Length threshold
    min_cycle_length = np.ceil(sampling_rate / (peakFreq + filtHalfBandWidth))
    max_cycle_length = np.ceil(sampling_rate / (peakFreq - filtHalfBandWidth))
    badLengthCycle = np.where((cycleLength < min_cycle_length) | (cycleLength > max_cycle_length))[0]
    badLengthInd = np.isin(cycleN, badLengthCycle)
    
    # Remove from data
    eegPhase[badLengthInd | badPowerInd] = np.nan
    
    # 6. MAP SPIKE TIMES TO PHASE
    spike_indices = (spike_times * sampling_rate).astype(int)

    # Initialize spike_phases array
    spike_phases = np.zeros_like(spike_times)

    # Handle spike times within the range of the LFP data
    in_bounds_indices = spike_indices < len(eegPhase)
    spike_phases[in_bounds_indices] = eegPhase[spike_indices[in_bounds_indices]]

    # Set the phase as np.nan for out-of-bounds spike times
    out_of_bounds_indices = ~in_bounds_indices
    spike_phases[out_of_bounds_indices] = np.nan


    return spike_phases