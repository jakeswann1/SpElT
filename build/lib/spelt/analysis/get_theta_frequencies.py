import numpy as np
from scipy.signal import welch

def get_theta_frequencies(lfp_data, sampling_rate):
    """
    Finds the peak power frequency in the theta range (4-10 Hz) for each channel in a 2D LFP array using Welch's method.

    Parameters:
    - LFP_2D_array (numpy array): The 2D array of local field potential signal traces, shape = (samples, channels).
    - sampling_rate (float): The sampling rate of the LFP traces in Hz.

    Returns:
    - numpy array: An array containing the frequency with peak power in the 4-10 Hz range for each channel.
    """

    num_channels = lfp_data.shape[1]
    peak_theta_freqs = np.zeros(num_channels)

    for i in range(num_channels):
        # Extract the LFP trace for the current channel
        lfp_trace = lfp_data[:, i]

        # Compute the power spectral density using Welch's method
        freq_values, power_spectrum = welch(lfp_trace, fs=sampling_rate, nperseg=1024)

        # Filter frequencies to only include those in the 4-10 Hz range
        theta_indices = np.where((freq_values >= 4) & (freq_values <= 10))
        theta_freqs = freq_values[theta_indices]
        theta_power = power_spectrum[theta_indices]

        # Find the frequency with the peak power in the theta range
        peak_theta_freqs[i] = theta_freqs[np.argmax(theta_power)]

    return peak_theta_freqs
