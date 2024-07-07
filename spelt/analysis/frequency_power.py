import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import morlet

def complex_morlet_wavelet_transform(signal, frequencies, fs):
    """
    Apply the complex Morlet wavelet transform to a signal for a range of frequencies
    using FFT for efficient convolution.

    This optimized version utilizes FFT-based convolution, which is significantly faster
    for large signals and frequency ranges, especially when the length of the signal is large.

    Parameters:
    - signal (numpy.ndarray): The input signal (time series).
    - frequencies (numpy.ndarray): Array of frequencies for which to compute the transform.
    - fs (float): Sampling frequency of the input signal.

    Returns:
    - numpy.ndarray: An array of wavelet coefficients, with dimensions (time x frequencies).
    """
    n = len(signal)
    signal_fft = np.fft.fft(signal)
    wavelet_coeffs = np.zeros((n, len(frequencies)), dtype=complex)

    for i, freq in enumerate(frequencies):
        # Generate the Morlet wavelet for the current frequency
        # 'complete' parameter is set to True for the complete Morlet wavelet
        wavelet = morlet(M=n, w=5, s=freq / fs, complete=True)

        # Compute the FFT of the Morlet wavelet
        wavelet_fft = np.fft.fft(wavelet, n)

        # Perform the convolution in the frequency domain by multiplying the FFTs
        # and then apply the inverse FFT to get back to the time domain
        wavelet_coeffs[:, i] = np.fft.ifft(signal_fft * wavelet_fft)

    return wavelet_coeffs

def calculate_morlet_df(arm_cycle_df, channel, lfp_sampling_rate, f_min, f_max, f_bins):
    '''
    Takes dataframe of LFP data with theta cycles and traversal indices
    FOR A GIVEN CHANNEL, calculates frequency power across wavelengths for each traversal individually and adds to the dataframe
    '''
    #Check channel is string
    if not isinstance(channel, str):
        channel = str(channel)

    frequencies = np.linspace(f_min, f_max, f_bins)
    channel_lfp = arm_cycle_df.loc(axis = 0)[channel]
    traversal_index = arm_cycle_df.loc(axis=0)['Traversal Index']

    # Get total number of traversals
    traversals = int(np.nanmax(traversal_index))

    # Initialise output dataframe
    morlet_df = pd.DataFrame(np.nan, index = frequencies, columns = arm_cycle_df.columns)

    # Loop through traversals and calculate complex Morlet wavelet transform for each channel
    for traversal in range(traversals):
        # Select traversal data from dataframe
        traversal_data = channel_lfp[traversal_index == traversal]
        traversal_timestamps = arm_cycle_df.columns[traversal_index == traversal]

        if traversal_data.size != 0:
            # print(channel, traversal, len(traversal_df.loc[channel]))
            # Calculate wavelet transform
            traversal_morlet = complex_morlet_wavelet_transform(signal = traversal_data,
                                   frequencies = frequencies,
                                   fs = lfp_sampling_rate)

            # Add to dataframe
            morlet_df.loc[frequencies, traversal_timestamps] = traversal_morlet.T

    # Add cycle index and traversal index back into morlet dataframe
    morlet_df.loc(axis = 0)['Traversal Index'] = arm_cycle_df.loc(axis = 0)['Traversal Index'].astype(int)
    morlet_df.loc(axis = 0)['Cycle Index'] = arm_cycle_df.loc(axis = 0)['Cycle Index'].astype(int)
    morlet_df.loc(axis = 0)['Cycle Theta Phase'] = arm_cycle_df.loc(axis = 0)['Cycle Theta Phase']
    morlet_df.loc(axis = 0)['Speed'] = arm_cycle_df.loc(axis = 0)['Speed']

    return morlet_df

def plot_wavelet_power_spectrum_theta(wavelet_coeffs, theta_phase, n_theta_bins, frequencies, arm, save_dir):
    """
    Plot a power spectrum across theta phase from the wavelet coefficients.

    :param wavelet_coeffs: Wavelet coefficients as returned by complex_morlet_wavelet_transform. 2D array of frequencies x timestamps
    :param theta_phase: 1D numpy array corresponding to theta phase for each timestamp
    :param n_theta_bins: number of theta bins to separate data into for plotting
    :param frequencies: Array of frequencies used in the wavelet transform
    """
    bin_edges = np.linspace(0, 2 * np.pi, n_theta_bins + 1)
    power_spectra = np.zeros((len(frequencies), n_theta_bins))

    for i in range(n_theta_bins):
        # Find indices for the current theta phase bin
        indices = np.where((theta_phase >= bin_edges[i]) & (theta_phase < bin_edges[i + 1]))[0]

        # Extract the wavelet coefficients for these indices
        selected_coeffs = wavelet_coeffs[indices, :]

        # Calculate the average power spectrum for this phase bin
        power_spectra[:, i] = np.mean(np.abs(selected_coeffs) ** 2, axis=0)

    # Smooth data across phase bins
    from scipy.ndimage import gaussian_filter1d
    # Apply this to each frequency bin
    sigma = 2  # Standard deviation for Gaussian kernel
    smoothed_power_spectra = np.array([gaussian_filter1d(power_spectra[i, :], sigma) for i in range(power_spectra.shape[0])])

    # Plotting
    theta_bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(theta_bin_centers, frequencies, smoothed_power_spectra, cmap='jet')
    plt.xlabel('Theta Phase [radians]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(label='Power')
    plt.title('Theta Phase vs Frequency Power Spectrum')

    # Adjust the x-axis to show multiples of pi
    x_ticks = np.linspace(0, 2 * np.pi, 5)  # 0, pi/2, pi, 3pi/2, 2pi
    x_labels = ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']
    plt.xticks(x_ticks, x_labels)

    # Overlay a theta cycle
    theta_cycle = np.cos(np.linspace(0, 2 * np.pi, 100))
    theta_phase_values = np.linspace(0, 2 * np.pi, 100)
    offset = np.median(frequencies)  # Adjust as per your plot
    theta_cycle_normalized = theta_cycle * np.ptp(frequencies)/4 + offset
    plt.plot(theta_phase_values, theta_cycle_normalized, color='white', linestyle='--', linewidth=2)
    plt.title(f'Frequency Spectrogram across theta phase - {arm} Arm')

    plt.savefig(save_dir)

    plt.show()
