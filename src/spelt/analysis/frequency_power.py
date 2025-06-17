import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt


def complex_morlet_wavelet_transform(
    signal: np.ndarray, frequencies: np.ndarray, fs: float
) -> np.ndarray:
    """
    Apply the complex Morlet wavelet transform to a signal for a range of frequencies
    using PyWavelets for efficient computation.

    Parameters:
    - signal (numpy.ndarray): The input signal (time series).
    - frequencies (numpy.ndarray): Array of frequencies to compute the transform over.
    - fs (float): Sampling frequency of the input signal.

    Returns:
    - numpy.ndarray: An array of wavelet coefficients, with shape (frequencies x time).
    """
    # Convert frequencies to scales for the wavelet transform
    # For a complex Morlet wavelet with center frequency 1.0 and bandwidth 1.5
    scales = pywt.frequency2scale("cmor1.5-1.0", frequencies / fs)

    # Apply the continuous wavelet transform
    # By default, the method='fft' is used for efficient computation
    coefs, _ = pywt.cwt(signal, scales, "cmor1.5-1.0", sampling_period=1.0 / fs)

    return coefs


def calculate_morlet_df(
    arm_cycle_df: pd.DataFrame,
    channel: str,
    lfp_sampling_rate: float,
    f_min: float,
    f_max: float,
    f_bins: int,
) -> pd.DataFrame:
    """
    Takes dataframe of LFP data with theta cycles and traversal indices
    FOR A GIVEN CHANNEL, calculates frequency power across wavelengths
    for each traversal individually and adds to the dataframe.

    Parameters:
    - arm_cycle_df (pandas.DataFrame): DataFrame containing LFP data
    - channel (str): Channel name to process
    - lfp_sampling_rate (float): Sampling rate of the LFP data in Hz
    - f_min (float): Minimum frequency for analysis in Hz
    - f_max (float): Maximum frequency for analysis in Hz
    - f_bins (int): Number of frequency bins between f_min and f_max

    Returns:
    - pandas.DataFrame: DataFrame with wavelet coefficients for each frequency bin
    """
    # Check channel is string
    if not isinstance(channel, str):
        channel = str(channel)

    frequencies = np.linspace(f_min, f_max, f_bins)
    channel_lfp = arm_cycle_df.loc(axis=0)[channel]
    traversal_index = arm_cycle_df.loc(axis=0)["Traversal Index"]

    # Get total number of traversals
    traversals = int(np.nanmax(traversal_index))

    # Initialize output dataframe
    morlet_df = pd.DataFrame(np.nan, index=frequencies, columns=arm_cycle_df.columns)

    # Loop through traversals and calculate wavelet transform for each channel
    for traversal in range(traversals):
        # Select traversal data from dataframe
        traversal_data = channel_lfp[traversal_index == traversal]
        traversal_timestamps = arm_cycle_df.columns[traversal_index == traversal]

        if traversal_data.size != 0:
            # Calculate wavelet transform using updated function
            traversal_morlet = complex_morlet_wavelet_transform(
                signal=traversal_data, frequencies=frequencies, fs=lfp_sampling_rate
            )

            # Add to dataframe
            morlet_df.loc[frequencies, traversal_timestamps] = traversal_morlet

    # Add cycle index and traversal index back into morlet dataframe
    morlet_df.loc(axis=0)["Traversal Index"] = arm_cycle_df.loc(axis=0)[
        "Traversal Index"
    ].astype(int)
    morlet_df.loc(axis=0)["Cycle Index"] = arm_cycle_df.loc(axis=0)[
        "Cycle Index"
    ].astype(int)
    morlet_df.loc(axis=0)["Cycle Theta Phase"] = arm_cycle_df.loc(axis=0)[
        "Cycle Theta Phase"
    ]
    morlet_df.loc(axis=0)["Speed"] = arm_cycle_df.loc(axis=0)["Speed"]

    return morlet_df


def plot_wavelet_power_spectrum_theta(
    wavelet_coeffs: np.ndarray,
    theta_phase: np.ndarray,
    n_theta_bins: int,
    frequencies: np.ndarray,
    arm: str,
    save_dir: str,
) -> None:
    """
    Plot a power spectrum across theta phase from the wavelet coefficients.

    Parameters:
    - wavelet_coeffs: Wavelet coefficients returned by complex_morlet_wavelet_transform.
                      2D array of frequencies x timestamps
    - theta_phase: 1D numpy array corresponding to theta phase for each timestamp
    - n_theta_bins: number of theta bins to separate data into for plotting
    - frequencies: Array of frequencies used in the wavelet transform
    - arm: String label for the arm (used in plot title)
    - save_dir: Directory path where to save the plot
    """
    bin_edges = np.linspace(0, 2 * np.pi, n_theta_bins + 1)
    power_spectra = np.zeros((len(frequencies), n_theta_bins))

    for i in range(n_theta_bins):
        # Find indices for the current theta phase bin
        indices = np.where(
            (theta_phase >= bin_edges[i]) & (theta_phase < bin_edges[i + 1])
        )[0]

        # Extract the wavelet coefficients for these indices
        selected_coeffs = wavelet_coeffs[indices, :]

        # Calculate the average power spectrum for this phase bin
        power_spectra[:, i] = np.mean(np.abs(selected_coeffs) ** 2, axis=0)

    # Smooth data across phase bins
    from scipy.ndimage import gaussian_filter1d

    # Apply this to each frequency bin
    sigma = 2  # Standard deviation for Gaussian kernel
    smoothed_power_spectra = np.array(
        [
            gaussian_filter1d(power_spectra[i, :], sigma)
            for i in range(power_spectra.shape[0])
        ]
    )

    # Plotting
    theta_bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(theta_bin_centers, frequencies, smoothed_power_spectra, cmap="jet")
    plt.xlabel("Theta Phase [radians]")
    plt.ylabel("Frequency [Hz]")
    plt.colorbar(label="Power")
    plt.title("Theta Phase vs Frequency Power Spectrum")

    # Adjust the x-axis to show multiples of pi
    x_ticks = np.linspace(0, 2 * np.pi, 5)  # 0, pi/2, pi, 3pi/2, 2pi
    x_labels = ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
    plt.xticks(x_ticks, x_labels)

    # Overlay a theta cycle
    theta_cycle = np.cos(np.linspace(0, 2 * np.pi, 100))
    theta_phase_values = np.linspace(0, 2 * np.pi, 100)
    offset = np.median(frequencies)  # Adjust as per your plot
    theta_cycle_normalized = theta_cycle * np.ptp(frequencies) / 4 + offset
    plt.plot(
        theta_phase_values,
        theta_cycle_normalized,
        color="white",
        linestyle="--",
        linewidth=2,
    )
    plt.title(f"Frequency Spectrogram across theta phase - {arm} Arm")

    plt.savefig(save_dir)
    plt.show()


def plot_spectrogram(
    data: np.ndarray,
    fs: float,
    start_time: float = None,
    stop_time: float = None,
    n_windows: int = 10,
    f_min: float = None,
    f_max: float = None,
    window: str = "hann",
    nperseg: int = 1024,
    noverlap: int = None,
    scaling: str = "density",
    db_scale: bool = True,
    cmap: str = "viridis",
    fig=None,
    title: str = None,
    vmin: float = None,
    vmax: float = None,
):
    """
    Compute and plot power spectra across time windows.

    Parameters:
    -----------
    data : numpy.ndarray
        Input time series data
    fs : float
        Sampling frequency in Hz
    start_time : float, optional
        Start time in seconds (default: 0)
    stop_time : float, optional
        Stop time in seconds (default: len(data)/fs)
    n_windows : int, optional
        Number of time windows to analyze (default: 10)
    f_min : float, optional
        Minimum frequency to plot (Hz)
    f_max : float, optional
        Maximum frequency to plot (Hz)
    window : str or tuple, optional
        Desired window to use for spectral analysis (default: 'hann')
    nperseg : int, optional
        Length of each segment for Welch's method (default: 1024)
    noverlap : int, optional
        Number of points to overlap between segments (default: nperseg//2)
    scaling : {'density', 'spectrum'}, optional
        Whether to scale the power spectrum by the sampling frequency
    db_scale : bool, optional
        Whether to plot power in decibels (dB)
    cmap : str, optional
        Colormap for the spectrogram (default: 'viridis')
    fig : matplotlib.figure.Figure, optional
        Figure object to plot on. If None, creates a new figure
    title : str, optional
        Title for the plot
    vmin : float, optional
        Minimum value for the color scale
    vmax : float, optional
        Maximum value for the color scale

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plots
    ax1, ax2 : matplotlib.axes.Axes
        The axes objects containing the plots
    """
    from scipy.signal import welch

    # Set default time range if not provided
    if start_time is None:
        start_time = 0
    if stop_time is None:
        stop_time = len(data) / fs

    # Convert times to sample indices
    start_idx = int(start_time * fs)
    stop_idx = int(stop_time * fs)
    data = data[start_idx:stop_idx]

    # Create time windows
    window_size = (stop_idx - start_idx) // n_windows
    times = np.linspace(start_time, stop_time, n_windows)

    # Initialize arrays to store power spectra
    freqs, temp_psd = welch(
        data[:window_size],
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling=scaling,
    )

    # Set frequency range
    if f_min is None:
        f_min = freqs[1]  # Skip DC component
    if f_max is None:
        f_max = freqs[-1]
    freq_mask = (freqs >= f_min) & (freqs <= f_max)
    freqs = freqs[freq_mask]

    # Initialize array to store all PSDs
    all_psds = np.zeros((n_windows, len(freqs)))

    # Compute power spectrum for each window
    for i in range(n_windows):
        start_idx = i * window_size
        stop_idx = (i + 1) * window_size
        window_data = data[start_idx:stop_idx]

        _, psd = welch(
            window_data,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling=scaling,
        )

        if db_scale:
            psd = 10 * np.log10(psd)

        all_psds[i] = psd[freq_mask]

    # Create figure with two subplots
    if fig is None:
        fig = plt.figure(figsize=(12, 8))

    # Plot spectrogram
    ax1 = fig.add_subplot(211)
    im = ax1.pcolormesh(
        times, freqs, all_psds.T, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax
    )
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_xlabel("Time (s)")
    plt.colorbar(im, ax=ax1, label="Power" + " (dB)" if db_scale else "")

    # Plot mean power spectrum
    ax2 = fig.add_subplot(212)
    mean_psd = np.mean(all_psds, axis=0)
    ax2.plot(freqs, mean_psd)
    ax2.set_xlabel("Frequency (Hz)")
    if db_scale:
        ax2.set_ylabel(
            "Power Spectral Density (dB/Hz)"
            if scaling == "density"
            else "Power Spectrum (dB)"
        )
    else:
        ax2.set_ylabel(
            "Power Spectral Density (V²/Hz)"
            if scaling == "density"
            else "Power Spectrum (V²)"
        )
    ax2.grid(True)

    # Set overall title
    if title:
        fig.suptitle(title)

    plt.tight_layout()

    return fig, (ax1, ax2)


def compute_power_by_cycle(
    signal: np.ndarray,
    cycle_numbers: np.ndarray,
    lfp_sampling_rate: float,
    f_min: float,
    f_max: float,
    f_bins: int = 100,
    channel_axis: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute power spectrum for each cycle in a given frequency band using vectorized operations.

    Parameters:
    -----------
    signal : np.ndarray
        Array of signal data. Can be 1D (samples) or 2D (samples, channels)
    cycle_numbers : np.ndarray
        Array of cycle numbers for each sample
    lfp_sampling_rate : float
        Sampling rate of the signal in Hz
    f_min : float
        Minimum frequency for analysis
    f_max : float
        Maximum frequency for analysis
    f_bins : int, optional
        Number of frequency bins between f_min and f_max (default: 100)
    channel_axis : int, optional
        Axis containing channel data if signal is multi-channel (default: -1)

    Returns:
    --------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - Array of power spectra for each cycle (cycles x frequencies)
        - Array of cycle numbers
        - Array of frequencies
    """
    # Generate frequency bins
    frequencies = np.linspace(f_min, f_max, f_bins)

    # Get unique cycles, excluding NaN
    unique_cycles = np.unique(cycle_numbers[~np.isnan(cycle_numbers)])

    # Compute wavelet transform for the entire signal
    wavelet_coeffs = complex_morlet_wavelet_transform(
        signal, frequencies, lfp_sampling_rate
    )

    # Compute power (magnitude squared) for all time points
    power = np.abs(wavelet_coeffs) ** 2

    # Create a mask array for each cycle
    cycle_masks = np.array([cycle_numbers == cycle for cycle in unique_cycles])

    # Handle multi-channel data if present
    if len(power.shape) == 3:  # If we have multiple channels
        power = power.mean(axis=channel_axis)  # Average across channels

    # Transpose power to (time, frequencies) to match cycle_masks dimension
    power = power.T

    # For each cycle mask, compute mean power across time points for each frequency
    cycle_powers = np.array([np.mean(power[mask], axis=0) for mask in cycle_masks])

    return cycle_powers, unique_cycles, frequencies
