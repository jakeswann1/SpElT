import numpy as np
from scipy.signal import welch


def find_peak_frequency(
    signal_data: np.ndarray,
    sampling_rate: float,
    freq_range: tuple[float, float],
    nperseg: int = 1024,
    average_channels: bool = False,
) -> np.ndarray:
    """
    Finds the peak power frequency within a specified frequency range for signals.

    Parameters:
    -----------
    signal_data : np.ndarray
        Array of signal data with shape (samples, channels).
    sampling_rate : float
        The sampling rate of the signals in Hz.
    freq_range : Tuple[float, float]
        The frequency range (min_freq, max_freq) to search within.
    nperseg : int, optional
        Length of each segment for Welch's method. Default is 1024.
    average_channels : bool, optional
        If True, returns a single peak frequency from the average power spectrum.
        If False, returns peak frequency for each channel. Default is False.

    Returns:
    --------
    np.ndarray
        Array containing the frequency with peak power in the specified range
        for each channel, or a single value if average_channels=True.
    """
    num_channels = signal_data.shape[1]

    # Initialize frequency values
    freq_values = None

    if average_channels:
        # Average signals across channels first, then calculate single PSD
        avg_signal = np.mean(signal_data, axis=1)
        freq_values, power_spectrum = welch(
            avg_signal, fs=sampling_rate, nperseg=nperseg
        )

        # Get indices for the requested frequency range
        range_mask = (freq_values >= freq_range[0]) & (freq_values <= freq_range[1])
        if not np.any(range_mask):
            # No frequencies in the specified range
            return np.array([np.nan])

        # Filter frequencies to the specified range
        range_freqs = freq_values[range_mask]
        range_power = power_spectrum[range_mask]

        return np.array([range_freqs[np.argmax(range_power)]])

    # For multiple channels, process in batches
    peak_freqs = np.zeros(num_channels)

    # Process channels in batches to avoid memory issues with very large arrays
    batch_size = min(100, num_channels)  # Adjust batch size based on available memory

    for batch_start in range(0, num_channels, batch_size):
        batch_end = min(batch_start + batch_size, num_channels)
        batch_channels = np.arange(batch_start, batch_end)

        # Calculate Welch's PSD for a batch of channels
        # This efficiently computes multiple channels at once
        freq_values, power_spectra = welch(
            signal_data[:, batch_channels], fs=sampling_rate, nperseg=nperseg, axis=0
        )

        # On first batch, set up the frequency masks
        if batch_start == 0:
            # Get indices for the requested frequency range
            range_mask = (freq_values >= freq_range[0]) & (freq_values <= freq_range[1])
            if not np.any(range_mask):
                # No frequencies in the specified range
                return np.full(num_channels, np.nan)

            # Filter frequencies to the specified range
            range_freqs = freq_values[range_mask]

        # Extract power values in the frequency range
        range_powers = power_spectra[range_mask, :]

        # Find indices of maximum power for each channel in the batch
        max_indices = np.argmax(range_powers, axis=0)

        # Map these indices back to the actual frequencies
        peak_freqs[batch_channels] = range_freqs[max_indices]

    return peak_freqs


def get_theta_frequencies(
    lfp_data: np.ndarray,
    sampling_rate: float,
    theta_range: tuple[float, float] = (4.0, 10.0),
    nperseg: int = 1024,
    average_channels: bool = False,
) -> np.ndarray:
    """
    Finds the peak power frequency in the theta range for signals.

    Parameters:
    -----------
    lfp_data : np.ndarray
        Array of local field potential signal data with shape (samples, channels).
    sampling_rate : float
        The sampling rate of the LFP traces in Hz.
    theta_range : Tuple[float, float], optional
        The frequency range defining theta. Default is (4.0, 10.0) Hz.
    nperseg : int, optional
        Length of each segment for Welch's method. Default is 1024.
    average_channels : bool, optional
        If True, returns a single peak frequency from the average power spectrum.
        If False, returns peak frequency for each channel. Default is False.

    Returns:
    --------
    np.ndarray
        Array containing the frequency with peak power in the theta range
        for each channel, or a single value if average_channels=True.
    """
    return find_peak_frequency(
        lfp_data,
        sampling_rate,
        freq_range=theta_range,
        nperseg=nperseg,
        average_channels=average_channels,
    )
