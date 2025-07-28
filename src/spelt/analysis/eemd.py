import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

import spelt.analysis.emdlib.emdlib as emd


def eemd(
    signal: np.ndarray,
    n_workers: int,
    n_ensembles: int,
    noise_std: float = 0.3,
    num_shifts: int = 10,
) -> np.ndarray:
    """
    Perform EEMD on multi-channel signal.

    Args:
        signal: 2D array of shape (samples, channels)
        n_workers: Number of parallel workers
        n_ensembles: Number of ensemble realizations
        noise_std: Standard deviation of added noise
        num_shifts: Number of shifts for sifting

    Returns:
        IMFs array of shape (n_imfs, samples, channels)
    """
    n_ensembles_per_worker = int(np.ceil(n_ensembles / n_workers))

    emd_queue = mp.Queue()
    processes = [
        mp.Process(
            target=_emd_worker,
            args=(signal, n_ensembles_per_worker, emd_queue, noise_std, num_shifts),
        )
        for _ in range(n_workers)
    ]

    for p in processes:
        p.start()

    worker_results = [emd_queue.get() for _ in processes]

    for p in processes:
        p.join()

    # Average all worker results
    imfs = np.mean(worker_results, axis=0)

    return imfs


def _emd_worker(
    signal: np.ndarray,
    n_ensembles: int,
    output_queue: mp.Queue,
    noise_std: float,
    num_shifts: int,
) -> None:
    """
    Worker function for parallel EEMD processing.

    Args:
        signal: 2D array of shape (samples, channels)
        n_ensembles: Number of ensemble realizations for this worker
        output_queue: Queue for returning results
        noise_std: Standard deviation of added noise
        num_shifts: Number of shifts for sifting
    """
    np.random.seed()

    n_samples, n_channels = signal.shape
    channel_imfs = []

    # Process each channel separately
    for ch in range(n_channels):
        channel_signal = signal[:, ch]
        all_modes = emd.eemd(channel_signal, noise_std, n_ensembles, num_shifts)

        # Extract only the IMFs (exclude original signal at index 0)
        # The library returns: [original, imf1, imf2, ..., imfN, residue]
        # We want: [imf1, imf2, ..., imfN, residue]
        imfs_only = all_modes[1:]  # Skip the original signal at index 0
        channel_imfs.append(imfs_only)

    # Stack results: (n_imfs, samples, channels)
    n_imfs = channel_imfs[0].shape[0]
    combined_imfs = np.zeros((n_imfs, n_samples, n_channels))

    for ch, ch_imfs in enumerate(channel_imfs):
        combined_imfs[:, :, ch] = ch_imfs

    output_queue.put(combined_imfs)


def calc_instantaneous_info(
    imfs: np.ndarray, fs: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate instantaneous frequency, amplitude, and phase for multi-channel IMFs.

    Args:
        imfs: 3D array of shape (n_imfs, samples, channels)
        fs: Sampling frequency

    Returns:
        Tuple of (instantaneous_frequency, instantaneous_amplitude, instantaneous_phase)
        Each array has shape (n_imfs, samples, channels)
    """
    n_imfs, n_samples, n_channels = imfs.shape

    inst_freq = np.zeros((n_imfs, n_samples, n_channels))
    inst_amp = np.zeros((n_imfs, n_samples, n_channels))
    inst_phase = np.zeros((n_imfs, n_samples, n_channels))

    # Process each channel separately
    for ch in range(n_channels):
        channel_imfs = imfs[:, :, ch]
        i_f, i_a, i_p = emd.calc_inst_info(channel_imfs, fs)

        inst_freq[:, :, ch] = i_f
        inst_amp[:, :, ch] = i_a
        inst_phase[:, :, ch] = i_p

    return inst_freq, inst_amp, inst_phase


def calculate_welch_psd(
    signal: np.ndarray, fs: float, nperseg: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate smooth power spectral density using Welch's method.

    Args:
        signal: 1D signal array
        fs: Sampling frequency
        nperseg: Length of each segment (None for auto)

    Returns:
        freqs, psd: Frequency array and power spectral density
    """
    if nperseg is None:
        # Good default: segment length that provides smooth results
        nperseg = min(len(signal) // 8, 2048)
        # Ensure minimum segment length
        nperseg = max(nperseg, 256)

    freqs, psd = welch(
        signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,  # 50% overlap
        window="hann",  # Hanning window
        detrend="linear",  # Remove linear trends
        scaling="density",  # Power spectral density
    )

    return freqs, psd


def calculate_peak_frequencies(
    imfs: np.ndarray,
    fs: float,
    channel: int = 0,
    welch_nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate peak frequencies for each IMF and group them into frequency bands.

    Args:
        imfs: 3D array of shape (n_imfs, samples, channels)
        fs: Sampling frequency
        channel: Channel to analyze (default: 0)
        welch_nperseg: Segment length for Welch method (None for auto)

    Returns:
        Tuple of (peak_frequencies, low_freq_mask, theta_mask, supra_theta_mask)
        - peak_frequencies: Array of peak frequencies for each IMF
        - low_freq_mask: mask for IMFs with peak < 5 Hz (including residue as False)
        - theta_mask: mask for IMFs with peak 5-12 Hz (including residue as False)
        - supra_theta_mask: mask for IMFs with peak > 12 Hz (including residue as False)
    """
    n_imfs = imfs.shape[0]
    peak_freqs = []

    for i in range(n_imfs - 1):  # Exclude residue
        freqs_welch, psd_welch = calculate_welch_psd(
            imfs[i, :, channel], fs, welch_nperseg
        )

        # Filter to frequency range (0.1 Hz to Nyquist)
        freq_mask = (freqs_welch >= 0.1) & (freqs_welch <= fs / 2)
        freqs_plot = freqs_welch[freq_mask]
        psd_plot = psd_welch[freq_mask]

        # Find peak frequency from raw PSD
        peak_idx = np.argmax(psd_plot)
        peak_freqs.append(freqs_plot[peak_idx])

    peak_freqs = np.array(peak_freqs)

    # Create full-size masks (including residue)
    low_freq_mask = np.zeros(n_imfs, dtype=bool)
    theta_mask = np.zeros(n_imfs, dtype=bool)
    supra_theta_mask = np.zeros(n_imfs, dtype=bool)

    # Set masks for non-residue IMFs
    low_freq_mask[:-1] = peak_freqs < 5
    theta_mask[:-1] = (peak_freqs >= 5) & (peak_freqs <= 12)
    supra_theta_mask[:-1] = peak_freqs > 12

    return peak_freqs, low_freq_mask, theta_mask, supra_theta_mask


def extract_frequency_bands(
    imfs: np.ndarray,
    low_freq_mask: np.ndarray,
    theta_mask: np.ndarray,
    supra_theta_mask: np.ndarray,
    channel: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract signals from different frequency bands by summing relevant IMFs.

    Args:
        imfs: 3D array of shape (n_imfs, samples, channels)
        low_freq_mask: Boolean mask for IMFs with peak < 5 Hz
        theta_mask: Boolean mask for IMFs with peak 5-12 Hz
        supra_theta_mask: Boolean mask for IMFs with peak > 12 Hz
        channel: Channel to analyze (default: 0)

    Returns:
        Tuple of (low_freq_signal, theta_signal, supra_theta_signal)
        - low_freq_signal: Signal from IMFs with peak < 5 Hz
        - theta_signal: Signal from IMFs with peak 5-12 Hz
        - supra_theta_signal: Signal from IMFs with peak > 12 Hz
    """
    # Sum IMFs in each band (excluding residue)
    low_freq_signal = np.sum(imfs[low_freq_mask, :, channel], axis=0)
    theta_signal = np.sum(imfs[theta_mask, :, channel], axis=0)
    supra_theta_signal = np.sum(imfs[supra_theta_mask, :, channel], axis=0)

    return low_freq_signal, theta_signal, supra_theta_signal


def plot_emd_analysis(
    signal: np.ndarray,
    imfs: np.ndarray,
    fs: float,
    inst_freq: np.ndarray | None = None,
    inst_amp: np.ndarray | None = None,
    channel: int = 0,
    time_window: tuple[float, float] | None = None,
    max_imfs_display: int | None = None,
    freq_range: tuple[float, float] = (0, 100),
    figsize: tuple[float, float] = (18, 20),
    welch_nperseg: int | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Create comprehensive EMD analysis plots including frequency band analysis.

    Args:
        signal: 2D array of shape (samples, channels)
        imfs: 3D array of shape (n_imfs, samples, channels)
        fs: Sampling frequency
        inst_freq: Instantaneous frequency array (optional)
        inst_amp: Instantaneous amplitude array (optional)
        channel: Channel to analyze (default: 0)
        time_window: Time range to display as (start, end) in seconds
        max_imfs_display: Maximum number of IMFs to show in plots
        freq_range: Frequency range for spectrum plots as (min, max)
        figsize: Figure size as (width, height)
        welch_nperseg: Segment length for Welch method (None for auto)

    Returns:
        Tuple of (figure, axes_array)
    """
    n_samples, n_channels = signal.shape
    n_imfs = imfs.shape[0]
    if max_imfs_display is None:
        max_imfs_display = n_imfs

    # Validate inputs
    if channel >= n_channels:
        raise ValueError(
            f"Channel {channel} not available. Signal has {n_channels} channels."
        )

    # Create time vector
    t = np.linspace(0, n_samples / fs, n_samples)

    # Determine time window for display
    if time_window is None:
        display_samples = min(int(fs), n_samples)
        t_display = t[:display_samples]
        time_slice = slice(0, display_samples)
    else:
        start_idx = int(time_window[0] * fs)
        end_idx = int(time_window[1] * fs)
        time_slice = slice(start_idx, min(end_idx, n_samples))
        t_display = t[time_slice]
        display_samples = len(t_display)

    # Create figure with 12 subplots in a 3x4 grid
    fig, axes = plt.subplots(4, 3, figsize=(24, 18))
    axes = axes.flatten()

    # Calculate peak frequencies and frequency band signals
    peak_freqs, low_freq_mask, theta_mask, supra_theta_mask = (
        calculate_peak_frequencies(imfs, fs, channel, welch_nperseg)
    )
    low_freq_signal, theta_signal, supra_theta_signal = extract_frequency_bands(
        imfs, low_freq_mask, theta_mask, supra_theta_mask, channel
    )

    # Plot 1: Original signals
    ax1 = axes[0]
    for ch in range(min(n_channels, 4)):  # Show max 4 channels
        ax1.plot(t_display, signal[time_slice, ch], label=f"Channel {ch+1}", alpha=0.8)
    ax1.set_title(
        f"Original Multi-channel Signal ({t_display[0]:.1f}-{t_display[-1]:.1f}s)"
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: IMFs for selected channel
    ax2 = axes[1]
    n_imfs_to_show = min(max_imfs_display, n_imfs)
    for i in range(n_imfs_to_show):
        offset = i * 2 * np.std(imfs[i, time_slice, channel])
        ax2.plot(t_display, imfs[i, time_slice, channel] + offset, label=f"IMF {i+1}")
    ax2.set_title(f"IMFs - Channel {channel+1}")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude (offset)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # # Plot 3: Instantaneous Frequency
    # ax3 = axes[2]
    # if inst_freq is not None:
    #     n_freq_imfs = min(3, n_imfs - 1)  # Exclude residue
    #     for i in range(n_freq_imfs):
    #         valid_freq = inst_freq[i, time_slice, channel]
    #         # Remove invalid values
    #         mask = np.isfinite(valid_freq) & (valid_freq > 0) & (valid_freq < fs / 2)
    #         if np.any(mask):
    #             # Smooth for visualization
    #             smooth_freq = np.convolve(
    #                 valid_freq,
    #                 np.ones(min(20, len(valid_freq) // 10))
    #                 / min(20, len(valid_freq) // 10),
    #                 mode="same",
    #             )
    #             ax3.plot(t_display, smooth_freq, label=f"IMF {i+1}", alpha=0.8)
    #     ax3.set_ylim(freq_range)
    # else:
    #     ax3.text(
    #         0.5,
    #         0.5,
    #         "Instantaneous frequency\nnot computed",
    #         ha="center",
    #         va="center",
    #         transform=ax3.transAxes,
    #     )
    # ax3.set_title(f"Instantaneous Frequency - Channel {channel+1}")
    # ax3.set_xlabel("Time (s)")
    # ax3.set_ylabel("Frequency (Hz)")
    # ax3.legend()
    # ax3.grid(True, alpha=0.3)

    # # Plot 4: Instantaneous Amplitude
    # ax4 = axes[3]
    # if inst_amp is not None:
    #     n_amp_imfs = min(3, n_imfs - 1)  # Exclude residue
    #     for i in range(n_amp_imfs):
    #         ax4.plot(
    #             t_display,
    #             inst_amp[i, time_slice, channel],
    #             label=f"IMF {i+1}",
    #             alpha=0.8,
    #         )
    # else:
    #     ax4.text(
    #         0.5,
    #         0.5,
    #         "Instantaneous amplitude\nnot computed",
    #         ha="center",
    #         va="center",
    #         transform=ax4.transAxes,
    #     )
    # ax4.set_title(f"Instantaneous Amplitude - Channel {channel+1}")
    # ax4.set_xlabel("Time (s)")
    # ax4.set_ylabel("Amplitude")
    # ax4.legend()
    # ax4.grid(True, alpha=0.3)

    # # Plot 5: Frequency spectrum comparison (using Welch method)
    # ax5 = axes[4]
    # freqs_welch_orig, psd_orig = calculate_welch_psd(
    #     signal[:, channel], fs, welch_nperseg
    # )
    # reconstructed = np.sum(imfs[:, :, channel], axis=0)
    # freqs_welch_recon, psd_recon = calculate_welch_psd(reconstructed, fs, welch_nperseg)

    # mask_orig = (freqs_welch_orig >= freq_range[0]) & (
    #     freqs_welch_orig <= freq_range[1]
    # )
    # mask_recon = (freqs_welch_recon >= freq_range[0]) & (
    #     freqs_welch_recon <= freq_range[1]
    # )

    # ax5.semilogy(
    #     freqs_welch_orig[mask_orig],
    #     psd_orig[mask_orig],
    #     "b-",
    #     label="Original",
    #     alpha=0.7,
    #     linewidth=2,
    # )
    # ax5.semilogy(
    #     freqs_welch_recon[mask_recon],
    #     psd_recon[mask_recon],
    #     "r--",
    #     label="Reconstructed",
    #     alpha=0.7,
    #     linewidth=2,
    # )
    # ax5.set_title(f"Frequency Spectrum (Welch) - Channel {channel+1}")
    # ax5.set_xlabel("Frequency (Hz)")
    # ax5.set_ylabel("Power Spectral Density")
    # ax5.legend()
    # ax5.grid(True, alpha=0.3)
    # ax5.set_xlim(freq_range)

    # # Plot 6: Reconstruction error and statistics
    # ax6 = axes[5]
    # reconstruction_error = signal[:, channel] - reconstructed
    # ax6.plot(t_display, reconstruction_error[time_slice], "r-", alpha=0.8)
    # ax6.set_title(f"Reconstruction Error - Channel {channel+1}")
    # ax6.set_xlabel("Time (s)")
    # ax6.set_ylabel("Error")
    # ax6.grid(True, alpha=0.3)

    # # Add statistics
    # mse = np.mean(reconstruction_error**2)
    # max_error = np.max(np.abs(reconstruction_error))
    # snr = 10 * np.log10(np.var(signal[:, channel]) / mse) if mse > 0 else np.inf

    # stats_text = f"MSE: {mse:.2e}\nMax Error: {max_error:.3f}\nSNR: {snr:.1f} dB"
    # ax6.text(
    #     0.02,
    #     0.98,
    #     stats_text,
    #     transform=ax6.transAxes,
    #     verticalalignment="top",
    #     bbox={"boxstyle": "round", "facecolor": "wheat"},
    # )

    # Plot 7: Z-scored Power Spectra using Welch method
    ax7 = axes[6]
    n_imfs_spectrum = min(max_imfs_display, n_imfs - 1)  # Exclude residue
    colors = plt.cm.tab10(np.linspace(0, 1, n_imfs_spectrum))
    peak_frequencies = []  # Store peak frequencies for annotation

    for i in range(n_imfs_spectrum):
        freqs_welch, psd_welch = calculate_welch_psd(
            imfs[i, :, channel], fs, welch_nperseg
        )
        freq_mask = (freqs_welch >= freq_range[0]) & (freqs_welch <= freq_range[1])
        freqs_plot = freqs_welch[freq_mask]
        psd_plot = psd_welch[freq_mask]

        if len(psd_plot) > 0 and np.std(psd_plot) > 0:
            psd_z = (psd_plot - np.mean(psd_plot)) / np.std(psd_plot)
        else:
            psd_z = np.zeros_like(psd_plot)

        ax7.semilogx(
            freqs_plot,
            psd_z,
            color=colors[i],
            label=f"IMF {i+1}",
            linewidth=2,
            alpha=0.8,
        )

        peak_freq = peak_freqs[i]
        peak_power = psd_z[np.argmin(np.abs(freqs_plot - peak_freq))]
        peak_frequencies.append((peak_freq, peak_power, colors[i], i + 1))

    for peak_freq, peak_power, color, _ in peak_frequencies:
        ax7.annotate(
            f"{peak_freq:.1f} Hz",
            xy=(peak_freq, peak_power),
            xytext=(peak_freq * 1.2, peak_power + 0.5),
            fontsize=9,
            ha="left",
            arrowprops={"arrowstyle": "->", "color": color, "alpha": 0.7},
        )

    ax7.set_title(f"Z-scored Power Spectra (Welch) - Channel {channel+1}")
    ax7.set_xlabel("Frequency (Hz)")
    ax7.set_ylabel("Z-scored Power")
    ax7.grid(True, alpha=0.3)
    ax7.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax7.set_xlim(freq_range)

    # Plot 8: Raw Power Spectra using Welch method (Non-normalized)
    ax8 = axes[7]
    peak_frequencies_raw = []  # Store peak frequencies for annotation

    for i in range(n_imfs_spectrum):
        freqs_welch, psd_welch = calculate_welch_psd(
            imfs[i, :, channel], fs, welch_nperseg
        )
        freq_mask = (freqs_welch >= freq_range[0]) & (freqs_welch <= freq_range[1])
        freqs_plot = freqs_welch[freq_mask]
        psd_plot = psd_welch[freq_mask]

        if len(psd_plot) > 0:
            ax8.loglog(
                freqs_plot,
                psd_plot,
                color=colors[i],
                label=f"IMF {i+1}",
                linewidth=2,
                alpha=0.8,
            )

            peak_idx = np.argmax(psd_plot)
            peak_freq = freqs_plot[peak_idx]
            peak_power = psd_plot[peak_idx]
            peak_frequencies_raw.append((peak_freq, peak_power, colors[i], i + 1))

    for peak_freq, peak_power, color, _ in peak_frequencies_raw:
        ax8.annotate(
            f"{peak_freq:.1f} Hz",
            xy=(peak_freq, peak_power),
            xytext=(peak_freq + freq_range[1] * 0.05, peak_power * 2),
            fontsize=9,
            ha="left",
            arrowprops={"arrowstyle": "->", "color": color, "alpha": 0.7},
        )

    ax8.set_title(f"Raw Power Spectra (Welch) - Channel {channel+1}")
    ax8.set_xlabel("Frequency (Hz)")
    ax8.set_ylabel("Power Spectral Density (log scale)")
    ax8.grid(True, alpha=0.3)
    ax8.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax8.set_xlim(freq_range)
    ax8.set_ylim(bottom=1e-2)  # Avoid log(0) issues

    # Plot 9: IMF Power Distribution
    ax9 = axes[8]
    imf_powers = []
    imf_labels = []

    for i in range(min(max_imfs_display, n_imfs)):
        power = np.var(imfs[i, :, channel])
        imf_powers.append(power)
        if i == n_imfs - 1:
            imf_labels.append("Residue")
        else:
            imf_labels.append(f"IMF {i+1}")

    bars = ax9.bar(
        range(len(imf_powers)), imf_powers, alpha=0.7, color=colors[: len(imf_powers)]
    )
    ax9.set_title(f"Power Distribution Across IMFs - Channel {channel+1}")
    ax9.set_xlabel("Component")
    ax9.set_ylabel("Power (Variance)")
    ax9.set_xticks(range(len(imf_labels)))
    ax9.set_xticklabels(imf_labels, rotation=45)
    ax9.grid(True, alpha=0.3)

    # Add percentage labels on bars
    total_power = sum(imf_powers)
    for _, (bar, power) in enumerate(zip(bars, imf_powers)):
        percentage = 100 * power / total_power
        ax9.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(imf_powers) * 0.01,
            f"{percentage:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Plot 10: Z-scored Power Spectra by Frequency Band
    ax10 = axes[9]

    # Calculate and plot Welch PSD for each band
    for signal_data, label, color in [
        (low_freq_signal, "Low-frequency signal", "blue"),
        (theta_signal, "Theta signal", "green"),
        (supra_theta_signal, "Supra-theta signal", "red"),
    ]:
        freqs_welch, psd_welch = calculate_welch_psd(signal_data, fs, welch_nperseg)

        # Filter to frequency range
        freq_mask = (freqs_welch >= 0.1) & (freqs_welch <= fs / 2)
        freqs_plot = freqs_welch[freq_mask]
        psd_plot = psd_welch[freq_mask]

        # Z-score normalization
        if len(psd_plot) > 0 and np.std(psd_plot) > 0:
            psd_z = (psd_plot - np.mean(psd_plot)) / np.std(psd_plot)
        else:
            psd_z = np.zeros_like(psd_plot)

        ax10.semilogx(freqs_plot, psd_z, label=label, color=color, linewidth=2)

    # Add raw signal power spectrum
    freqs_welch, psd_welch = calculate_welch_psd(signal[:, channel], fs, welch_nperseg)
    freq_mask = (freqs_welch >= 0.1) & (freqs_welch <= fs / 2)
    freqs_plot = freqs_welch[freq_mask]
    psd_plot = psd_welch[freq_mask]

    # Z-score normalization
    if len(psd_plot) > 0 and np.std(psd_plot) > 0:
        psd_z = (psd_plot - np.mean(psd_plot)) / np.std(psd_plot)
    else:
        psd_z = np.zeros_like(psd_plot)

    ax10.semilogx(
        freqs_plot,
        psd_z,
        label="Raw LFP",
        color="black",
        linewidth=2,
        linestyle="--",
    )

    ax10.set_title("Z-scored Power Spectra by Frequency Band")
    ax10.set_xlabel("Frequency (Hz)")
    ax10.set_ylabel("Z-scored Power")
    ax10.grid(True, alpha=0.3)
    ax10.legend()
    ax10.set_xlim(1, fs / 2)  # Show from 1 Hz to Nyquist frequency

    # Plot 11: Example traces by frequency band
    ax11 = axes[10]

    # Plot example traces with different offsets for visibility
    offset = 4 * np.std(low_freq_signal)
    ax11.plot(
        t_display,
        low_freq_signal[time_slice] + offset,
        label="Low-frequency signal",
        color="blue",
        alpha=0.8,
    )
    ax11.plot(
        t_display,
        theta_signal[time_slice],
        label="Theta signal",
        color="green",
        alpha=0.8,
    )
    ax11.plot(
        t_display,
        supra_theta_signal[time_slice] - offset,
        label="Supra-theta signal",
        color="red",
        alpha=0.8,
    )

    ax11.set_title("Example Traces by Frequency Band")
    ax11.set_xlabel("Time (s)")
    ax11.set_ylabel("Amplitude (offset)")
    ax11.grid(True, alpha=0.3)
    ax11.legend()

    # Hide the last subplot (index 11)
    axes[11].set_visible(False)

    plt.tight_layout()

    # Print summary
    print(f"\nEMD Analysis Summary (Channel {channel+1}):")
    print(f"Number of IMFs: {n_imfs}")
    print(f"Signal length: {n_samples} samples ({n_samples/fs:.2f}s)")
    # print(f"Reconstruction MSE: {mse:.2e}")
    # print(f"Signal-to-Noise Ratio: {snr:.1f} dB")
    # print(f"Max absolute error: {max_error:.3f}")

    # # Print power distribution
    # print("\nPower Distribution:")
    # for _, (label, power) in enumerate(zip(imf_labels, imf_powers)):
    #     percentage = 100 * power / total_power
    #     print(f"  {label}: {percentage:.1f}% of total power")

    # Print dominant frequencies from Welch analysis
    print("\nDominant Frequencies (Welch method):")
    for peak_freq, _, _, imf_num in peak_frequencies_raw:
        print(f"  IMF {imf_num}: {peak_freq:.1f} Hz")

    # Print frequency band analysis
    print("\nFrequency Band Analysis Summary:")
    print(f"Low-frequency components (<5 Hz): {np.sum(low_freq_mask)} IMFs")
    print(f"Theta components (5-12 Hz): {np.sum(theta_mask)} IMFs")
    print(f"Supra-theta components (>12 Hz): {np.sum(supra_theta_mask)} IMFs")

    return fig, axes


def example_usage():
    """Example demonstrating multi-channel EEMD with visualization."""
    # Generate example multi-channel signal
    fs = 1000
    t = np.linspace(0, 3, 3000)
    n_channels = 2

    # Create multi-channel signal with known components
    signal = np.zeros((len(t), n_channels))
    for ch in range(n_channels):
        # Channel-specific frequency modulation
        freq_mod = 5 + ch * 2  # Different base frequencies
        amp_mod = 0.8 + ch * 0.2  # Different amplitudes

        signal[:, ch] = (
            amp_mod * np.sin(2 * np.pi * freq_mod * t)  # Low frequency
            + 0.6 * np.sin(2 * np.pi * (20 + ch * 5) * t)  # Mid frequency
            + 0.3 * np.sin(2 * np.pi * (50 + ch * 10) * t)  # High frequency
            + 0.1 * np.random.randn(len(t))  # Noise
        )

    print("Performing EEMD decomposition...")
    # Perform EEMD (reduced ensembles for faster demo)
    imfs = eemd(signal, n_workers=2, n_ensembles=50)
    print(f"IMFs shape: {imfs.shape}")

    # Calculate instantaneous information
    print("Calculating instantaneous frequency and amplitude...")
    inst_freq, inst_amp, inst_phase = calc_instantaneous_info(imfs, fs)

    # Create plots using the updated function with Welch method
    fig, axes = plot_emd_analysis(
        signal=signal,
        imfs=imfs,
        fs=fs,
        inst_freq=inst_freq,
        inst_amp=inst_amp,
        channel=0,  # Analyze first channel
        time_window=(0, 1),  # Show first second
        max_imfs_display=5,
        freq_range=(0, 100),
        welch_nperseg=512,  # Segment length for Welch method
    )

    plt.show()

    return imfs, inst_freq, inst_amp, inst_phase


if __name__ == "__main__":
    example_usage()
