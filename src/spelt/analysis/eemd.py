import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np

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


def plot_emd(
    signal: np.ndarray,
    imfs: np.ndarray,
    fs: float,
    inst_freq: np.ndarray | None = None,
    inst_amp: np.ndarray | None = None,
    channel: int = 0,
    time_window: tuple[float, float] | None = None,
    max_imfs_display: int = 5,
    freq_range: tuple[float, float] = (0, 100),
    figsize: tuple[float, float] = (18, 14),
) -> tuple[plt.Figure, np.ndarray]:
    """
    Create comprehensive EMD analysis plots.

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

    Returns:
        Tuple of (figure, axes_array)
    """
    import matplotlib.pyplot as plt

    n_samples, n_channels = signal.shape
    n_imfs = imfs.shape[0]

    # Validate inputs
    if channel >= n_channels:
        raise ValueError(
            f"Channel {channel} not available. Signal has {n_channels} channels."
        )

    # Create time vector
    t = np.linspace(0, n_samples / fs, n_samples)

    # Determine time window for display
    if time_window is None:
        # Show first 1 second or full signal if shorter
        display_samples = min(int(fs), n_samples)
        t_display = t[:display_samples]
        time_slice = slice(0, display_samples)
    else:
        start_idx = int(time_window[0] * fs)
        end_idx = int(time_window[1] * fs)
        time_slice = slice(start_idx, min(end_idx, n_samples))
        t_display = t[time_slice]
        display_samples = len(t_display)

    # Create figure with 7 subplots
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    axes = axes.flatten()

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

    # Plot 3: Instantaneous Frequency
    ax3 = axes[2]
    if inst_freq is not None:
        n_freq_imfs = min(3, n_imfs - 1)  # Exclude residue
        for i in range(n_freq_imfs):
            valid_freq = inst_freq[i, time_slice, channel]
            # Remove invalid values
            mask = np.isfinite(valid_freq) & (valid_freq > 0) & (valid_freq < fs / 2)
            if np.any(mask):
                # Smooth for visualization
                smooth_freq = np.convolve(
                    valid_freq,
                    np.ones(min(20, len(valid_freq) // 10))
                    / min(20, len(valid_freq) // 10),
                    mode="same",
                )
                ax3.plot(t_display, smooth_freq, label=f"IMF {i+1}", alpha=0.8)
        ax3.set_ylim(freq_range)
    else:
        ax3.text(
            0.5,
            0.5,
            "Instantaneous frequency\nnot computed",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )
    ax3.set_title(f"Instantaneous Frequency - Channel {channel+1}")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Frequency (Hz)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Instantaneous Amplitude
    ax4 = axes[3]
    if inst_amp is not None:
        n_amp_imfs = min(3, n_imfs - 1)  # Exclude residue
        for i in range(n_amp_imfs):
            ax4.plot(
                t_display,
                inst_amp[i, time_slice, channel],
                label=f"IMF {i+1}",
                alpha=0.8,
            )
    else:
        ax4.text(
            0.5,
            0.5,
            "Instantaneous amplitude\nnot computed",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )
    ax4.set_title(f"Instantaneous Amplitude - Channel {channel+1}")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Amplitude")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Frequency spectrum comparison
    ax5 = axes[4]
    freqs = np.fft.fftfreq(n_samples, 1 / fs)
    fft_orig = np.abs(np.fft.fft(signal[:, channel]))

    # Reconstruct signal from IMFs
    reconstructed = np.sum(imfs[:, :, channel], axis=0)
    fft_recon = np.abs(np.fft.fft(reconstructed))

    # Plot positive frequencies only
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    ax5.semilogy(
        freqs[mask], fft_orig[mask], "b-", label="Original", alpha=0.7, linewidth=2
    )
    ax5.semilogy(
        freqs[mask],
        fft_recon[mask],
        "r--",
        label="Reconstructed",
        alpha=0.7,
        linewidth=2,
    )
    ax5.set_title(f"Frequency Spectrum - Channel {channel+1}")
    ax5.set_xlabel("Frequency (Hz)")
    ax5.set_ylabel("Magnitude")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(freq_range)

    # Plot 6: Reconstruction error and statistics
    ax6 = axes[5]
    reconstruction_error = signal[:, channel] - reconstructed
    ax6.plot(t_display, reconstruction_error[time_slice], "r-", alpha=0.8)
    ax6.set_title(f"Reconstruction Error - Channel {channel+1}")
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Error")
    ax6.grid(True, alpha=0.3)

    # Add statistics
    mse = np.mean(reconstruction_error**2)
    max_error = np.max(np.abs(reconstruction_error))
    snr = 10 * np.log10(np.var(signal[:, channel]) / mse) if mse > 0 else np.inf

    stats_text = f"MSE: {mse:.2e}\nMax Error: {max_error:.3f}\nSNR: {snr:.1f} dB"
    ax6.text(
        0.02,
        0.98,
        stats_text,
        transform=ax6.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat"),
    )

    # Plot 7: Power Spectrum of IMFs (normalized and non-normalized)
    ax7 = axes[6]
    freqs = np.fft.fftfreq(n_samples, 1 / fs)
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_plot = freqs[freq_mask]

    # Calculate power spectra for each IMF
    n_imfs_spectrum = min(max_imfs_display, n_imfs - 1)  # Exclude residue
    colors = plt.cm.tab10(np.linspace(0, 1, n_imfs_spectrum))

    # Plot normalized (z-scored) spectra
    for i in range(n_imfs_spectrum):
        # Calculate power spectrum
        fft_imf = np.fft.fft(imfs[i, :, channel])
        power_spectrum = np.abs(fft_imf[freq_mask]) ** 2

        # Z-score normalization
        power_mean = np.mean(power_spectrum)
        power_std = np.std(power_spectrum)
        if power_std > 0:
            power_z = (power_spectrum - power_mean) / power_std
        else:
            power_z = np.zeros_like(power_spectrum)

        # Plot
        ax7.plot(
            freqs_plot,
            power_z,
            color=colors[i],
            label=f"IMF {i+1}",
            linewidth=1.5,
            alpha=0.8,
        )

        # Find and annotate peak frequency
        if len(power_z) > 0:
            peak_idx = np.argmax(power_z)
            peak_freq = freqs_plot[peak_idx]
            peak_power = power_z[peak_idx]

            # Annotate peak frequency
            ax7.annotate(
                f"{peak_freq:.1f} Hz",
                xy=(peak_freq, peak_power),
                xytext=(peak_freq + freq_range[1] * 0.05, peak_power + 0.5),
                fontsize=9,
                ha="left",
                arrowprops=dict(arrowstyle="->", color=colors[i], alpha=0.7),
            )

    ax7.set_title(f"Z-scored Power Spectra (Normalized) - Channel {channel+1}")
    ax7.set_xlabel("Frequency (Hz)")
    ax7.set_ylabel("Z-scored Power")
    ax7.grid(True, alpha=0.3)
    ax7.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax7.set_xlim(freq_range)

    # Plot 8: Raw Power Spectra (Non-normalized)
    ax8 = axes[7]

    # Plot raw (non-normalized) spectra on log scale
    for i in range(n_imfs_spectrum):
        # Calculate power spectrum
        fft_imf = np.fft.fft(imfs[i, :, channel])
        power_spectrum = np.abs(fft_imf[freq_mask]) ** 2

        # Plot on log scale
        ax8.semilogy(
            freqs_plot,
            power_spectrum,
            color=colors[i],
            label=f"IMF {i+1}",
            linewidth=1.5,
            alpha=0.8,
        )

        # Find and annotate peak frequency
        if len(power_spectrum) > 0:
            peak_idx = np.argmax(power_spectrum)
            peak_freq = freqs_plot[peak_idx]
            peak_power = power_spectrum[peak_idx]

            # Annotate peak frequency
            ax8.annotate(
                f"{peak_freq:.1f} Hz",
                xy=(peak_freq, peak_power),
                xytext=(peak_freq + freq_range[1] * 0.05, peak_power * 2),
                fontsize=9,
                ha="left",
                arrowprops=dict(arrowstyle="->", color=colors[i], alpha=0.7),
            )

    ax8.set_title(f"Raw Power Spectra (Non-normalized) - Channel {channel+1}")
    ax8.set_xlabel("Frequency (Hz)")
    ax8.set_ylabel("Power (log scale)")
    ax8.grid(True, alpha=0.3)
    ax8.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax8.set_xlim(freq_range)

    # Plot 9: IMF Power Distribution (moved from position 8)
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
    for i, (bar, power) in enumerate(zip(bars, imf_powers)):
        percentage = 100 * power / total_power
        ax9.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(imf_powers) * 0.01,
            f"{percentage:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    # Print summary
    print(f"\nEMD Analysis Summary (Channel {channel+1}):")
    print(f"Number of IMFs: {n_imfs}")
    print(f"Signal length: {n_samples} samples ({n_samples/fs:.2f}s)")
    print(f"Reconstruction MSE: {mse:.2e}")
    print(f"Signal-to-Noise Ratio: {snr:.1f} dB")
    print(f"Max absolute error: {max_error:.3f}")

    # Print power distribution
    print(f"\nPower Distribution:")
    for i, (label, power) in enumerate(zip(imf_labels, imf_powers)):
        percentage = 100 * power / total_power
        print(f"  {label}: {percentage:.1f}% of total power")

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

    # Create plots using the new function
    fig, axes = plot_emd(
        signal=signal,
        imfs=imfs,
        fs=fs,
        inst_freq=inst_freq,
        inst_amp=inst_amp,
        channel=0,  # Analyze first channel
        time_window=(0, 1),  # Show first second
        max_imfs_display=5,
        freq_range=(0, 100),
    )

    plt.show()

    return imfs, inst_freq, inst_amp, inst_phase


if __name__ == "__main__":
    example_usage()
