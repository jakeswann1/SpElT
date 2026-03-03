"""
Phase-Amplitude Coupling (PAC) analysis for LFP signals.

Implements theta-phase / gamma-amplitude coupling metrics used to measure
hippocampal theta-gamma coordination across CA1 laminae.

Metrics
-------
- Modulation Index (MI)  — Tort et al. 2010, J Neurophysiol
- Mean Vector Length (MVL) — Canolty et al. 2006, Science

Phase convention: matches get_signal_phase() — 0/2π = peak, π = trough.

Typical usage
-------------
>>> theta_phase, _ = get_signal_phase(lfp_pyr, fs, bandpass_range=(4, 12))
>>> result = compute_pac_depth_profile(
...     theta_phase=theta_phase,
...     lfp_column=lfp_all_channels,   # (n_samples, n_channels)
...     fs=1000.0,
...     gamma_bands=GAMMA_BANDS,
... )
"""

import numpy as np
import pywt
from scipy.signal import hilbert

from .filtering import bandpass_filter_lfp

# Default gamma sub-band definitions (Hz)
GAMMA_BANDS: dict[str, tuple[float, float]] = {
    "slow": (30, 45),  # EC→SLM input
    "medium": (60, 90),  # CA3→SR input
    "fast": (90, 140),  # Local / SP
}


def morlet_n_cycles(
    center_freq: float,
    min_cycles: float = 3.0,
    max_cycles: float = 8.0,
    min_freq: float = 20.0,
    max_freq: float = 150.0,
) -> float:
    """Linearly scale Morlet wavelet cycles with centre frequency.

    Matches the convention used in ``nf_dcwt.m`` (Rawls 2023 / NeuroFreq
    toolbox) and ``theta_phase_spectrogram.m``: the wavelet has
    ``min_cycles`` cycles at ``min_freq`` and ``max_cycles`` at
    ``max_freq``, scaling linearly between.  This trades frequency
    resolution at low frequencies for better temporal resolution — the
    inverse of using a fixed high cycle count.

    Parameters
    ----------
    center_freq : float
        Target centre frequency in Hz.
    min_cycles, max_cycles : float
        Cycle counts at ``min_freq`` and ``max_freq`` respectively.
    min_freq, max_freq : float
        Frequency range over which cycles scale linearly.

    Returns
    -------
    float
        Number of Morlet cycles at ``center_freq``.

    Examples
    --------
    >>> morlet_n_cycles(20)   # 3.0
    >>> morlet_n_cycles(150)  # 8.0
    >>> morlet_n_cycles(37.5) # ≈ 3.67 (slow gamma centre)
    >>> morlet_n_cycles(75)   # ≈ 5.12 (medium gamma centre)
    >>> morlet_n_cycles(115)  # ≈ 6.65 (fast gamma centre)
    """
    t = float(np.clip((center_freq - min_freq) / (max_freq - min_freq), 0.0, 1.0))
    return min_cycles + (max_cycles - min_cycles) * t


# ---------------------------------------------------------------------------
# Low-level building blocks
# ---------------------------------------------------------------------------


def compute_amplitude_envelope_cwt(
    lfp: np.ndarray, fs: float, center_freq: float, n_cycles: float = 7.0
) -> np.ndarray:
    """Amplitude envelope via complex Morlet CWT (pywavelets).

    Parameters
    ----------
    lfp : np.ndarray
        1-D (n_samples,) or 2-D (n_samples, n_channels) LFP array.
    fs : float
        Sampling rate in Hz.
    center_freq : float
        Target frequency in Hz (typically (freq_min + freq_max) / 2).
    n_cycles : float
        Number of cycles in the Morlet wavelet (default 7). Controls the
        time–frequency resolution trade-off: more cycles → finer frequency
        resolution, coarser time resolution.  Mapped to pywt bandwidth
        parameter as B = n_cycles² / 2 (wavelet ``cmor{B}-1.0``).

    Returns
    -------
    np.ndarray
        Amplitude envelope, same shape as *lfp*.
    """
    # cmor{B}-1.0: Gaussian envelope σ_t = sqrt(B/2).
    # Setting B = n_cycles²/2 gives ~n_cycles oscillations within ±1σ.
    wavelet = f"cmor{n_cycles ** 2 / 2.0:.4f}-1.0"
    scale = pywt.scale2frequency(wavelet, 1.0, precision=8) * fs / center_freq

    if lfp.ndim == 1:
        coeffs, _ = pywt.cwt(
            lfp.astype(float), [scale], wavelet, sampling_period=1.0 / fs
        )
        return np.abs(coeffs[0])

    n_samples, n_chan = lfp.shape
    envelope = np.empty((n_samples, n_chan), dtype=float)
    for c in range(n_chan):
        coeffs, _ = pywt.cwt(
            lfp[:, c].astype(float), [scale], wavelet, sampling_period=1.0 / fs
        )
        envelope[:, c] = np.abs(coeffs[0])
    return envelope


def compute_amplitude_envelopes_cwt(
    lfp: np.ndarray,
    fs: float,
    center_freqs: np.ndarray | list[float],
    n_cycles: float | np.ndarray = 7.0,
    gpu_batch_channels: int | None = None,
) -> np.ndarray:
    """Amplitude envelopes for multiple frequencies via the complex Morlet CWT.

    When ``n_cycles`` is a scalar, all frequencies share one FFT of the
    input signal (multi-scale CWT), which is substantially faster than
    calling :func:`compute_amplitude_envelope_cwt` once per frequency.
    GPU acceleration is used automatically when ``ptwt`` and CUDA are
    available.

    When ``n_cycles`` is an array (one value per frequency), each frequency
    uses its own wavelet shape.  In this case the multi-scale optimisation
    does not apply and one CWT call is made per (frequency, channel).  This
    matches the linearly-scaled convention of ``nf_dcwt.m`` (Rawls 2023)
    used in ``theta_phase_spectrogram.m``; use :func:`morlet_n_cycles` to
    build the array.

    Parameters
    ----------
    lfp : np.ndarray
        1-D (n_samples,) or 2-D (n_samples, n_channels).
    fs : float
        Sampling rate in Hz.
    center_freqs : array-like of float
        Target centre frequencies in Hz.
    n_cycles : float or np.ndarray
        Morlet wavelet cycles.  Pass a scalar for a fixed cycle count
        (default 7); pass an array of length ``len(center_freqs)`` for
        per-frequency scaling (e.g. from :func:`morlet_n_cycles`).
    gpu_batch_channels : int, optional
        Channels per GPU batch (scalar n_cycles only).  Reduce if you hit
        VRAM limits — each batch uses roughly
        ``n_freqs × batch × n_samples × 8`` bytes of VRAM.

    Returns
    -------
    np.ndarray
        Shape (n_samples, n_freqs) for 1-D input, or
        (n_samples, n_channels, n_freqs) for 2-D input.
        dtype is float32 on the GPU path, float64 on the CPU path.
    """
    center_freqs = np.asarray(center_freqs, dtype=float)

    # Per-frequency n_cycles: different wavelet per band; no multi-scale batching
    if np.ndim(n_cycles) > 0:
        n_cycles_arr = np.asarray(n_cycles, dtype=float)
        one_d = lfp.ndim == 1
        data = lfp[:, np.newaxis] if one_d else lfp
        n_samples, n_chan = data.shape
        n_freqs = len(center_freqs)
        envelopes = np.empty((n_samples, n_chan, n_freqs), dtype=float)
        for f_idx, (fc, nc) in enumerate(zip(center_freqs, n_cycles_arr)):
            wavelet_f = f"cmor{nc ** 2 / 2.0:.4f}-1.0"
            scale_f = pywt.scale2frequency(wavelet_f, 1.0, precision=8) * fs / fc
            for c in range(n_chan):
                coeffs, _ = pywt.cwt(
                    data[:, c].astype(float),
                    [scale_f],
                    wavelet_f,
                    sampling_period=1.0 / fs,
                )
                envelopes[:, c, f_idx] = np.abs(coeffs[0])
        return envelopes[:, 0, :] if one_d else envelopes

    # Scalar n_cycles: all frequencies share one wavelet → efficient multi-scale CWT
    wavelet = f"cmor{n_cycles ** 2 / 2.0:.4f}-1.0"
    ref = pywt.scale2frequency(wavelet, 1.0, precision=8) * fs
    scales = ref / center_freqs  # (n_freqs,)

    # GPU path: ptwt processes all channels × all scales in one CUDA kernel
    try:
        import ptwt  # noqa: PLC0415
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            one_d = lfp.ndim == 1
            data = lfp[:, np.newaxis] if one_d else lfp  # (n_samples, n_chan)
            n_samples, n_chan = data.shape
            n_freqs = len(scales)
            envelopes = np.empty((n_samples, n_chan, n_freqs), dtype=np.float32)
            batch = gpu_batch_channels or n_chan
            for start in range(0, n_chan, batch):
                end = min(start + batch, n_chan)
                chunk = torch.from_numpy(
                    data[:, start:end].T.astype("float32")
                ).cuda()  # (batch_ch, n_samples)
                coeffs, _ = ptwt.cwt(chunk, scales, wavelet)
                # coeffs: (n_freqs, batch_ch, n_samples) complex
                envelopes[:, start:end, :] = coeffs.abs().permute(2, 1, 0).cpu().numpy()
            return envelopes[:, 0, :] if one_d else envelopes
    except ImportError:
        pass

    # CPU fallback: pywt, one channel at a time
    if lfp.ndim == 1:
        coeffs, _ = pywt.cwt(
            lfp.astype(float), scales, wavelet, sampling_period=1.0 / fs
        )
        return np.abs(coeffs).T  # (n_samples, n_freqs)

    n_samples, n_chan = lfp.shape
    n_freqs = len(center_freqs)
    envelopes = np.empty((n_samples, n_chan, n_freqs), dtype=float)
    for c in range(n_chan):
        coeffs, _ = pywt.cwt(
            lfp[:, c].astype(float), scales, wavelet, sampling_period=1.0 / fs
        )
        envelopes[:, c, :] = np.abs(coeffs).T
    return envelopes


def compute_amplitude_envelope(
    lfp: np.ndarray,
    fs: float,
    freq_min: float,
    freq_max: float,
    method: str = "wavelet",
    n_cycles: float = 7.0,
) -> np.ndarray:
    """Amplitude envelope of a bandpass LFP signal.

    Parameters
    ----------
    lfp : np.ndarray
        1-D (n_samples,) or 2-D (n_samples, n_channels) LFP array.
    fs : float
        Sampling rate in Hz.
    freq_min, freq_max : float
        Band edges in Hz.  The wavelet is centred at (freq_min + freq_max) / 2.
    method : str
        ``'wavelet'`` (default) — complex Morlet CWT at the band centre; or
        ``'hilbert'`` — Butterworth bandpass filter followed by Hilbert transform.
    n_cycles : float
        Morlet wavelet cycles; only used when *method* = ``'wavelet'``.

    Returns
    -------
    np.ndarray
        Amplitude envelope, same shape as *lfp*.
    """
    if method == "wavelet":
        return compute_amplitude_envelope_cwt(
            lfp, fs, (freq_min + freq_max) / 2.0, n_cycles=n_cycles
        )
    # 'hilbert' (legacy)
    filtered = bandpass_filter_lfp(lfp, fs=fs, freq_min=freq_min, freq_max=freq_max)
    if filtered.ndim == 1:
        return np.abs(hilbert(filtered))
    envelope = np.empty_like(filtered, dtype=float)
    for c in range(filtered.shape[1]):
        envelope[:, c] = np.abs(hilbert(filtered[:, c]))
    return envelope


def compute_mean_amplitude_profile(
    theta_phase: np.ndarray, amplitude: np.ndarray, n_bins: int = 18
) -> tuple[np.ndarray, np.ndarray]:
    """Mean gamma amplitude as a function of theta phase bin.

    NaN values in *theta_phase* (bad cycles from get_signal_phase) are ignored.

    Parameters
    ----------
    theta_phase : np.ndarray
        1-D phase timeseries in [0, 2π] with NaN for excluded samples.
    amplitude : np.ndarray
        1-D amplitude envelope timeseries, same length.
    n_bins : int
        Number of phase bins (default 18 → 20° resolution).

    Returns
    -------
    bin_centers : np.ndarray, shape (n_bins,)
        Centre of each phase bin in radians.
    mean_amp : np.ndarray, shape (n_bins,)
        Mean amplitude per bin.  Bins with no valid samples are NaN.
    """
    bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    valid = np.isfinite(theta_phase) & np.isfinite(amplitude)
    phase_v = theta_phase[valid]
    amp_v = amplitude[valid]

    mean_amp = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (phase_v >= bin_edges[i]) & (phase_v < bin_edges[i + 1])
        if mask.sum() > 0:
            mean_amp[i] = amp_v[mask].mean()

    return bin_centers, mean_amp


# ---------------------------------------------------------------------------
# PAC metrics
# ---------------------------------------------------------------------------


def compute_pac_mi(
    theta_phase: np.ndarray, amplitude: np.ndarray, n_bins: int = 18
) -> float:
    """Modulation Index (Tort et al. 2010).

    MI = (log N − H(p)) / log N
    where p is the amplitude distribution across phase bins (normalised to sum=1)
    and H is its Shannon entropy.  Range [0, 1].

    Parameters
    ----------
    theta_phase : np.ndarray
        1-D phase timeseries in [0, 2π], NaN = excluded.
    amplitude : np.ndarray
        1-D amplitude envelope, same length.
    n_bins : int
        Number of phase bins.

    Returns
    -------
    float
        MI value.  Returns NaN if fewer than n_bins valid samples.
    """
    _, mean_amp = compute_mean_amplitude_profile(theta_phase, amplitude, n_bins)

    if np.all(np.isnan(mean_amp)):
        return np.nan

    # Fill any empty bins with 0 before normalising
    mean_amp = np.where(np.isnan(mean_amp), 0.0, mean_amp)
    total = mean_amp.sum()
    if total == 0:
        return np.nan

    p = mean_amp / total
    # Shannon entropy (avoid log(0))
    p_safe = np.where(p > 0, p, 1.0)
    H = -np.sum(p * np.log(p_safe))  # noqa: N806
    H_max = np.log(n_bins)  # noqa: N806
    return float((H_max - H) / H_max)


def compute_pac_mvl(
    theta_phase: np.ndarray, amplitude: np.ndarray
) -> tuple[float, float]:
    """Mean Vector Length (Canolty et al. 2006).

    z = mean(A * exp(i * φ));  MVL = |z|,  preferred_phase = angle(z)

    Parameters
    ----------
    theta_phase : np.ndarray
        1-D phase timeseries in [0, 2π], NaN = excluded.
    amplitude : np.ndarray
        1-D amplitude envelope, same length.

    Returns
    -------
    mvl : float
        Mean vector length (unnormalised; units of amplitude).
    preferred_phase : float
        Preferred theta phase in [0, 2π] radians.
    """
    valid = np.isfinite(theta_phase) & np.isfinite(amplitude)
    if valid.sum() < 2:
        return np.nan, np.nan

    z = np.mean(amplitude[valid] * np.exp(1j * theta_phase[valid]))
    mvl = float(np.abs(z))
    preferred_phase = float(np.angle(z) % (2 * np.pi))
    return mvl, preferred_phase


def compute_pac_mvl_normalised(
    theta_phase: np.ndarray, amplitude: np.ndarray
) -> tuple[float, float]:
    """Normalised MVL: divide by mean amplitude to compare across sessions.

    Returns (normalised_mvl, preferred_phase_rad).
    """
    valid = np.isfinite(theta_phase) & np.isfinite(amplitude)
    if valid.sum() < 2:
        return np.nan, np.nan

    amp_v = amplitude[valid]
    mean_amp = amp_v.mean()
    if mean_amp == 0:
        return np.nan, np.nan

    z = np.mean((amp_v / mean_amp) * np.exp(1j * theta_phase[valid]))
    return float(np.abs(z)), float(np.angle(z) % (2 * np.pi))


# ---------------------------------------------------------------------------
# Surrogate statistics
# ---------------------------------------------------------------------------


def compute_pac_surrogates(
    theta_phase: np.ndarray,
    amplitude: np.ndarray,
    n_surrogates: int = 200,
    method: str = "time_shift",
    min_shift_s: float = 1.0,
    fs: float = 1000.0,
    metric: str = "mi",
    n_bins: int = 18,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Surrogate PAC distribution via random time-shifting of the amplitude envelope.

    Parameters
    ----------
    theta_phase : np.ndarray
        1-D phase timeseries in [0, 2π], NaN = excluded.
    amplitude : np.ndarray
        1-D amplitude envelope, same length.
    n_surrogates : int
        Number of surrogates to generate.
    method : str
        ``'time_shift'``: roll amplitude by a random lag ≥ min_shift_s.
    min_shift_s : float
        Minimum shift in seconds to avoid near-original surrogates.
    fs : float
        Sampling rate (needed for min_shift_s → samples conversion).
    metric : str
        ``'mi'`` or ``'mvl'`` — which PAC metric to compute on surrogates.
    n_bins : int
        Phase bins (only used when metric='mi').
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray, shape (n_surrogates,)
        Surrogate PAC values.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(amplitude)
    min_shift = int(min_shift_s * fs)
    max_shift = n - min_shift
    if max_shift <= min_shift:
        raise ValueError(
            f"Signal too short for min_shift_s={min_shift_s} s at fs={fs} Hz."
        )

    shifts = rng.integers(min_shift, max_shift, size=n_surrogates)

    surrogates = np.empty(n_surrogates)
    for k, shift in enumerate(shifts):
        amp_shifted = np.roll(amplitude, int(shift))
        if metric == "mi":
            surrogates[k] = compute_pac_mi(theta_phase, amp_shifted, n_bins)
        else:
            surrogates[k], _ = compute_pac_mvl_normalised(theta_phase, amp_shifted)

    return surrogates


def compute_pac_zscore(observed: float, surrogates: np.ndarray) -> tuple[float, float]:
    """Z-score and p-value of observed PAC against surrogate distribution.

    Parameters
    ----------
    observed : float
        Observed PAC value.
    surrogates : np.ndarray
        Surrogate distribution from compute_pac_surrogates().

    Returns
    -------
    z_score : float
    p_value : float
        Proportion of surrogates ≥ observed (one-tailed).
    """
    surr = surrogates[np.isfinite(surrogates)]
    if len(surr) == 0 or not np.isfinite(observed):
        return np.nan, np.nan

    mu, sigma = surr.mean(), surr.std()
    z = float((observed - mu) / sigma) if sigma > 0 else np.nan
    p = float((surr >= observed).mean())
    return z, p


# ---------------------------------------------------------------------------
# Depth-profile batch helper
# ---------------------------------------------------------------------------


def compute_pac_depth_profile(
    theta_phase: np.ndarray,
    lfp_column: np.ndarray,
    fs: float,
    gamma_bands: dict[str, tuple[float, float]] | None = None,
    n_bins: int = 18,
    n_surrogates: int = 200,
    speed_mask: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    amplitude_method: str = "wavelet",
    n_cycles: float | str = "linear",
) -> list[dict]:
    """Compute PAC (MI, normalised MVL, z-score, preferred phase) for every
    combination of (channel, gamma_band) in a CA1 column LFP array.

    Parameters
    ----------
    theta_phase : np.ndarray, shape (n_samples,)
        Theta phase from the reference (pyramidal) channel.  NaN = bad cycles.
    lfp_column : np.ndarray, shape (n_samples, n_channels)
        Raw LFP for all channels in the CA1 column.
    fs : float
        Sampling rate in Hz.
    gamma_bands : dict, optional
        Band name → (freq_min, freq_max).  Defaults to GAMMA_BANDS.
    n_bins : int
        Phase bins for MI.
    n_surrogates : int
        Surrogates per (channel, band) pair.  Set 0 to skip statistics.
    speed_mask : np.ndarray of bool, shape (n_samples,), optional
        True for samples to *include* (e.g. running periods).  Applied to both
        theta_phase and amplitude before computing PAC.
    rng : np.random.Generator, optional
        For reproducible surrogate generation.
    amplitude_method : str
        Passed to :func:`compute_amplitude_envelope`.  ``'wavelet'`` (default)
        or ``'hilbert'``.
    n_cycles : float or ``'linear'``
        Morlet wavelet cycles; only used when *amplitude_method* = ``'wavelet'``.
        ``'linear'`` (default) uses :func:`morlet_n_cycles` to scale cycles
        linearly with the band centre frequency (3 cycles at 20 Hz → 8 at
        150 Hz), matching the ``nf_dcwt.m`` / ``theta_phase_spectrogram.m``
        convention.  Pass a float for a fixed cycle count.

    Returns
    -------
    list of dict
        One dict per (channel_idx, band_name) with keys:
        channel_idx, band, freq_min, freq_max, mi, mvl, mvl_norm,
        preferred_phase, z_score, p_value, n_valid_samples.
    """
    if gamma_bands is None:
        gamma_bands = GAMMA_BANDS

    if rng is None:
        rng = np.random.default_rng()

    n_samples, n_channels = lfp_column.shape

    # Apply speed mask
    if speed_mask is not None:
        theta_use = theta_phase.copy()
        theta_use[~speed_mask] = np.nan
    else:
        theta_use = theta_phase

    results = []
    for band_name, (fmin, fmax) in gamma_bands.items():
        fc = (fmin + fmax) / 2.0
        nc = morlet_n_cycles(fc) if n_cycles == "linear" else float(n_cycles)
        # Compute amplitude envelope for all channels at once
        envelope = compute_amplitude_envelope(
            lfp_column, fs, fmin, fmax, method=amplitude_method, n_cycles=nc
        )

        for ch in range(n_channels):
            amp = envelope[:, ch]

            # Apply speed mask to amplitude too
            if speed_mask is not None:
                amp = amp.copy()
                amp[~speed_mask] = np.nan

            n_valid = int(np.sum(np.isfinite(theta_use) & np.isfinite(amp)))

            mi = compute_pac_mi(theta_use, amp, n_bins)
            mvl, preferred_phase = compute_pac_mvl(theta_use, amp)
            mvl_norm, _ = compute_pac_mvl_normalised(theta_use, amp)

            if n_surrogates > 0:
                surr = compute_pac_surrogates(
                    theta_use,
                    amp,
                    n_surrogates=n_surrogates,
                    fs=fs,
                    metric="mi",
                    n_bins=n_bins,
                    rng=rng,
                )
                z, p = compute_pac_zscore(mi, surr)
            else:
                z, p = np.nan, np.nan

            results.append(
                {
                    "channel_idx": ch,
                    "band": band_name,
                    "freq_min": fmin,
                    "freq_max": fmax,
                    "mi": mi,
                    "mvl": mvl,
                    "mvl_norm": mvl_norm,
                    "preferred_phase": preferred_phase,
                    "z_score": z,
                    "p_value": p,
                    "n_valid_samples": n_valid,
                }
            )

    return results
