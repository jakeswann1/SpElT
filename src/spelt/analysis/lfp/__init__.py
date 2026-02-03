"""LFP (Local Field Potential) analysis utilities."""

from .csd import (
    bz_csd,
    calculate_csd_df,
    compute_event_locked_csd,
    compute_phase_binned_csd,
    mean_csd_theta_phase,
    plot_csd_theta_phase,
    plot_event_locked_csd,
)
from .decomposition import calc_instantaneous_info, eemd
from .filtering import bandpass_filter_lfp, get_filter_frequencies
from .frequency_power import (
    compute_band_power,
    compute_band_power_from_ephys,
    compute_band_power_single_channel,
)
from .peak_frequencies import find_peak_frequency, get_theta_frequencies
from .phase import get_signal_phase, get_spike_phase
from .ripple_detection import detect_ripples

__all__ = [
    # Filtering
    "bandpass_filter_lfp",
    "get_filter_frequencies",
    # Frequency power
    "compute_band_power",
    "compute_band_power_from_ephys",
    "compute_band_power_single_channel",
    # Phase
    "get_signal_phase",
    "get_spike_phase",
    # Peak frequencies
    "find_peak_frequency",
    "get_theta_frequencies",
    # CSD
    "bz_csd",
    "calculate_csd_df",
    "compute_event_locked_csd",
    "compute_phase_binned_csd",
    "mean_csd_theta_phase",
    "plot_csd_theta_phase",
    "plot_event_locked_csd",
    # Decomposition
    "eemd",
    "calc_instantaneous_info",
    # Ripple detection
    "detect_ripples",
]
