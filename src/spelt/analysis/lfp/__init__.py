"""LFP (Local Field Potential) analysis utilities."""

from .csd import bz_csd, calculate_csd_df
from .decomposition import calc_instantaneous_info, eemd
from .event_locked import extract_event_locked_windows, pool_windows_across_trials
from .filtering import bandpass_filter_lfp, get_filter_frequencies
from .frequency_power import (
    compute_band_power,
    compute_band_power_from_ephys,
    compute_band_power_single_channel,
)
from .peak_frequencies import find_peak_frequency, get_theta_frequencies
from .phase import (
    compute_relative_phase,
    detect_phase_crossings,
    get_signal_phase,
    get_spike_phase,
)
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
    "detect_phase_crossings",
    "compute_relative_phase",
    # Peak frequencies
    "find_peak_frequency",
    "get_theta_frequencies",
    # CSD
    "bz_csd",
    "calculate_csd_df",
    # Event-locked analysis
    "extract_event_locked_windows",
    "pool_windows_across_trials",
    # Decomposition
    "eemd",
    "calc_instantaneous_info",
    # Ripple detection
    "detect_ripples",
]
