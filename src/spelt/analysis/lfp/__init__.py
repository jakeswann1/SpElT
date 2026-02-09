"""LFP (Local Field Potential) analysis utilities."""

from .csd import bz_csd, calculate_csd_df
from .decomposition import calc_instantaneous_info, eemd
from .depth_profiles import (
    compute_conditional_depth_power_profile,
    compute_depth_power_profile,
)
from .event_locked import extract_event_locked_windows, pool_windows_across_trials
from .event_locked_csd import compute_event_locked_csd
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
from .utils import apply_common_reference

__all__ = [
    # Preprocessing
    "apply_common_reference",
    # Filtering
    "bandpass_filter_lfp",
    "get_filter_frequencies",
    # Frequency power
    "compute_band_power",
    "compute_band_power_from_ephys",
    "compute_band_power_single_channel",
    # Depth profiles
    "compute_depth_power_profile",
    "compute_conditional_depth_power_profile",
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
    "compute_event_locked_csd",
    # Decomposition
    "eemd",
    "calc_instantaneous_info",
    # Ripple detection
    "detect_ripples",
]
