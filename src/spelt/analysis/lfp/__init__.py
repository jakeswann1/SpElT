"""LFP (Local Field Potential) analysis utilities."""

from .csd import bz_csd, calculate_csd_df, mean_csd_theta_phase, plot_csd_theta_phase
from .decomposition import calc_instantaneous_info, eemd
from .filtering import bandpass_filter_lfp, get_filter_frequencies
from .frequency_power import (
    compute_band_power,
    compute_band_power_from_ephys,
    compute_band_power_single_channel,
)
from .peak_frequencies import find_peak_frequency, get_theta_frequencies
from .phase import get_signal_phase, get_spike_phase

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
    "mean_csd_theta_phase",
    "plot_csd_theta_phase",
    # Decomposition
    "eemd",
    "calc_instantaneous_info",
]
