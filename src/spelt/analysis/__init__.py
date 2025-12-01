"""Analysis utilities for ephys data.

This module is organized into submodules by analysis type:
- lfp: Local field potential analysis (filtering, power, phase, CSD, decomposition)
- spatial: Spatial analysis (information, correlation, significance)
- phase_locking: Phase locking and circular statistics (precession, correlation)
- behavioral: Behavioral trajectory analysis (traversals)
- t_maze: T-maze specific analysis
- linear_track: Linear track specific analysis
"""

# Import submodules for convenient access
from . import behavioral, lfp, linear_track, phase_locking, spatial, t_maze
from .behavioral import get_data_for_traversals, get_traversal_cycles

# Re-export commonly used functions at top level for backwards compatibility
from .lfp import (
    bandpass_filter_lfp,
    compute_band_power,
    compute_band_power_from_ephys,
    compute_band_power_single_channel,
    get_filter_frequencies,
    get_signal_phase,
    get_spike_phase,
)
from .phase_locking import binned_cl_corr, cl_corr, rayleigh_vector
from .spatial import spatial_correlation, spatial_info, spatial_significance

__all__ = [
    # Submodules
    "lfp",
    "spatial",
    "phase_locking",
    "behavioral",
    "t_maze",
    "linear_track",
    # LFP functions (backwards compatibility)
    "bandpass_filter_lfp",
    "get_filter_frequencies",
    "compute_band_power",
    "compute_band_power_from_ephys",
    "compute_band_power_single_channel",
    "get_signal_phase",
    "get_spike_phase",
    # Spatial functions (backwards compatibility)
    "spatial_info",
    "spatial_correlation",
    "spatial_significance",
    # Phase locking functions (backwards compatibility)
    "rayleigh_vector",
    "binned_cl_corr",
    "cl_corr",
    # Behavioral functions (backwards compatibility)
    "get_traversal_cycles",
    "get_data_for_traversals",
]
