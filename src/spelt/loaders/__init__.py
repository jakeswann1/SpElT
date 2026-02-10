"""Data loading utilities for ephys data.

This package contains modular loading functions for different data types:
- cache: Pickle file utilities
- position_loader: Position/tracking data loading
- lfp_loader: LFP data loading and validation
- theta_loader: Theta phase computation
- ttl_loader: TTL/sync data loading
"""

from .cache import get_cache_path, load_pickle, save_pickle
from .lfp_loader import (
    has_requested_channels,
    load_lfp_data,
    subset_lfp_channels,
    validate_lfp_cache,
)
from .position_loader import (
    load_axona_position,
    load_bonsai_leds_position,
    load_bonsai_roi_position,
    load_dlc_position,
    load_position_data,
)
from .theta_loader import add_theta_phase_to_lfp, compute_theta_phase
from .ttl_loader import get_ttl_frequency, load_ttl_data

__all__ = [
    # Cache utilities
    "get_cache_path",
    "load_pickle",
    "save_pickle",
    # Position loaders
    "load_position_data",
    "load_axona_position",
    "load_bonsai_roi_position",
    "load_bonsai_leds_position",
    "load_dlc_position",
    # LFP loaders
    "load_lfp_data",
    "validate_lfp_cache",
    "subset_lfp_channels",
    "has_requested_channels",
    # Theta loaders
    "compute_theta_phase",
    "add_theta_phase_to_lfp",
    # TTL loaders
    "load_ttl_data",
    "get_ttl_frequency",
]
