"""Theta phase computation functions."""

import numpy as np


def compute_theta_phase(
    lfp_data: np.ndarray,
    sampling_rate: float,
    channels: list[int] | None = None,
    clip_value: float | None = 32000,
) -> dict:
    """
    Compute theta phase from LFP data.

    Parameters
    ----------
    lfp_data : np.ndarray
        LFP data array with shape (time, channels)
    sampling_rate : float
        Sampling rate in Hz
    channels : list[int], optional
        Channel IDs corresponding to data columns (for labeling)
        If None, channels are numbered 0, 1, 2, ...
    clip_value : float, optional
        Value to clip for Axona recordings (default: 32000)
        Set to None to disable clipping

    Returns
    -------
    dict
        Theta phase data with keys:
        - 'theta_phase': Phase values per channel (time, channels)
        - 'cycle_numbers': Theta cycle numbers (time, channels)
        - 'theta_freqs': Peak frequencies per channel (dict)
    """
    from spelt.analysis.lfp import get_signal_phase, get_theta_frequencies

    # Use sequential channel numbering if not specified
    if channels is None:
        channels = list(range(lfp_data.shape[1]))

    # Find peak theta frequencies for all channels
    theta_freqs = get_theta_frequencies(lfp_data, sampling_rate)
    theta_freqs_dict = dict(zip(channels, theta_freqs))

    # Calculate theta phase for all channels
    theta_phase, cycle_numbers = get_signal_phase(
        lfp_data, sampling_rate, peak_freq=theta_freqs, clip_value=clip_value
    )

    return {
        "theta_phase": theta_phase,
        "cycle_numbers": cycle_numbers,
        "theta_freqs": theta_freqs_dict,
    }


def add_theta_phase_to_lfp(
    lfp_data_dict: dict, clip_value: float | None = 32000
) -> dict:
    """
    Add theta phase data to an existing LFP data dictionary.

    Parameters
    ----------
    lfp_data_dict : dict
        LFP data dictionary with 'data', 'sampling_rate', and 'channels' keys
    clip_value : float, optional
        Value to clip for Axona recordings (default: 32000)

    Returns
    -------
    dict
        Updated LFP data dictionary with theta phase data added
    """
    lfp_array = lfp_data_dict["data"]
    sampling_rate = lfp_data_dict["sampling_rate"]
    available_channels = lfp_data_dict.get("channels")

    # Parse channels
    if available_channels:
        channels = [int(ch) for ch in available_channels]
    else:
        channels = list(range(lfp_array.shape[1]))

    # Compute theta phase
    theta_data = compute_theta_phase(lfp_array, sampling_rate, channels, clip_value)

    # Update the dictionary
    lfp_data_dict.update(theta_data)

    return lfp_data_dict
