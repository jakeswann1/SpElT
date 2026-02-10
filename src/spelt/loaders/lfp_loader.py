"""LFP data loading and validation functions."""

import shutil
from pathlib import Path

import spikeinterface as si
import spikeinterface.preprocessing as spre


def load_lfp_data(
    recording: si.BaseRecording,
    segment_index: int,
    sampling_rate: int,
    recording_type: str,
    temp_folder: Path,
    channels: list[int] | None = None,
    bandpass_filter: tuple[float, float] | None = None,
) -> dict:
    """
    Load and process LFP data from a recording segment.

    Parameters
    ----------
    recording : si.BaseRecording
        SpikeInterface recording object
    segment_index : int
        Trial/segment index to load
    sampling_rate : int
        Desired sampling rate (Hz)
    recording_type : str
        Recording system type ('nexus', 'NP2_openephys', 'NP2_onebox')
    temp_folder : Path
        Path for temporary files
    channels : list[int], optional
        Channel IDs to load (None = all)
    bandpass_filter : tuple[float, float], optional
        Bandpass filter range [min_hz, max_hz]

    Returns
    -------
    dict
        LFP data with keys:
        - 'data': (time, channels) array
        - 'timestamps': Time values per sample (absolute)
        - 'timestamps_relative': Time relative to trial start
        - 'sampling_rate': Sampling rate
        - 'channels': Channel IDs
        - 'filter_range': Applied filter [min, max]

    Raises
    ------
    Exception
        If LFP loading fails
    """
    try:
        # Apply bandpass filter if specified
        if bandpass_filter is not None:
            recording = spre.bandpass_filter(
                recording, freq_min=bandpass_filter[0], freq_max=bandpass_filter[1]
            )

        # AXONA ONLY: clip values of >+- 32000
        if recording_type == "nexus":
            recording = spre.clip(recording, a_min=-32000, a_max=32000)

        # Resample
        recording: si.BaseRecording = spre.resample(recording, sampling_rate)
        print(f"Resampled to {sampling_rate} Hz")

        # Convert channel IDs to strings to match recording object
        if channels is not None:
            channels = list(map(str, channels))

        # Create temporary recording object on disk
        recording.save(format="zarr", folder=temp_folder)
        recording = recording.load(f"{temp_folder}.zarr")

        # Extract LFP traces
        lfp_data = recording.get_traces(
            segment_index=segment_index, channel_ids=channels, return_scaled=True
        ).astype(float)

        # Get timestamps
        lfp_timestamps = recording.get_times(segment_index=segment_index)
        lfp_timestamps_relative = lfp_timestamps - lfp_timestamps[0]

        # If no channels specified, get all channel IDs from recording
        if channels is None:
            channels = [str(ch) for ch in recording.get_channel_ids()]

        return {
            "data": lfp_data,
            "timestamps": lfp_timestamps,  # Absolute time
            "timestamps_relative": lfp_timestamps_relative,  # Relative to trial start
            "sampling_rate": sampling_rate,
            "channels": channels,
            "filter_range": bandpass_filter,
        }

    finally:
        # Clean up temporary files
        try:
            zarr_path = Path(f"{temp_folder}.zarr")
            if zarr_path.exists():
                shutil.rmtree(zarr_path)
                print(f"Cleaned up temporary recording: {zarr_path}")
        except Exception as e:
            print(f"Warning: Could not clean up temporary files: {e}")


def validate_lfp_cache(
    cached_data: dict,
    requested_sampling_rate: int,
    requested_channels: list[int] | None,
    requested_filter: tuple[float, float] | None,
) -> tuple[bool, str]:
    """
    Validate that cached LFP data matches requested parameters.

    Parameters
    ----------
    cached_data : dict
        Cached LFP data dictionary
    requested_sampling_rate : int
        Requested sampling rate (Hz)
    requested_channels : list[int], optional
        Requested channel IDs
    requested_filter : tuple[float, float], optional
        Requested bandpass filter range [min, max]

    Returns
    -------
    is_valid : bool
        True if cache can be used
    reason : str
        Reason for invalidation (empty if valid)
    """
    # Check sampling rate
    saved_sampling_rate = cached_data.get("sampling_rate")
    if (
        requested_sampling_rate is not None
        and saved_sampling_rate is not None
        and saved_sampling_rate != requested_sampling_rate
    ):
        return (
            False,
            f"Sampling rate mismatch: requested {requested_sampling_rate} Hz, "
            f"cached {saved_sampling_rate} Hz",
        )

    # Check filter range
    saved_filter = cached_data.get("filter_range")
    if (
        requested_filter is not None
        and saved_filter is not None
        and saved_filter != requested_filter
    ):
        return (
            False,
            f"Filter mismatch: requested {requested_filter}, cached {saved_filter}",
        )

    # Check if requested channels are available (if specified)
    if requested_channels is not None:
        available_channels = cached_data.get("channels")
        if available_channels is not None:
            available_channels = [int(ch) for ch in available_channels]
            missing = set(requested_channels) - set(available_channels)
            if missing:
                return (False, f"Requested channels {missing} not in cached data")

    return True, ""


def subset_lfp_channels(lfp_data: dict, requested_channels: list[int]) -> dict:
    """
    Extract specific channels from LFP data.

    Parameters
    ----------
    lfp_data : dict
        Full LFP data
    requested_channels : list[int]
        Channel IDs to extract

    Returns
    -------
    dict
        LFP data with only requested channels

    Raises
    ------
    ValueError
        If requested channels not found in data
    """
    available_channels = lfp_data.get("channels")

    if available_channels is None:
        # If no channel info, assume channels are in order
        available_channels = list(range(lfp_data["data"].shape[1]))
    else:
        available_channels = [int(ch) for ch in available_channels]

    # Find indices of requested channels
    try:
        channel_indices = [available_channels.index(ch) for ch in requested_channels]
    except ValueError as e:
        missing_channels = [
            ch for ch in requested_channels if ch not in available_channels
        ]
        raise ValueError(
            f"Requested channels {missing_channels} not found in available data "
            f"(available: {available_channels})."
        ) from e

    # Create a copy and subset the data
    subset_data = lfp_data.copy()
    subset_data["data"] = lfp_data["data"][:, channel_indices]
    subset_data["channels"] = [str(ch) for ch in requested_channels]

    # Subset theta phase data if it exists
    if "theta_phase" in lfp_data:
        subset_data["theta_phase"] = lfp_data["theta_phase"][:, channel_indices]
        subset_data["cycle_numbers"] = lfp_data["cycle_numbers"][:, channel_indices]
        subset_data["theta_freqs"] = {
            ch: lfp_data["theta_freqs"][ch]
            for ch in requested_channels
            if ch in lfp_data["theta_freqs"]
        }

    return subset_data


def has_requested_channels(lfp_data: dict, requested_channels: list[int]) -> bool:
    """
    Check if LFP data already contains all requested channels.

    Parameters
    ----------
    lfp_data : dict
        LFP data dictionary
    requested_channels : list[int]
        Requested channel IDs

    Returns
    -------
    bool
        True if all requested channels are present
    """
    if lfp_data is None:
        return False

    available_channels = lfp_data.get("channels")
    if available_channels is None:
        return False

    available_channels = [int(ch) for ch in available_channels]
    return all(ch in available_channels for ch in requested_channels)
