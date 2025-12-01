import numpy as np
import pandas as pd


def get_traversal_cycles(
    arm_times: np.ndarray,
    cycle_numbers: np.ndarray,
    lfp_timestamps: np.ndarray,
    lfp_sampling_rate: float,
    gap_threshold: float = 1,
    min_segment_size: int = 10,
) -> list[np.ndarray]:
    """
    Efficiently finds theta cycle IDs for individual arm traversals.

    Parameters:
    -----------
    arm_times : np.ndarray
        Timestamps of arm traversal samples
    cycle_numbers : np.ndarray
        Theta cycle numbers for each LFP timestamp
    lfp_timestamps : np.ndarray
        Timestamps of LFP samples
    lfp_sampling_rate : float
        Sampling rate of LFP in Hz
    gap_threshold : float, optional
        Minimum time gap (in seconds) to consider separate traversals
    min_segment_size : int, optional
        Minimum number of samples needed for a valid segment

    Returns:
    --------
    List[np.ndarray]
        List of arrays, each containing theta cycle IDs for a single traversal
    """
    # Early return for empty input
    if len(arm_times) == 0:
        return []

    # Sort arm_times to ensure chronological order
    if not np.all(np.diff(arm_times) >= 0):
        arm_times = np.sort(arm_times)

    # Find gaps in arm times that exceed the threshold
    gaps = np.diff(arm_times) > gap_threshold
    segment_indices = np.where(gaps)[0] + 1

    # Create segments
    segments = []
    start_idx = 0

    for end_idx in segment_indices:
        segment = arm_times[start_idx:end_idx]
        if len(segment) >= min_segment_size:
            segments.append((segment[0], segment[-1]))
        start_idx = end_idx

    # Add the last segment if it exists and meets minimum size
    if start_idx < len(arm_times) and len(arm_times) - start_idx >= min_segment_size:
        segments.append((arm_times[start_idx], arm_times[-1]))

    # No valid segments found
    if not segments:
        return []

    # Convert times to sample indices (all at once)
    segments_array = np.array(segments)
    sample_indices = np.floor(segments_array * lfp_sampling_rate).astype(np.int32)

    # Find cycle numbers for each traversal
    arm_cycle_numbers = []
    for start_idx, end_idx in sample_indices:
        # Safety check for bounds
        start_idx = max(0, start_idx)
        end_idx = min(len(cycle_numbers) - 1, end_idx)

        # Get unique cycle numbers in this window, maintaining order
        # Use np.unique with return_index to preserve order
        traversal_cycle_nums = cycle_numbers[start_idx : end_idx + 1]
        if len(traversal_cycle_nums) > 0:
            unique_cycles, idx = np.unique(traversal_cycle_nums, return_index=True)
            sorted_idx = np.argsort(idx)
            traversal_cycle_numbers = unique_cycles[sorted_idx]

            # Only trim if we have enough cycles
            if len(traversal_cycle_numbers) > 2:
                traversal_cycle_numbers = traversal_cycle_numbers[1:-1]

            # Add non-empty cycle lists
            if len(traversal_cycle_numbers) > 0:
                arm_cycle_numbers.append(traversal_cycle_numbers)

    return arm_cycle_numbers


def get_data_for_traversals(
    arm_traversal_cycles,
    cycle_numbers,
    lfp_data,
    speed_data,
    channels_to_load,
    theta_phase,
    lfp_timestamps,
):
    """
    Makes a dataframe of LFP data from all traversals in a given arm.
    Columns are timestamps of LFP data samples
    Indices are:
    - Channel IDs (containing LFP data samples, one row for each channel loaded)
    - Cycle Theta Phase (taken from channel 35 at 0um)
    - Cycle Index (index of theta cycle for each LFP sample
    - Speed data interpolated up to match LFP sampling rate
    - Traversal Index (index of traversal in that arm, counting from 0 for each trial)
    """
    # Convert channels to strings to preserve ordering by depth
    channel_str_ids = [str(ch) for ch in channels_to_load]

    # Handle empty traversals case
    if not arm_traversal_cycles or all(len(t) == 0 for t in arm_traversal_cycles):
        return pd.DataFrame(
            index=channel_str_ids
            + ["Cycle Theta Phase", "Cycle Index", "Traversal Index", "Speed"]
        )

    # Create a list to collect all valid data columns
    data_columns = []
    column_timestamps = []

    # Process each traversal
    for traversal_idx, traversal_cycles in enumerate(arm_traversal_cycles):
        # Skip empty traversals
        if len(traversal_cycles) == 0:
            continue

        # Find all timestamps for this traversal's cycles
        mask = np.isin(cycle_numbers, traversal_cycles).flatten()

        # Skip if no matching samples
        if not np.any(mask):
            continue

        # Extract masked data all at once
        selected_timestamps = lfp_timestamps[mask]
        selected_phases = theta_phase[mask].flatten()
        selected_cycles = cycle_numbers[mask].flatten()
        selected_speeds = speed_data[mask]
        selected_lfp = lfp_data[mask].T

        # Create traversal index array
        traversal_indices = np.full(len(selected_timestamps), traversal_idx)

        # For each matching timestamp, build a column for our final DataFrame
        for i in range(len(selected_timestamps)):
            ts = selected_timestamps[i]
            column_timestamps.append(ts)

            # Build a complete column including LFP and metadata
            column_data = {
                "Cycle Theta Phase": selected_phases[i],
                "Cycle Index": selected_cycles[i],
                "Traversal Index": traversal_indices[i],
                "Speed": selected_speeds[i],
            }

            # Add LFP values for each channel
            for ch_idx, ch_str in enumerate(channel_str_ids):
                if ch_idx < selected_lfp.shape[0]:
                    column_data[ch_str] = selected_lfp[ch_idx, i]
                else:
                    column_data[ch_str] = np.nan

            data_columns.append(column_data)

    # If we collected no data, return empty DataFrame with correct structure
    if not data_columns:
        return pd.DataFrame(
            index=channel_str_ids
            + ["Cycle Theta Phase", "Cycle Index", "Traversal Index", "Speed"]
        )

    # Create DataFrame with timestamps as columns and channels/metadata as rows
    df = pd.DataFrame(data_columns, index=column_timestamps).T

    # Ensure all expected rows exist
    for idx in channel_str_ids + [
        "Cycle Theta Phase",
        "Cycle Index",
        "Traversal Index",
        "Speed",
    ]:
        if idx not in df.index:
            df.loc[idx] = np.nan
    return df


def drop_extreme_cycles(df):
    """
    Drops columns with the lowest and highest 'Cycle Index' value for each unique 'Traversal Index'.

    Args:
        df (pd.DataFrame): DataFrame with 'Cycle Index' and 'Traversal Index' as rows.

    Returns:
        pd.DataFrame: A new DataFrame with specified columns dropped.
    """  # noqa E501

    # Find the rows for 'Cycle Index' and 'Traversal Index'
    cycle_index_row = df.index[df.index == "Cycle Index"][0]
    traversal_index_row = df.index[df.index == "Traversal Index"][0]

    # Convert these rows to columns for easy processing
    df_transposed = df.T
    df_transposed["Cycle Index"] = df_transposed[cycle_index_row]
    df_transposed["Traversal Index"] = df_transposed[traversal_index_row]

    # Group by 'Traversal Index' and find columns to drop
    columns_to_drop = set()
    for _, group in df_transposed.groupby("Traversal Index"):
        # Find min and max cycle number in each group
        cycle_min = np.nanmin(np.unique(group["Cycle Index"]))
        cycle_max = np.nanmax(np.unique(group["Cycle Index"]))

        # Find columns where 'Cycle Index' is equal to cycle_min or cycle_max
        cols_to_drop = group[
            (group["Cycle Index"] == cycle_min) | (group["Cycle Index"] == cycle_max)
        ].index
        columns_to_drop.update(cols_to_drop)

    # Drop the identified columns and the added rows
    df_dropped = df_transposed.drop(index=list(columns_to_drop))

    # Transpose back to the original format
    return df_dropped.T
