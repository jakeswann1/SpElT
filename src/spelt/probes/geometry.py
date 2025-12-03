"""Probe geometry utilities for channel selection and spatial analysis."""

import numpy as np


def _extract_channel_locations(probe_or_locations):
    """
    Helper to extract (n_channels, 2) array from various inputs.

    Parameters:
    -----------
    probe_or_locations : np.ndarray, ProbeInterface Probe, or SpikeInterface Recording
        Source of channel location data. Can be:
        - np.ndarray: (n_channels, 2) array of [x, y] positions
        - ProbeInterface Probe object with .contact_positions attribute
        - SpikeInterface Recording with .get_channel_locations() method

    Returns:
    --------
    np.ndarray
        (n_channels, 2) array of [x, y] positions in μm

    Raises:
    -------
    TypeError
        If input type is not recognized
    ValueError
        If array shape is invalid
    """
    # Case 1: Already a numpy array
    if isinstance(probe_or_locations, np.ndarray):
        if probe_or_locations.ndim != 2 or probe_or_locations.shape[1] != 2:
            raise ValueError(
                f"Channel locations array must have shape (n_channels, 2), "
                f"got {probe_or_locations.shape}"
            )
        return probe_or_locations

    # Case 2: ProbeInterface Probe object
    if hasattr(probe_or_locations, "contact_positions"):
        return probe_or_locations.contact_positions

    # Case 3: SpikeInterface Recording object
    if hasattr(probe_or_locations, "get_channel_locations"):
        return probe_or_locations.get_channel_locations()

    raise TypeError(
        f"Cannot extract channel locations from type {type(probe_or_locations)}. "
        "Expected numpy array, ProbeInterface Probe, or SpikeInterface Recording."
    )


def get_longest_column(channel_locations, return_all_info=False):
    """
    Find the column (unique x position) with the greatest depth range.

    This is useful for identifying the best vertical span of electrodes
    for depth profiling, e.g., to plot ripple power vs depth.

    Parameters:
    -----------
    channel_locations : np.ndarray, ProbeInterface Probe, or SpikeInterface Recording
        Either (n_channels, 2) array of [x, y] positions in μm,
        or an object with channel location data
    return_all_info : bool, optional
        If True, return dict with column info. If False, return just x position.
        Default: False

    Returns:
    --------
    float or dict or None
        If return_all_info=False:
            - x position (float) of column with max depth range
            - None if multiple columns tie for max range
        If return_all_info=True:
            - dict with keys:
                * 'x_position': x coordinate of the column
                * 'depth_range': peak-to-peak depth range (μm)
                * 'channel_mask': boolean array for channels in this column
                * 'is_unique': whether this column uniquely has the max range
                * 'n_channels': number of channels in this column
            - None if multiple columns tie for max range

    Examples:
    ---------
    >>> # Simple usage
    >>> x_pos = get_longest_column(channel_locs)
    >>> if x_pos is not None:
    ...     mask = get_channels_in_column(channel_locs, x_pos)
    ...     depths = channel_locs[mask, 1]

    >>> # Get detailed info
    >>> result = get_longest_column(channel_locs, return_all_info=True)
    >>> if result is not None and result['is_unique']:
    ...     print(f"Column at x={result['x_position']} μm")
    ...     print(f"Depth range: {result['depth_range']} μm")
    """
    # Extract channel locations
    locations = _extract_channel_locations(channel_locations)

    # Get unique x positions
    unique_x = np.unique(locations[:, 0])

    # Calculate depth range for each column
    max_y_range = 0
    best_x = None
    columns_with_max_range = 0

    for x_pos in unique_x:
        mask = locations[:, 0] == x_pos
        y_values = locations[mask, 1]
        y_range = np.ptp(y_values)  # Peak-to-peak (max - min)

        if y_range > max_y_range:
            max_y_range = y_range
            best_x = x_pos
            columns_with_max_range = 1
        elif y_range == max_y_range:
            columns_with_max_range += 1

    # Check if there's a unique winner
    is_unique = columns_with_max_range == 1

    if not is_unique:
        return None

    # Return based on requested format
    if return_all_info:
        mask = locations[:, 0] == best_x
        return {
            "x_position": best_x,
            "depth_range": max_y_range,
            "channel_mask": mask,
            "is_unique": is_unique,
            "n_channels": np.sum(mask),
        }
    else:
        return best_x


def get_channels_in_column(channel_locations, x_position, tolerance=0.1):
    """
    Get all channels at a specific x position (within tolerance).

    Parameters:
    -----------
    channel_locations : np.ndarray, ProbeInterface Probe, or SpikeInterface Recording
        Channel positions
    x_position : float
        X coordinate to select (μm)
    tolerance : float, optional
        Tolerance for matching x position in μm (default: 0.1)

    Returns:
    --------
    np.ndarray
        Boolean mask array of shape (n_channels,) indicating which channels
        are in this column

    Examples:
    ---------
    >>> mask = get_channels_in_column(channel_locs, x_position=250.0)
    >>> column_channels = channel_locs[mask]
    >>> column_depths = column_channels[:, 1]
    """
    # Extract channel locations
    locations = _extract_channel_locations(channel_locations)

    # Find channels within tolerance of the specified x position
    mask = np.abs(locations[:, 0] - x_position) <= tolerance

    return mask


def get_shank_assignments(channel_locations, shank_spacing=250):
    """
    Assign channels to shanks based on x-position clustering.

    Groups channels into shanks by assuming shanks are separated by
    at least `shank_spacing` micrometers in the x-direction.

    Parameters:
    -----------
    channel_locations : np.ndarray, ProbeInterface Probe, or SpikeInterface Recording
        Channel positions
    shank_spacing : float, optional
        Minimum spacing between shanks in μm (default: 250)

    Returns:
    --------
    np.ndarray
        Integer array of shape (n_channels,) with shank assignments.
        Shanks are numbered 0, 1, 2, ... from left to right (increasing x).

    Examples:
    ---------
    >>> shank_ids = get_shank_assignments(channel_locs, shank_spacing=250)
    >>> n_shanks = np.max(shank_ids) + 1
    >>> for shank in range(n_shanks):
    ...     shank_channels = channel_locs[shank_ids == shank]
    ...     print(f"Shank {shank}: {len(shank_channels)} channels")
    """
    # Extract channel locations
    locations = _extract_channel_locations(channel_locations)

    # Get unique x positions sorted
    unique_x = np.sort(np.unique(locations[:, 0]))

    # Group x positions into shanks
    shank_x_groups = []
    current_group = [unique_x[0]]

    for x in unique_x[1:]:
        if x - current_group[-1] >= shank_spacing:
            # Start new shank
            shank_x_groups.append(current_group)
            current_group = [x]
        else:
            # Add to current shank
            current_group.append(x)

    # Don't forget the last group
    shank_x_groups.append(current_group)

    # Assign shank IDs to each channel
    shank_assignments = np.zeros(len(locations), dtype=int)

    for shank_id, x_positions in enumerate(shank_x_groups):
        for x_pos in x_positions:
            mask = locations[:, 0] == x_pos
            shank_assignments[mask] = shank_id

    return shank_assignments
