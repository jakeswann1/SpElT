"""Helper functions for ensuring consistent rate map dimensions."""

import numpy as np


def pad_rate_maps_to_match(
    rate_maps_1: dict[int, np.ndarray], rate_maps_2: dict[int, np.ndarray]
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Pad two sets of rate maps to have matching dimensions.

    Useful when comparing rate maps from different trajectory types that
    may cover slightly different spatial extents (e.g., left vs right
    choice trajectories in T-maze).

    Parameters
    ----------
    rate_maps_1 : dict
        First set of rate maps {unit_id: rate_map}
    rate_maps_2 : dict
        Second set of rate maps {unit_id: rate_map}

    Returns
    -------
    padded_maps_1 : dict
        First set of rate maps padded to common dimensions
    padded_maps_2 : dict
        Second set of rate maps padded to common dimensions

    Notes
    -----
    - Determines the maximum dimensions across all maps
    - Pads with NaN values to preserve masking in correlation calculations
    - Both output dicts will contain only units present in both inputs

    Examples
    --------
    >>> left_maps = {1: np.random.rand(22, 41), 2: np.random.rand(20, 40)}
    >>> right_maps = {1: np.random.rand(26, 41), 2: np.random.rand(25, 42)}
    >>> left_padded, right_padded = pad_rate_maps_to_match(left_maps, right_maps)
    >>> left_padded[1].shape == right_padded[1].shape
    True
    """
    # Find common units
    common_units = set(rate_maps_1.keys()) & set(rate_maps_2.keys())

    if not common_units:
        return {}, {}

    # Determine maximum dimensions
    max_shape = [0, 0]
    for unit_id in common_units:
        shape1 = rate_maps_1[unit_id].shape
        shape2 = rate_maps_2[unit_id].shape
        max_shape[0] = max(max_shape[0], shape1[0], shape2[0])
        max_shape[1] = max(max_shape[1], shape1[1], shape2[1])

    # Pad both sets of rate maps
    padded_maps_1 = {}
    padded_maps_2 = {}

    for unit_id in common_units:
        map1 = rate_maps_1[unit_id]
        map2 = rate_maps_2[unit_id]

        # Create padded arrays filled with NaN
        padded_1 = np.full(max_shape, np.nan)
        padded_2 = np.full(max_shape, np.nan)

        # Copy original data into padded arrays
        padded_1[: map1.shape[0], : map1.shape[1]] = map1
        padded_2[: map2.shape[0], : map2.shape[1]] = map2

        padded_maps_1[unit_id] = padded_1
        padded_maps_2[unit_id] = padded_2

    return padded_maps_1, padded_maps_2
