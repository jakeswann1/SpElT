import numpy as np


def spatial_info(rate_maps, pos_map):
    """
    Compute spatial information (Skaggs information) of place fields for 2D rate maps, accommodating
    for either a single shared position map or individual position maps for each rate map.

    Parameters:
    rate_maps (dict, list, or ndarray): A dictionary with cell IDs as keys and 2D rate maps as values,
        or a list or 3D array of 2D rate maps. Each rate map represents the firing rate of a neuron
        at different locations in a 2D space.
    pos_map (dict, list, ndarray, or 2D ndarray): A single 2D position map to be applied to each rate map,
        or a structure (dict, list, 3D ndarray) with a position map for each rate map.

    Returns:
    bits_per_spike (dict or ndarray): The spatial information per spike, in bits.
    bits_per_sec (dict or ndarray): The spatial information per second, in bits.
    """
    is_dict_rate_maps = isinstance(rate_maps, dict)
    is_dict_pos_map = isinstance(pos_map, dict)
    single_pos_map = isinstance(pos_map, np.ndarray) and pos_map.ndim == 2

    # Convert rate_maps to a list of 2D arrays
    if is_dict_rate_maps:
        keys = sorted(rate_maps.keys())
        rate_maps_list = [rate_maps[key] for key in keys]
    else:
        if isinstance(rate_maps, np.ndarray) and rate_maps.ndim == 3:
            rate_maps_list = [rate_maps[i, :, :] for i in range(rate_maps.shape[0])]
        elif isinstance(rate_maps, list):
            rate_maps_list = rate_maps
        else:
            raise ValueError("rate_maps should be a dict, list, or 3D ndarray.")

    # Handle pos_map based on its type
    if single_pos_map:
        # Use the same position map for each rate map
        pos_maps_list = [pos_map for _ in range(len(rate_maps_list))]
    elif is_dict_pos_map:
        # Use corresponding position maps from dict
        pos_maps_list = [pos_map[key] for key in keys]
    else:
        if isinstance(pos_map, np.ndarray) and pos_map.ndim == 3:
            pos_maps_list = [pos_map[i, :, :] for i in range(pos_map.shape[0])]
        elif isinstance(pos_map, list):
            pos_maps_list = pos_map
        else:
            raise ValueError(
                "pos_map should be a dict, list, 3D ndarray, or 2D ndarray."
            )

    # Ensure inputs are 3D NumPy arrays
    rate_maps_array = np.array(rate_maps_list)
    pos_maps_array = np.array(pos_maps_list)

    if rate_maps_array.shape != pos_maps_array.shape:
        raise ValueError("Shape mismatch between rate maps and position maps.")

    # Calculate total occupancy and mean rates
    total_occupancy = np.nansum(pos_maps_array, axis=(1, 2))
    mean_rates = (
        np.nansum(rate_maps_array * pos_maps_array, axis=(1, 2)) / total_occupancy
    )

    # Make probability maps
    p_x = pos_maps_array / total_occupancy[:, None, None]
    p_r = rate_maps_array / mean_rates[:, None, None]

    # Replace NaNs and infinities in rate_maps_array and p_r
    rate_maps_array = np.nan_to_num(rate_maps_array)
    p_r = np.nan_to_num(p_r)

    # Calculate spatial information
    with np.errstate(divide="ignore", invalid="ignore"):
        bits_per_sec_array = np.array(
            [
                (
                    np.nansum(p_x[i] * rate_maps_array[i] * np.log2(p_r[i]))
                    if mean_rates[i] != 0
                    else 0
                )
                for i in range(rate_maps_array.shape[0])
            ]
        )
    bits_per_spike_array = bits_per_sec_array / mean_rates

    # Prepare output format to match input format
    if is_dict_rate_maps:
        bits_per_spike = {key: bits_per_spike_array[i] for i, key in enumerate(keys)}
        bits_per_sec = {key: bits_per_sec_array[i] for i, key in enumerate(keys)}
    else:
        bits_per_spike = bits_per_spike_array
        bits_per_sec = bits_per_sec_array

    return bits_per_spike, bits_per_sec
