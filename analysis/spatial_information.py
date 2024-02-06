import numpy as np

def spatial_info(rate_maps, pos_map):
    """
    Find spatial information (Skaggs information) of place fields for 2D rate maps.

    Returns Skaggs et al's estimate of spatial information in bits per second:

    I = sum_x p(x) r(x) log2(r(x)/r)  
    then divide by mean rate over bins to get bits per spike.

    Parameters:
    rate_maps (dict, list, or ndarray): A dictionary with cell IDs as keys and 2D rate maps as values, or a list or 3D array of 2D rate maps. Each rate map represents the firing rate of a neuron at different locations in a 2D space.
    pos_maps (dict, list, or ndarray): A dictionary with cell IDs as keys and 2D position maps as values, or a list or 3D array of 2D position maps. Each position map represents the time spent by the animal at different locations in a 2D space.

    Returns:
    bits_per_spike (dict or ndarray): The spatial information per spike, in bits. If the input was a dictionary, this will be a dictionary with cell IDs as keys and spatial information values as values. If the input was a list or array, this will be a 1D array of spatial information values.
    bits_per_sec (dict or ndarray): The spatial information per second, in bits. If the input was a dictionary, this will be a dictionary with cell IDs as keys and spatial information values as values. If the input was a list or array, this will be a 1D array of spatial information values.
    """
    is_dict_rate_maps = isinstance(rate_maps, dict)

    # Check if rate_maps is a dictionary and convert to list if necessary
    if is_dict_rate_maps:
        keys = sorted(rate_maps.keys())
        rate_maps_list = [rate_maps[key] for key in keys]
    else:
        rate_maps_list = [rate_maps] if not isinstance(rate_maps, list) else rate_maps

    # Ensure rate_maps and pos_maps are NumPy arrays
    rate_maps_array = np.array(rate_maps_list)
    pos_maps_array = np.array(pos_map)

    # Handle case where number of cells is more than number of position maps
    if len(rate_maps_array) > len(pos_maps_array):
        pos_maps_array = np.tile(pos_maps_array, (len(rate_maps_array), 1))

    # Calculate total duration
    duration = np.nansum(pos_maps_array)

    # Mean rate for each map
    mean_rates = np.array([np.nansum(rate_map * pos_maps_array) / duration for rate_map in rate_maps_array])

    # Normalise occupancy and rate maps to give probability distributions
    p_x = pos_maps_array / duration
    p_r = rate_maps_array / mean_rates[:, np.newaxis, np.newaxis]
    
    # Replace NaNs and infinities in rate_maps_array and p_r
    rate_maps_array = np.nan_to_num(rate_maps_array)
    p_r = np.nan_to_num(p_r)

    # Replace zeroes with a small positive number in p_r before taking the logarithm
    p_r = np.where(p_r == 0, 1e-10, p_r)

    # Calculate spatial information
    bits_per_sec_array = np.array([np.nansum(p_x * rate_map * np.log2(p_r_i)) if mean_rate != 0 else 0 for rate_map, p_r_i, mean_rate in zip(rate_maps_array, p_r, mean_rates) if mean_rate != 0])
    bits_per_spike_array = bits_per_sec_array / mean_rates

    # Return either dict or array, matching input
    if is_dict_rate_maps:
        bits_per_spike = {key: bits_per_spike_array[i] for i, key in enumerate(keys)}
        bits_per_sec =  {key: bits_per_sec_array[i] for i, key in enumerate(keys)}
    else:
        bits_per_spike = bits_per_spike_array
        bits_per_sec = bits_per_sec_array
    return bits_per_spike, bits_per_sec

