import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from ipywidgets import widgets, interact
import matplotlib.pyplot as plt
from postprocessing.adaptive_smooth import *

def bin_pos_data_axona(pos_data, bin_length = 2.5, speed_threshold = 2.5):

    """
    Unpacks the position data from the Axona format for rate map generation, and bin poitions.
    params:
    pos_data (dict): A dictionary containing the animal's position data in pixels and header information.
                     'xy_position' key has a DataFrame with x and y coordinates as row indices 
                     and time as columns.
                     'header' key contains metadata such as 'min_x', 'max_x', 'min_y', 'max_y'.
    bin_length (float, optional): The length of each square bin in centimeters. Defaults to 2.5.
    speed_threshold (float, optional): The speed threshold in cm/s for filtering the position data. Defaults to 2.5.

    """

    positions = pos_data['xy_position']
    pos_sample_times = pos_data['xy_position'].columns.to_numpy()

    speed = pos_data['speed']
    pos_sampling_rate = pos_data['pos_sampling_rate']
    scaled_ppm = pos_data['header']['scaled_ppm'] #pixels per meter
    
    # Get bin length in pixels
    bin_length = bin_length * scaled_ppm / 100 #convert to meters and then to pixels

    # Extract raw field of view (FOV) pixel boundaries
    # These coordinates are rounded to the nearest lower and upper bin edges, respectively
    min_x_raw = np.floor_divide(pos_data['header']['min_x'], bin_length) * bin_length
    max_x_raw = np.ceil(pos_data['header']['max_x'] / bin_length) * bin_length
    min_y_raw = np.floor_divide(pos_data['header']['min_y'], bin_length) * bin_length
    max_y_raw = np.ceil(pos_data['header']['max_y'] / bin_length) * bin_length
    
    # They are then scaled so that the NW corner is (0,0) to match the position data
    min_x = 0
    max_x = max_x_raw - min_x_raw
    min_y = 0
    max_y = max_y_raw - min_y_raw

    # Calculate the number of bins along the x and y axes
    x_bins = int((max_x - min_x) / bin_length)
    y_bins = int((max_y - min_y) / bin_length)
    
    # Generate the bin edges for the x and y axes based on the FOV
    x_bin_edges = np.linspace(min_x, max_x, x_bins + 1)
    y_bin_edges = np.linspace(min_y, max_y, y_bins + 1)
   
    # Impute missing values (NaNs) in the 'positions' DataFrame with their respective mean values
    positions.fillna(positions.mean(), inplace=True)
    
    # Extract x and y coordinates and their corresponding timestamps from the DataFrame
    x_coords, y_coords = positions.values[0, :], positions.values[1, :]

    # Speed filter
    speed_mask = speed >= speed_threshold
    x_coords = x_coords[speed_mask]
    y_coords = y_coords[speed_mask]
    pos_sample_times = pos_sample_times[speed_mask]

    # Digitize the x and y coordinates to find which bin they belong to
    x_bin_idx = np.digitize(x_coords, x_bin_edges) - 1
    y_bin_idx = np.digitize(y_coords, y_bin_edges) - 1
    
    # Clip the bin indices to lie within the valid range [0, number_of_bins - 1]
    x_bin_idx = np.clip(x_bin_idx, 0, x_bins - 1)
    y_bin_idx = np.clip(y_bin_idx, 0, y_bins - 1)
    pos_bin_idx = (x_bin_idx, y_bin_idx)

    return pos_bin_idx, pos_sample_times, pos_sampling_rate

def make_rate_maps(spike_data, pos_sample_times, pos_bin_idx, pos_sampling_rate, dt = 1.0, adaptive_smoothing = True, alpha = 200, max_rates = True):
    """
    Generate smoothed rate maps for neurons with optimized computational efficiency.
    
    This function computes the rate maps for given clusters of neurons based on spike 
    times and binned animal positions. The rate map for each neuron is smoothed using a 
    uniform filter. The function also accounts for cases where occupancy is zero.

    XY position data should be binned into a 2D grid (i.e. using bin_axona_pos_data), and the corresponding bin indices
    should be provided. The function also requires the sampling rate of the position data.

    Args:
        spike_data (dict): A dictionary containing spike times for each cluster. 
                           The keys are cluster identifiers, and the values are 
                           NumPy arrays of spike times in seconds.
        pos_sample_times (ndarray): A 1D NumPy array representing the sample times of the animal's position.
        pos_bin_idx (ndarray): A 2D NumPy array representing the bin indices of the animal's position.
        pos_sampling_rate (float): The sampling rate of the animal's position in Hz.
        dt (float, optional): The time step in seconds for binning the spike data. 
                              Defaults to 1.0.
        adaptive_smoothing (bool, optional): Whether to use adaptive smoothing. Defaults to True.
        alpha (float, optional): The alpha parameter for adaptive smoothing. Defaults to 200.
        max_rates (bool, optional): Whether to calculate max and mean firing rates. Defaults to True.
        
    Returns:
        tuple: A tuple containing the following elements:
            - rate_maps_dict (dict or ndarray): A dictionary containing the rate maps for each cluster. 
                                    The keys are cluster identifiers, and the values are 2D NumPy arrays.
                                    If only one rate map is generated, the function returns a 2D NumPy array.
            - pos_map (ndarray): A 2D NumPy array representing the 2D occupancy map.

            - max_rates_dict (dict): A dictionary containing the maximum firing rates for each cluster. Only returned if `max_rates` is True.
            - mean_rates_dict (dict): A dictionary containing the mean firing rates for each cluster. Only returned if `max_rates` is True.
    
    Raises:
        TypeError: If the input types do not match the expected types for `spike_data` 
                   and `pos_data`.
    """
    # Determine if spike_times is dict or array. If array, convert to dict
    if isinstance(spike_data, dict):
        return_dict = True
        pass
    elif isinstance(spike_data, np.ndarray):
        spike_data = {0: spike_data}
        return_dict = False
    else:
        raise TypeError("spike_data must be a dictionary or a 1D numpy array")

    bins = [np.arange(0, pos_bin_idx[0].max() + 1), np.arange(0, pos_bin_idx[1].max() + 1)]

    # Compute the 2D occupancy map
    pos_map, _, _ = np.histogram2d(pos_bin_idx[0], pos_bin_idx[1], bins=bins)
    # Normalize by the sampling rate to give seconds per bin. This ensures that rate maps are in units of Hz
    pos_map /= pos_sampling_rate
        
    # Initialize the rate maps dictionary with zeros using cluster keys
    rate_maps_dict = {cluster: np.zeros_like(pos_map) for cluster in spike_data.keys()}
    max_rates_dict = {cluster: [np.nan] for cluster in spike_data.keys()}
    mean_rates_dict = {cluster: [np.nan] for cluster in spike_data.keys()}
         
    # Populate the rate maps based on spike times
    for cluster, spike_times in spike_data.items():
        # Find the corresponding bins for each spike time
        binned_spikes = np.digitize(spike_times, pos_sample_times) - 1
        # Make spike map
        spike_map, _, _ = np.histogram2d(pos_bin_idx[0][binned_spikes], pos_bin_idx[1][binned_spikes], bins=bins)

        # Set spike count to 0 where occupancy is 0 to avoid division by 0
        spike_map[pos_map == 0] = 0

        # Smooth the spike map using an adaptive kernel
        if adaptive_smoothing:
            _, pos_map, rate_map, _ = adaptive_smooth(spike_map, pos_map, alpha)
            rate_maps_dict[cluster] = rate_map
        else:
            # Calculate the raw rate map by dividing spike count by occupancy time (plus a small constant)
            raw_rate_map = spike_map / (pos_map * dt + 1e-10)
            # Return raw rate and pos map if adaptive smoothing is not used
            rate_maps_dict[cluster] = raw_rate_map

        if max_rates == True:
            # Calculate max and mean firing rates
            max_rates_dict[cluster] = np.nanmax(rate_maps_dict[cluster])
            mean_rates_dict[cluster] = np.nanmean(rate_maps_dict[cluster])
                
    # Set pos map to NaN where occupancy is 0
    pos_map[pos_map == 0] = np.nan

    # Before returning, transpose the arrays to account for an axis transformation that np.histogram2D does
    rate_maps_dict = {cluster: rate_map.T for cluster, rate_map in rate_maps_dict.items()}
    pos_map = pos_map.T

    # If only one rate map, convert to array
    if return_dict == False:
        rate_maps_dict = rate_maps_dict[0]

    if max_rates is True:
        return rate_maps_dict, pos_map, max_rates_dict, mean_rates_dict
    else:
        return rate_maps_dict, pos_map



def plot_cluster_across_sessions(rate_maps_dict, cluster_id, max_rates_dict, mean_rates_dict, spatial_info_dict, spatial_significance_dict, session="N/A", age=None):
    """
    Plots rate maps for a given cluster across multiple sessions.

    Args:
        rate_maps_dict (dict): A dictionary containing rate maps for each session.
        cluster_id (int): The ID of the cluster to plot.
        max_rates_dict (dict): A dictionary containing maximum firing rates for each session and cluster.
        mean_rates_dict (dict): A dictionary containing mean firing rates for each session and cluster.
        spatial_info_dict (dict): A dictionary containing spatial information for each session and cluster.
        spatial_significance_dict (dict): A dictionary containing spatial significance for each session and cluster.
        session (str, optional): The session identifier. Defaults to "N/A".
        age (int, optional): The age of the cluster. Defaults to None.
    """
    n_sessions = sum(cluster_id in sub_dict for sub_dict in rate_maps_dict.values())
    
    if n_sessions == 0:
        print(f"Cluster {cluster_id} is not found in any session.")
        return
    
    # Create a figure with subplots for each session
    fig, axes = plt.subplots(1, n_sessions, figsize=(15, 5))
    fig.suptitle(f"Rate maps for Session {session} Cluster {cluster_id} Age P{age}", y = 1.1)
    
    # Convert axes to list if there is only one session
    if n_sessions == 1:
        axes = [axes]
    ax_idx = 0
    
    # Plot rate maps for each session
    for session_key, sub_dict in rate_maps_dict.items():
        if cluster_id in sub_dict:
            rate_map = sub_dict[cluster_id]
            im = axes[ax_idx].imshow(rate_map, cmap='jet', origin='lower', vmin = 0)
            axes[ax_idx].set_title(f"Trial {session_key}.\nMax FR: {max_rates_dict[session_key][cluster_id]:.2f} Hz. Mean FR: {mean_rates_dict[session_key][cluster_id]:.2f} Hz\n Spatial Info: {spatial_info_dict[session_key][cluster_id]:.2f}. P = {spatial_significance_dict[session_key][cluster_id]}")
            axes[ax_idx].invert_yaxis() # Needed to match rate maps to theta phase plots
            axes[ax_idx].axis('off')
            # plt.colorbar(im, ax=axes[ax_idx])
            ax_idx += 1


def speed_filter_spikes(current_trial_spikes, speed_data, position_sampling_rate, speed_lower_bound, speed_upper_bound):
    """
    Filters spike times based on the animal's speed at the time of the spike.
    
    Parameters:
    - current_trial_spikes (dict): Dictionary containing spike times. Keys are cluster IDs and values are numpy arrays of spike times.
    - speed_array (array): Array containing speed values sampled at `position_sampling_rate`.
    - position_sampling_rate (float): Sampling rate of the position data in Hz.
    - speed_lower_bound (float): Lower bound of the speed range.
    - speed_upper_bound (float): Upper bound of the speed range.
    
    Returns:
    - filtered_spikes (dict): Dictionary containing filtered spike times.
    """
    filtered_spikes = {}
    
    sampling_interval = 1 / position_sampling_rate
    
    for cluster, spikes in current_trial_spikes.items():
        # Convert spikes to closest speed index
        closest_indices = np.round(spikes / sampling_interval).astype(int)
        
        # Pre-filter indices that are out of bounds
        valid_indices = (closest_indices >= 0) & (closest_indices < len(speed_data))
        closest_indices = closest_indices[valid_indices]
        spikes = spikes[valid_indices]
        
        # Retrieve speeds at the closest indices
        speeds_at_spikes = speed_data[closest_indices]
        
        # Check which speeds are within the specified range
        valid_speeds = (speed_lower_bound <= speeds_at_spikes) & (speeds_at_spikes <= speed_upper_bound)
        
        # Update the dictionary with filtered spikes for this cluster
        filtered_spikes[cluster] = spikes[valid_speeds]
        
    return filtered_spikes