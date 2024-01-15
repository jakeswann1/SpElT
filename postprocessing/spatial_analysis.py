import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from ipywidgets import widgets, interact
import matplotlib.pyplot as plt
from postprocessing.adaptive_smooth import *


def make_rate_maps(spike_data, pos_data, bin_length = 10, dt = 1.0, adaptive_smoothing = True, smoothing_window = None, alpha = None):
    """
    Generate smoothed rate maps for neurons with optimized computational efficiency.
    
    This function computes the rate maps for given clusters of neurons based on spike 
    times and animal positions. The rate map for each neuron is smoothed using a 
    uniform filter. The function also accounts for cases where occupancy is zero.

    Args:
        spike_data (dict): A dictionary containing spike times for each cluster. 
                           The keys are cluster identifiers, and the values are 
                           NumPy arrays of spike times in seconds.
        pos_data (dict): A dictionary containing the animal's position data in pixels and header information.
                         'xy_position' key has a DataFrame with x and y coordinates as row indices 
                         and time as columns.
                         'header' key contains metadata such as 'min_x', 'max_x', 'min_y', 'max_y'.
        bin_length (int, optional): The length of each square bin in pixels. 
                                    Defaults to 10.
        dt (float, optional): The time step in seconds for binning the spike data. 
                              Defaults to 1.0.
        adaptive_smoothing (bool, optional): Whether to use adaptive smoothing. Defaults to True.
        smoothing_window (int, optional): The size of the uniform filter window for smoothing 
                                          the rate maps. Defaults to None.
        alpha (float, optional): The alpha parameter for adaptive smoothing. Defaults to None.
        
    Returns:
        tuple: A tuple containing four elements:
            - rate_maps_dict (dict): A dictionary containing the smoothed rate maps 
                                     for each cluster. The keys are cluster identifiers, 
                                     and the values are 2D NumPy arrays representing the 
                                     rate maps.
            - occupancy (ndarray): A 2D NumPy array representing the time spent by the 
                                   animal in each spatial bin.
            - max_rates_dict (dict): A dictionary containing the maximum firing rates 
                                     for each cluster. The keys are cluster identifiers, 
                                     and the values are lists of maximum firing rates.
            - mean_rates_dict (dict): A dictionary containing the mean firing rates 
                                      for each cluster. The keys are cluster identifiers, 
                                      and the values are lists of mean firing rates.
    
    Raises:
        ValueError: If the input dictionaries `spike_data` or `pos_data` are empty.
        TypeError: If the input types do not match the expected types for `spike_data` 
                   and `pos_data`.
    """
    
    # Unpack the 'xy_position' DataFrame from the 'pos_data' dictionary
    positions = pos_data['xy_position']
    # Unpack speed
    speed = pos_data['speed']
    # Unpack position sampling rate
    pos_sampling_rate = pos_data['pos_sampling_rate']
    
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
    sample_times = positions.columns.to_numpy()
    
    # Digitize the x and y coordinates to find which bin they belong to
    x_bin_idx = np.digitize(x_coords, x_bin_edges) - 1
    y_bin_idx = np.digitize(y_coords, y_bin_edges) - 1
    
    # Clip the bin indices to lie within the valid range [0, number_of_bins - 1]
    x_bin_idx = np.clip(x_bin_idx, 0, x_bins - 1)
    y_bin_idx = np.clip(y_bin_idx, 0, y_bins - 1)
    
    # Filter xy positions for speed <2.5cm/s
    # THRESHOLD HARD_CODED FOR NOW
    speed_mask = speed >= 2.5
    x_coords = x_coords[speed_mask]
    y_coords = y_coords[speed_mask]
    
    # Compute the 2D occupancy map using a 2D histogram and normalize by the sampling rate to give seconds per bin
    # This ensures that rate maps are in units of Hz
    occupancy, _, _ = np.histogram2d(x_coords, y_coords, bins=[x_bin_edges, y_bin_edges])
    occupancy /= pos_sampling_rate
        
    # Initialize the rate maps dictionary with zeros using cluster keys
    rate_maps_dict = {cluster: np.zeros_like(occupancy) for cluster in spike_data.keys()}
    max_rates_dict = {cluster: [np.nan] for cluster in spike_data.keys()}
    mean_rates_dict = {cluster: [np.nan] for cluster in spike_data.keys()}
        
    # Populate the rate maps based on spike times
    for cluster, spike_times in spike_data.items():
        # Find the corresponding bins for each spike time
        in_bins = np.digitize(spike_times, sample_times) - 1
        # Increment the spike count in the respective bins
        np.add.at(rate_maps_dict[cluster], (x_bin_idx[in_bins], y_bin_idx[in_bins]), 1)
        
    # Smooth rate maps
    for cluster, spike_count in rate_maps_dict.items():
        # Set spike count to 0 where occupancy is 0 to avoid division by 0
        spike_count[occupancy == 0] = 0

        if adaptive_smoothing:
            # Apply adaptive smoothing
            smoothed_spk, smoothed_pos, smoothed_rate, _ = adaptive_smooth(spike_count, occupancy, alpha)
            rate_maps_dict[cluster] = smoothed_rate
        else:
            # Calculate the raw rate map by dividing spike count by occupancy time (plus a small constant)
            rate_map_raw = spike_count / (occupancy * dt + 1e-10)

            # # Apply uniform smoothing to the raw rate map
            # rate_map_smoothed = uniform_filter(rate_map_raw, size=smoothing_window)
            # rate_map_smoothed[occupancy == 0] = np.nan
            rate_maps_dict[cluster] = rate_map_raw

        # Calculate max and mean firing rates
        max_rates_dict[cluster] = np.nanmax(rate_maps_dict[cluster])
        mean_rates_dict[cluster] = np.nanmean(rate_maps_dict[cluster])
                

    # Before returning, transpose the arrays to account for an axis transformation that np.histogram2D does
    rate_maps_dict = {cluster: rate_map.T for cluster, rate_map in rate_maps_dict.items()}
    occupancy = occupancy.T
    
    return rate_maps_dict, occupancy, max_rates_dict, mean_rates_dict


def plot_cluster_across_sessions(rate_maps_dict, cluster_id, max_rates_dict, mean_rates_dict, spatial_info_dict, session="N/A", age=None):
    """
    Plots rate maps for a given cluster across multiple sessions.

    Args:
        rate_maps_dict (dict): A dictionary containing rate maps for each session.
        cluster_id (int): The ID of the cluster to plot.
        max_rates_dict (dict): A dictionary containing maximum firing rates for each session and cluster.
        mean_rates_dict (dict): A dictionary containing mean firing rates for each session and cluster.
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
            im = axes[ax_idx].imshow(rate_map, cmap='jet', origin='lower')
            axes[ax_idx].set_title(f"Trial {session_key}.\nMax FR: {max_rates_dict[session_key][cluster_id]:.2f} Hz. Mean FR: {mean_rates_dict[session_key][cluster_id]:.2f} Hz\n Spatial Info: {spatial_info_dict[session_key][cluster_id]:.2f} bits/spike")
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
    
    # Initialize a dictionary to hold filtered spikes
    filtered_spikes = {}
    
    # Sampling interval in seconds
    sampling_interval = 1 / position_sampling_rate
    
    for cluster, spikes in current_trial_spikes.items():
        # Convert spikes to closest speed index
        closest_indices = np.round(spikes / sampling_interval).astype(int)
        
        # Initialize an empty list to store filtered spikes for this cluster
        filtered_cluster_spikes = []
        
        for spike, closest_index in zip(spikes, closest_indices):
            # Retrieve speed at the closest index, if index is within bounds
            if closest_index < len(speed_data):
                speed_at_spike = speed_data[closest_index]
                
                # Check if speed is within the specified range
                if speed_lower_bound <= speed_at_spike <= speed_upper_bound:
                    filtered_cluster_spikes.append(spike)
        
        # Update the dictionary with filtered spikes for this cluster
        filtered_spikes[cluster] = np.array(filtered_cluster_spikes)
        
    return filtered_spikes