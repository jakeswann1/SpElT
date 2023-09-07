import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from ipywidgets import widgets, interact
import matplotlib.pyplot as plt


# Function to generate rate maps for a defined number of X and Y bins
def generate_rate_maps(spike_data, positions, ppm=1, x_bins=10, y_bins=10, dt=1.0, smoothing_window=5):
    """
    Generate smoothed rate maps for neurons with optimized computational efficiency.
    
    Parameters:
    - spike_data: dict
        Dictionary containing spike times organized by clusters.
    - positions: pd.DataFrame
        DataFrame containing 'x' and 'y' coordinates with timestamps as the column index.
    - ppm: float
        Pixels per meter for converting spatial coordinates.
    - x_bins, y_bins: int
        Number of bins along the x and y axes.
    - dt: float
        Time window for spike count.
    - smoothing_window: int
        Size of the boxcar smoothing window.
        
    Returns:
    - rate_maps_dict: dict
        Dictionary containing smoothed rate maps organized by clusters.
    """
    
    # Handle NaN values by replacing them with the mean of the respective coordinates
    positions.fillna(positions.mean(), inplace=True)
    
    positions /= ppm  # Simplified positions calculation
    x_coords, y_coords = positions.values[0, :], positions.values[1, :]
    sampleTimes = positions.columns.to_numpy()
    
    # Determine the bin edges based on the min and max coordinates and the number of bins
    x_bin_edges = np.linspace(np.nanmin(x_coords), np.nanmax(x_coords), x_bins + 1)
    y_bin_edges = np.linspace(np.nanmin(y_coords), np.nanmax(y_coords), y_bins + 1)
    
    x_bin_idx = np.digitize(x_coords, x_bin_edges) - 1
    y_bin_idx = np.digitize(y_coords, y_bin_edges) - 1
    
    x_bin_idx = np.clip(x_bin_idx, 0, x_bins - 1)
    y_bin_idx = np.clip(y_bin_idx, 0, y_bins - 1)
    
    occupancy, _, _ = np.histogram2d(x_coords, y_coords, bins=[x_bin_edges, y_bin_edges])
    
    rate_maps_dict = {}
    
    for key, sub_dict in spike_data.items():
        rate_maps_sub_dict = {cluster: np.zeros_like(occupancy) for cluster in sub_dict.keys()}
        
        for cluster, spike_times in sub_dict.items():
            in_bins = np.digitize(spike_times, sampleTimes) - 1
            np.add.at(rate_maps_sub_dict[cluster], (x_bin_idx[in_bins], y_bin_idx[in_bins]), 1)
        
        for cluster, spike_count in rate_maps_sub_dict.items():
            rate_map_raw = spike_count / (occupancy * dt + 1e-10)
            rate_map_smoothed = uniform_filter(rate_map_raw, size=smoothing_window)
            rate_map_smoothed[occupancy == 0] = np.nan
            rate_maps_sub_dict[cluster] = rate_map_smoothed
            
    return rate_maps_sub_dict, occupancy

def plot_cluster_across_sessions(rate_maps_dict, cluster_id, title_prefix=""):
    n_sessions = sum(cluster_id in sub_dict for sub_dict in rate_maps_dict.values())
    if n_sessions == 0:
        print(f"Cluster {cluster_id} is not found in any session.")
        return
    fig, axes = plt.subplots(1, n_sessions, figsize=(15, 5))
    fig.suptitle(f"{title_prefix} Cluster {cluster_id}")
    if n_sessions == 1:
        axes = [axes]
    ax_idx = 0
    for session_key, sub_dict in rate_maps_dict.items():
        if cluster_id in sub_dict:
            rate_map = sub_dict[cluster_id]
            im = axes[ax_idx].imshow(rate_map, cmap='jet', origin='lower')
            axes[ax_idx].set_title(f"Trial {session_key}")
            # plt.colorbar(im, ax=axes[ax_idx])
            ax_idx += 1
    plt.show()

def interactive_cluster_plot(rate_maps_dict, title_prefix=""):
    unique_clusters = set(cluster for sub_dict in rate_maps_dict.values() for cluster in sub_dict.keys())
    @interact(cluster_id=widgets.Dropdown(options=sorted(unique_clusters), description='Cluster ID:', disabled=False))
    def plot_selected_cluster(cluster_id):
        plot_cluster_across_sessions(rate_maps_dict, cluster_id, title_prefix)

        
from scipy.stats import entropy

def calculate_skaggs_information(rate_maps, occupancy, dt=1.0):
    """
    Calculate Skaggs' spatial information score for given rate maps and occupancy.
    
    Parameters:
    - rate_maps: dict
        Dictionary containing smoothed rate maps organized by clusters.
    - occupancy: np.ndarray
        2D array indicating occupancy of each bin.
    - dt: float
        Time window for spike count.
        
    Returns:
    - skaggs_info_dict: dict
        Dictionary containing Skaggs' spatial information scores organized by clusters.
    """
    
    skaggs_info_dict = {}
    
    # Calculate the total time spent in the environment
    total_time = np.nansum(occupancy) * dt

    for cluster, rate_map in rate_maps.items():
        
        # Calculate mean firing rate across all bins
        mean_firing_rate = np.nansum(rate_map * occupancy) / total_time

        # Calculate probability of occupancy for each bin
        prob_occupancy = occupancy / np.nansum(occupancy)

        # Calculate Skaggs' spatial information score
        non_zero_idx = (rate_map > 0) & (prob_occupancy > 0)
        skaggs_info = np.nansum(
            prob_occupancy[non_zero_idx] *
            rate_map[non_zero_idx] *
            np.log2(rate_map[non_zero_idx] / mean_firing_rate)
        )

        skaggs_info_dict[cluster] = skaggs_info
            
    return skaggs_info_dict