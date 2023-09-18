import numpy as np
# Function to calculate burst index and autocorrelograms
def compute_autocorrelograms_and_burst_indices(spike_times, spike_clusters, bin_size, time_window, burst_threshold):
    """
    Function to compute autocorrelograms and burst indices for multiple clusters, 
    returning them as separate dictionaries.
    
    Parameters:
    - spike_times (numpy array): Array of spike times in seconds.
    - spike_clusters (numpy array): Array of cluster IDs corresponding to each spike time.
    - bin_size (float): The size of the bins for the autocorrelogram in seconds.
    - time_window (float): The time window around each spike to consider for autocorrelation in seconds.
    - burst_threshold (float): Time threshold for defining a burst, in seconds.
    
    Returns:
    - autocorrelograms (dict): A dictionary containing the bin centers and counts for each cluster's autocorrelogram.
    - burst_indices (dict): A dictionary containing the burst index for each cluster.
    """
    # Define the bin edges for the histogram (common for all clusters)
    bin_edges = np.arange(-time_window, time_window + bin_size, bin_size)
    
    # Initialize dictionaries for storing results
    autocorrelograms = {}
    burst_indices = {}
    
    # Identify unique clusters
    unique_clusters = np.unique(spike_clusters)
    
    for cluster in unique_clusters:
        # Extract spike times for the current cluster
        cluster_spike_times = spike_times[spike_clusters == cluster]
        
        # Efficiently compute the time differences matrix using broadcasting
        time_diffs = cluster_spike_times[:, None] - cluster_spike_times[None, :]
        
        # Remove the diagonal elements and flatten the matrix
        time_diffs = time_diffs[~np.eye(time_diffs.shape[0], dtype=bool)].flatten()
        
        # Early filtering of time differences within the specified time window
        time_diffs = time_diffs[np.abs(time_diffs) <= time_window]
        
        # Compute the histogram for autocorrelogram
        counts, _ = np.histogram(time_diffs, bins=bin_edges)
        
        # Calculate the centers of the bins for autocorrelogram
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Store autocorrelogram in dictionary
        autocorrelograms[cluster] = {'bin_centers': bin_centers, 'counts': counts}
        
        # Calculate inter-spike intervals (ISIs) for burst index
        isis = np.diff(cluster_spike_times)
        
        # Identify burst spikes based on ISIs and the burst threshold using boolean indexing
        burst_spikes = np.nonzero(isis < burst_threshold)[0]
        
        # Make sure each spike is counted only once for burst index calculation
        burst_spikes = np.concatenate([burst_spikes, burst_spikes + 1])
        burst_spikes = np.unique(burst_spikes)
        
        # Count the unique number of burst spikes
        num_burst_spikes = burst_spikes.size
        
        # Calculate the burst index
        burst_index = num_burst_spikes / cluster_spike_times.size
        
        # Store burst index in dictionary
        burst_indices[cluster] = burst_index
        
    return autocorrelograms, burst_indices

def plot_autocorrelogram(session, cluster, autocorrelogram, burst_index):
    """
    Function to plot an autocorrelogram for a given cluster and annotate it with the burst index.
    
    Parameters:
    - cluster (int): Cluster ID.
    - autocorrelogram (dict): A dictionary containing the bin centers and counts for the cluster's autocorrelogram.
    - burst_index (float): The burst index for the cluster.
    
    Returns:
    - fig, ax (matplotlib figure and axis): Figure and axis containing the plot.
    """
    fig, ax = plt.subplots()
    
    # Plot the autocorrelogram
    ax.bar(autocorrelogram['bin_centers'], autocorrelogram['counts'], width=0.001)  # Assuming bin size of 1ms
    ax.set_title(f"{session} Cluster {cluster}: Burst Index = {burst_index:.3f}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Counts")
    
    return fig, ax

import matplotlib.pyplot as plt
# Function to create a dropdown widget for plotting autocorrelograms
def plot_autocorrelograms_with_dropdown(autocorrelograms):
    """
    Function to plot autocorrelograms for each cluster using a dropdown widget.
    Note: This function is intended to be run in a Jupyter Notebook environment with ipywidgets installed.

    Parameters:
    - autocorrelograms (dict): A dictionary containing the bin centers and counts for each cluster's autocorrelogram.
    """
    from ipywidgets import interact, widgets  # Importing widgets within the function for portability

    # Create a dropdown widget with cluster IDs
    cluster_dropdown = widgets.Dropdown(
        options=list(autocorrelograms.keys()),
        description='Cluster ID:',
        disabled=False,
    )

    # Display the dropdown widget and plot the selected autocorrelogram
    interact(plot_autocorrelogram, cluster=cluster_dropdown, autocorrelograms = autocorrelograms)
