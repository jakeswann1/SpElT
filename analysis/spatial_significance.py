import numpy as np
import pynapple as nap
from spatial_information import spatial_info


def make_map_simple(spike_times, pos_sample_times, pos_bin_idx):
    """
    Make a simple, unsmoothed rate map and position map from spike times position sample times, and binned position data.
    """
    
    # Bin spike times into pos samples
    binned_spikes = np.digitize(spike_times, pos_sample_times)
    
    # Calculate position map
    pos_map, _, _ = np.histogram2d(pos_bin_idx[0], pos_bin_idx[1], bins = [np.arange(0, pos_bin_idx[0].max() + 1), np.arange(0, pos_bin_idx[1].max() + 1)])
    
    # Calculate rate map
    rate_map, _, _ = np.histogram2d(pos_bin_idx[0][binned_spikes], pos_bin_idx[1][binned_spikes], bins = [np.arange(0, pos_bin_idx[0].max() + 1), np.arange(0, pos_bin_idx[1].max() + 1)])
    
    return rate_map, pos_map

def spatial_significance(pos_sample_times, pos_bin_idx, spike_times_real, n_shuffles = 1000):
    """
    Calculate the significance of spatial information for a given cluster by shuffling spike times and recalculating spatial information.
    
    Parameters
    ----------
    pos_sample_times : array
        Array of time points at which position data was sampled
    pos_bin_idx : array
        Array of bin indices corresponding to x and y position data
    spike_times_real : array
        Array of spike times for a given cluster
    n_shuffles : int
        Number of shuffles to perform
    
    Returns
    -------
    bits_per_spike_shuffled : array
        Array of bits per spike for each shuffle
    bits_per_sec_shuffled : array
        Array of bits per second for each shuffle
    """

    bits_per_spike_shuffled = np.zeros(n_shuffles)
    bits_per_sec_shuffled = np.zeros(n_shuffles)

    # Calculate real rate map and spatial info
    rate_map_real, pos_map_real = make_map_simple(spike_times_real, pos_sample_times, pos_bin_idx)
    bits_per_spike_real, bits_per_sec_real = spatial_info(rate_map_real, pos_map_real)

    spike_times = nap.Ts(t=spike_times_real, time_units="s") #Pynapple object
        
    for shuffle in range(n_shuffles):
        
        # Shuffle spike times
        spike_times_shuffled = nap.shuffle_ts_intervals(spike_times)
        spike_times_shuffled = np.array(spike_times_shuffled.as_series().index)

        # Calculate rate map
        rate_map_shuffled, _ = make_map_simple(spike_times_shuffled, pos_sample_times, pos_bin_idx)
        
        # Calculate spatial information from UNSMOOTHED rate and pos maps
        bits_per_spike_shuffled[shuffle], bits_per_sec_shuffled[shuffle] = spatial_info(rate_map_shuffled, pos_map_real)

    # Calculate mean and standard deviation of the shuffled Skaggs information
    mean_shuffled = bits_per_spike_shuffled.mean()
    std_shuffled = bits_per_spike_shuffled.std()

    # Z-score calculation: difference between real and mean of shuffled, divided by std of shuffled
    bits_per_spike_z = ((bits_per_spike_real - mean_shuffled) / std_shuffled)[0]

    # P-value calculation: proportion of shuffled values greater than the real value
    # Note: This calculation assumes a one-tailed test, as we're only interested if the real value is significantly higher
    p_value = np.sum(bits_per_spike_shuffled >= bits_per_spike_real) / n_shuffles

    return p_value, bits_per_spike_z
