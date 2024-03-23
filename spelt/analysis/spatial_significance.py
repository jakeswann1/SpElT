import numpy as np
import pynapple as nap
from .spatial_information import spatial_info
from spelt.maps.rate_maps import make_rate_maps
from joblib import Parallel, delayed

def compute_shuffle(spike_times_real, pos_sample_times, pos_bin_idx, pos_sampling_rate):
    
    # Shuffle spike times
    spike_times_shuffled = nap.shuffle_ts_intervals(spike_times_real)
    spike_times_shuffled = np.array(spike_times_shuffled.as_series().index)

    # Calculate rate map
    rate_map_shuffled, pos_map = make_rate_maps(spike_times_shuffled, pos_sample_times, pos_bin_idx, pos_sampling_rate, max_rates = False)
    
    # Calculate spatial information from rate and pos maps
    bits_per_spike_shuffled, bits_per_sec_shuffled = spatial_info(rate_map_shuffled, pos_map)

    return bits_per_spike_shuffled, bits_per_sec_shuffled

def spatial_significance(pos_sample_times, pos_bin_idx, pos_sampling_rate, spike_times_real, n_shuffles = 1000):
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
    rate_maps_real, pos_map_real = make_rate_maps(spike_times_real, pos_sample_times, pos_bin_idx, pos_sampling_rate, max_rates = False)
    bits_per_spike_real, bits_per_sec_real = spatial_info(rate_maps_real, pos_map_real)

    spike_times_real = nap.Ts(t=spike_times_real, time_units="s") #Pynapple object

    # Perform shuffles in parallel        
    results = Parallel(n_jobs=-1)(delayed(compute_shuffle)(spike_times_real, pos_sample_times, pos_bin_idx, pos_sampling_rate) for _ in range(n_shuffles))

    # Unpack results
    bits_per_spike_shuffled, bits_per_sec_shuffled = zip(*results)
    bits_per_spike_shuffled = np.concatenate(bits_per_spike_shuffled)
    bits_per_sec_shuffled = np.concatenate(bits_per_sec_shuffled)

    # Calculate mean and standard deviation of the shuffled Skaggs information
    mean_shuffled = bits_per_spike_shuffled.mean()
    std_shuffled = bits_per_spike_shuffled.std()

    # Z-score calculation: difference between real and mean of shuffled, divided by std of shuffled
    bits_per_spike_z = ((bits_per_spike_real - mean_shuffled) / std_shuffled)[0]

    # P-value calculation: proportion of shuffled values greater than the real value
    # Note: This calculation assumes a one-tailed test, as we're only interested if the real value is significantly higher
    p_value = np.sum(bits_per_spike_shuffled >= bits_per_spike_real) / n_shuffles

    return p_value, bits_per_spike_z, bits_per_spike_shuffled