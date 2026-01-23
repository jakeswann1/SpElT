import numpy as np
import pynapple as nap
from joblib import Parallel, delayed

from spelt.maps.rate_maps import make_rate_maps

from .information import spatial_info


def compute_shuffle(spike_times_real, pos_sample_times, pos_bin_idx, pos_sampling_rate):
    # Shuffle spike times
    spike_times_shuffled = nap.shuffle_ts_intervals(spike_times_real)
    spike_times_shuffled = np.array(spike_times_shuffled.as_series().index)

    # Calculate rate map
    rate_map_shuffled, pos_map = make_rate_maps(
        spike_times_shuffled,
        pos_sample_times,
        pos_bin_idx,
        pos_sampling_rate,
        max_rates=False,
    )

    # Calculate spatial information from rate and pos maps
    # Wrap single rate map in a list for spatial_info compatibility
    bits_per_spike_shuffled, bits_per_sec_shuffled = spatial_info(
        [rate_map_shuffled], [pos_map]
    )

    return bits_per_spike_shuffled, bits_per_sec_shuffled


def spatial_significance(
    pos_sample_times, pos_bin_idx, pos_sampling_rate, spike_times_real, n_shuffles=1000
):
    """
    Calculate significance of spatial information by shuffling spike times.

    Tests spatial information significance for a cluster by shuffling spike times
    and recalculating spatial information.

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
    p_value : float
        P-value from shuffle test (NaN if insufficient spikes)
    bits_per_spike_z : float
        Z-score of bits per spike (NaN if insufficient spikes)
    bits_per_spike_shuffled : array
        Array of bits per spike for each shuffle (empty if insufficient spikes)
    """

    # FIX 3: Validate spike count before attempting shuffle
    # Cells with very few spikes (e.g., after speed filtering) cannot be reliably tested
    if len(spike_times_real) < 10:
        return np.nan, np.nan, np.array([])

    bits_per_spike_shuffled = np.zeros(n_shuffles)
    bits_per_sec_shuffled = np.zeros(n_shuffles)

    # Calculate real rate map and spatial info
    rate_maps_real, pos_map_real = make_rate_maps(
        spike_times_real,
        pos_sample_times,
        pos_bin_idx,
        pos_sampling_rate,
        max_rates=False,
    )
    # Wrap single rate map in a list for spatial_info compatibility
    bits_per_spike_real, bits_per_sec_real = spatial_info(
        [rate_maps_real], [pos_map_real]
    )

    spike_times_real = nap.Ts(t=spike_times_real, time_units="s")  # Pynapple object

    # Perform shuffles in parallel
    results = Parallel(n_jobs=-1)(
        delayed(compute_shuffle)(
            spike_times_real, pos_sample_times, pos_bin_idx, pos_sampling_rate
        )
        for _ in range(n_shuffles)
    )

    # Unpack results
    bits_per_spike_shuffled, bits_per_sec_shuffled = zip(*results)
    bits_per_spike_shuffled = np.concatenate(bits_per_spike_shuffled)
    bits_per_sec_shuffled = np.concatenate(bits_per_sec_shuffled)

    # Calculate mean and standard deviation of the shuffled Skaggs information
    mean_shuffled = bits_per_spike_shuffled.mean()
    std_shuffled = bits_per_spike_shuffled.std()

    # Z-score calculation: difference between real and mean of shuffled,
    # divided by std of shuffled
    bits_per_spike_z = ((bits_per_spike_real - mean_shuffled) / std_shuffled)[0]

    # P-value calculation: proportion of shuffled values greater than real value
    # Note: This assumes a one-tailed test, as we're only interested if the
    # real value is significantly higher
    # Adding 1 to numerator and denominator prevents p=0
    p_value = (np.sum(bits_per_spike_shuffled > bits_per_spike_real) + 1) / (
        n_shuffles + 1
    )

    return p_value, bits_per_spike_z, bits_per_spike_shuffled
