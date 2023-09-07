# Various functions useful for analysis
import numpy as np

def load_session(obj):
    '''
    Loads all metadata, position files, and spikes for all trials of a session into an existing ephys class object
    
    Params:
     - obj: instance of an ephys class object
    '''
    for i in range(len(obj.trial_list)):
        obj.load_metadata(i)
        obj.load_pos(i)
    obj.load_spikes('good')

def select_spikes_by_trial(spike_data, trials, trial_offsets):
    """
    Select spikes from specific trials. Returns spikes time-indexed from 0 at the start of each trial
    
    Parameters:
    - spike_data: Dictionary containing spike data (including 'trial_for_spike').
    - trials: Single trial number or a list of trial numbers to filter by.
    
    Returns:
    - Dictionary containing filtered spike times and clusters.
    """
    if isinstance(trials, int):
        trials = [trials]  # Convert single trial number to list
    
    result = {}
    # Select spikes by trial and reset spike times to start at 0 for each trial
    for trial in trials:
        mask = np.isin(spike_data['spike_trial'], trial)
        result[trial] = {
            'spike_times': spike_data['spike_times'][mask] - trial_offsets[trial],
            'spike_clusters': spike_data['spike_clusters'][mask]
        }
    
    return result

# Function to transform the spike data dictionary
def transform_spike_data(spike_data_dict):
    """
    Transforms a dictionary containing sub-dictionaries of spike times and spike clusters
    into a dictionary where each sub-dictionary contains spike times organized by spike clusters.
    
    Parameters:
    - spike_data_dict: dict
        The original dictionary containing sub-dictionaries with keys 'spike_times' and 'spike_clusters'.
        
    Returns:
    - transformed_dict: dict
        Transformed dictionary where each sub-dictionary contains spike times organized by spike clusters.
    """
    transformed_dict = {}
    
    for key, sub_dict in spike_data_dict.items():
        # Extract spike times and clusters from the sub-dictionary
        spike_times = sub_dict['spike_times'].flatten()  # Flattening the array for easier indexing
        spike_clusters = sub_dict['spike_clusters']
        
        # Initialize an empty dictionary to store spike times by cluster
        transformed_sub_dict = {}
        
        # Iterate through each unique cluster and collect corresponding spike times
        for cluster in np.unique(spike_clusters):
            cluster_spike_times = spike_times[spike_clusters == cluster]
            transformed_sub_dict[cluster] = cluster_spike_times
            
        # Store the transformed sub-dictionary in the main dictionary
        transformed_dict[key] = transformed_sub_dict
    
    return transformed_dict

from scipy.signal import firwin, filtfilt, hilbert

def get_theta_phase(lfp, spike_times, sampling_rate, peakFreq, filtHalfBandWidth=2, powerThresh=5):
    """
    Calculate the theta phase of spikes based on the LFP using the corrected bad data handling logic.

    Parameters:
    - lfp: Local Field Potential time series.
    - spike_times: Times (in seconds) at which the spikes occurred.
    - sampling_rate: The sampling rate of the LFP.
    - peakFreq: The central frequency around which the LFP is filtered.
    - filtHalfBandWidth: Half bandwidth for filtering. Default is 3 Hz.
    - powerThresh: Threshold (percentile) for minimum power per cycle. Default is 5.

    Returns:
    - spike_phases: Theta phase (in radians) of spikes.
    """

    # 1. FILTER THE LFP AROUND THE PEAK FREQUENCY
    low = (peakFreq - filtHalfBandWidth)
    high = (peakFreq + filtHalfBandWidth)
    taps = firwin(round(sampling_rate) + 1, [low, high], pass_zero=False, window='blackman', fs=sampling_rate)
    padlen = min(3 * (len(taps) - 1), len(lfp) - 1)
    theta_lfp = filtfilt(taps, 1, lfp, padlen=padlen)

    # 2. EXTRACT INSTANTANEOUS PHASE USING HILBERT TRANSFORM
    analytic_signal = hilbert(theta_lfp)
    eegPhase = np.angle(analytic_signal)
    eegPhase = np.mod(eegPhase, 2 * np.pi)

    # 3. IDENTIFY AND HANDLE PHASE TRANSITIONS AND SLIPS
    phaseTrans = np.diff(eegPhase) < -np.pi
    phaseTrans = np.hstack(([True], phaseTrans, [True]))
    phaseSlips = np.hstack(([False], np.diff(np.unwrap(eegPhase)) < 0, [False]))
    phaseTrans[phaseSlips] = False

    # 4. CALCULATE EEG POWER AND POWER PER CYCLE
    cycleN = np.cumsum(phaseTrans[:-1])
    power = theta_lfp**2
    powerPerCycle = np.bincount(cycleN, power) / np.bincount(cycleN)
    
    # Calculate cycle length
    cycleLength = np.bincount(cycleN)
    
    # 5. HANDLE BAD DATA
    # Power threshold
    thresh = np.nanpercentile(powerPerCycle, powerThresh)
    badPowerCycle = np.where(powerPerCycle < thresh)[0]
    badPowerInd = np.isin(cycleN, badPowerCycle)

    # Length threshold
    min_cycle_length = np.ceil(sampling_rate / (peakFreq + filtHalfBandWidth))
    max_cycle_length = np.ceil(sampling_rate / (peakFreq - filtHalfBandWidth))
    badLengthCycle = np.where((cycleLength < min_cycle_length) | (cycleLength > max_cycle_length))[0]
    badLengthInd = np.isin(cycleN, badLengthCycle)
    
    # Remove from data
    eegPhase[badLengthInd | badPowerInd] = np.nan

    # 6. MAP SPIKE TIMES TO PHASE
    spike_indices = (spike_times * sampling_rate).astype(int)
    spike_phases = eegPhase[spike_indices]

    return spike_phases

from scipy.sparse import coo_matrix
def find_template_for_clusters(clu, spike_templates):
    """
    Determine the most represented template for each cluster and return as a dictionary.
    
    Parameters:
    -----------
    clu : np.ndarray
        Array of cluster IDs corresponding to each spike.
    spike_templates : np.ndarray
        Array of template IDs corresponding to each spike.
        
    Returns:
    --------
    temp_per_clu_dict : dict
        Dictionary where keys are cluster IDs and values are the template most represented 
        for that cluster.
    """
    # Ensure the input arrays are 1D
    clu = clu.reshape(-1)
    spike_templates = spike_templates.reshape(-1)
    
    # Create a sparse matrix to count occurrences
    temp_counts_by_clu = coo_matrix((np.ones(clu.shape[0]), (clu, spike_templates))).toarray()
    
    # Find the column index with the maximum count for each row (cluster)
    temp_per_clu = np.argmax(temp_counts_by_clu, axis=1) - 1
    
    # Convert to float array for inserting NaN values
    temp_per_clu = temp_per_clu.astype(float)
    
    # Identify and set NaN for non-existent clusters
    existent_clusters = np.unique(clu)
    non_existent_clusters = np.setdiff1d(np.arange(len(temp_per_clu)), existent_clusters)
    temp_per_clu[non_existent_clusters] = np.nan
    
    # Create dictionary mapping cluster ID to the most represented template ID
    temp_per_clu_dict = {cluster: template for cluster, template in enumerate(temp_per_clu) if not np.isnan(template)}
    
    return temp_per_clu_dict