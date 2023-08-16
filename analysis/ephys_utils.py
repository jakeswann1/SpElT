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

