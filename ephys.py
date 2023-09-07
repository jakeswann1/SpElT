import pandas as pd
import numpy as np
from tkinter import filedialog

from preprocessing.utils import gs_to_df

class ephys:
    '''
    A class to manage ephys data, including metadata, position, LFP, and spike data recorded from (currently): 
    - raw DACQ recordings sorted with Kilosort 2 and curated with phy
    
    Assumes a basic file structure of path_to_data/animal/YYYY-MM-DD/ for each session

    Usage:
        # Initialize the class with recording type and optional path to session.
        obj = ephys('nexus', 'path/to/recording/data/animalID/date')
        obj = ephys('nexus') will prompt a box to select the recording folder
        

        # Load metadata for the first trial.
        obj.load_metadata(0)

        # Load position data for the first trial.
        obj.load_pos(0)

        # Load LFP data for the first trial with a specific sampling rate, start time, end time, and channels.
        obj.load_lfp(0, 30000, 0, 600, [1, 2, 3, 4])

        # Load spike data for the session.
        obj.load_spikes()
    
    Attributes:
        recording_type (str): Type of the recording. Current options: 'nexus'
        
        base_path (str): Base path to the recording data. Hard-coded for now
        recording_path (str): Path to the specific animal and date recording folder
        sorting_path (str): Path to the sorting data folder
        
        session (pd.DataFrame): Dataframe with session information.
        trial_list (list): List containing the names of each trial.
        
        Recording data:
            metadata (list): List to store metadata for each trial.
            lfp_data (list): List to store LFP data for each trial.
            pos_data (list): List to store position data for each trial.
            spike_data (list): List to store spike data for the session.
        
        Constants for position processing:
            max_speed (int): Maximum speed constant for position processing.
            smoothing_window_size (int): Smoothing window size constant for position processing.
            
    Dependencies:
        spikeinterface (pip install spikeinterface)
        process_pos_data (custom)
        numpy, pandas
    '''

    def __init__(self, recording_type, path = None):
        self.recording_type = recording_type
        
        if path:
            self.recording_path = path
        else:
            print('Locate path to recording session')
            self.recording_path = filedialog.askdirectory()
        
        # Get date and animal ID from path
        self.date = self.recording_path.split('/')[-1]
        self.date_short = f'{self.date[2:4]}{self.date[5:7]}{self.date[8:10]}'
        self.animal = self.recording_path.split('/')[-2]
        
        # Get age from Google Sheet
        df = gs_to_df('https://docs.google.com/spreadsheets/d/1_Xs5i-rHNTywV-WuQ8-TZliSjTxQCCqGWOD2AL_LIq0/edit#gid=0')
        self.age = df['Age'].loc[df['Session'] == f'{self.animal}_{self.date_short}'].iloc[0]

        self.sorting_path = f'{self.recording_path}/{self.date_short}_sorting_ks2_custom'
        
        # Load session information from session.csv which is within the sorting folder as dataframe
        self.session = pd.read_csv(f'{self.sorting_path}/session.csv', index_col = 0)
        
        # Collect each trial name into a numpy array
        self.trial_list = self.session.iloc[:,1].to_list()
        
        # Collect trial offsets for aligning spike data
        recordings = self.session.iloc[:,0].to_list()
        self.trial_offsets = []
        offset = 0
        for i in recordings:
            self.trial_offsets.append(offset)
            i = float(i[-8:-1])
            offset += i

        # Initialise data variables
        self.metadata = [None] * len(self.trial_list)
        self.lfp_data = [None] * len(self.trial_list)
        self.pos_data = [None] * len(self.trial_list)
        self.spike_data = [None]
        
        
        # Set constants for position processing
        self.max_speed = 5
        self.smoothing_window_size = 3
    
    def load_metadata(self, trial_iterator):
        """
        Loads the metadata for a specified trial. Currently only a subset of lines from the Dacq .set file

        Args:
            trial_iterator (int): The index of the trial for which metadata is to be loaded.

        Populates:
            self.metadata (list): A list that stores metadata for each trial. The metadata for the specified trial is added at the given index.
        """        
        if self.recording_type == 'nexus':
            # Get path of trial to load
            path = f'{self.recording_path}/{self.trial_list[trial_iterator]}.set'

            print(f'Loading set file: {path}')

            with open(path, 'rb') as fid:
                setHeader = {}

                # Read the lines of the file up to the specified number (5 in this case) and write into dict
                for _ in range(5):
                    line = fid.readline()
                    if not line:
                        break
                    elements = line.decode().strip().split()
                    setHeader[elements[0]] = ' '.join(elements[1:])
                    
            # Add sampling rate - HARDCODED FOR NOW
            setHeader['sampling_rate'] = 48000

            #Populate basic metdata       
            self.metadata[trial_iterator] = setHeader

    def load_pos(self, trial_list, reload_flag = False):
        """
        Loads  and postprocesses the position data for a specified trial. Currently only from Dacq .pos files

        Args:
            trial_iterator (int or array): The index of the trial for which position data is to be loaded.

        Populates:
            self.pos_data (list): A list that stores position data for each trial. The position data for the specified trial is added at the given index.
        """  
        # Deal with int trial_iterator
        if isinstance(trial_list, int):
            trial_list = [trial_list]
            
        for trial_iterator in trial_list:
        
            # Check if position data is already loaded for session:
            if reload_flag == False and self.pos_data[trial_iterator] != None:
                print(f'Position data already loaded for trial {trial_iterator}')
            else:

                # Get path of trial to load
                path = f'{self.recording_path}/{self.trial_list[trial_iterator]}'

                if self.recording_type == 'nexus':
                    # Load position data from DACQ files

                    print(f'Loading pos file: {path}.pos')

                    # Read position data from csv file (faster)
                    data = pd.read_csv(f'{path}_pos.csv', index_col = 0).T

                    # Read header from pos file into dictionary
                    with open(f'{path}.pos', 'rb') as fid:
                        posHeader = {}

                        # Read the lines of the file up to the specified number (27 in this case) and write into dict
                        for _ in range(27):
                            line = fid.readline()
                            if not line:
                                break
                            elements = line.decode().strip().split()
                            posHeader[elements[0]] = ' '.join(elements[1:])
                            
                    # Get sampling rate
                    pos_sampling_rate = float(posHeader['sample_rate'][:-3])

                    # Extract LED position data and tracked pixel size data
                    led_pos = data.loc[:, ['X1', 'Y1', 'X2', 'Y2']]
                    led_pix = data.loc[:, ['Pixels LED 1', 'Pixels LED 2']]

                    # Set missing values to NaN
                    led_pos[led_pos == 1023] = np.nan
                    led_pix[led_pix == 1023] = np.nan

                    ## Scale pos data to specific PPM 
                    # Currently hard coded to 400 PPM
                    realPPM = int(posHeader['pixels_per_metre'])
                    posHeader['scaled_ppm'] = 400
                    goalPPM = 400

                    scale_fact = goalPPM / realPPM

                    # Scale area boundaries in place
                    posHeader['min_x'] = int(posHeader['min_x']) * scale_fact
                    posHeader['max_x'] = int(posHeader['max_x']) * scale_fact
                    posHeader['min_y'] = int(posHeader['min_y']) * scale_fact
                    posHeader['max_y'] = int(posHeader['max_y']) * scale_fact

                    # Scale pos data in place
                    led_pos['X1'] *= scale_fact
                    led_pos['X2'] *= scale_fact
                    led_pos['Y1'] *= scale_fact
                    led_pos['Y2'] *= scale_fact

                    # Collect header and data into a dict
                    raw_pos_data = {'header': posHeader,
                              'led_pos': led_pos.T,
                              'led_pix': led_pix.T}

                    # Postprocess posdata and return to self as dict
                    from postprocessing.postprocess_pos_data import process_position_data

                    xy_pos, led_pos, led_pix, speed, direction, direction_disp = process_position_data(raw_pos_data, self.max_speed, self.smoothing_window_size)
                    
                    # Divide by sampling rate to give dataframe columns in seconds
                    xy_pos.columns /= pos_sampling_rate
                    led_pos.columns/= pos_sampling_rate
                    led_pix.columns/= pos_sampling_rate
                    
                    # Populate processed pos data
                    self.pos_data[trial_iterator] = {
                        'xy_position': xy_pos,
                        'led_positions': led_pos,
                        'led_pixel_size': led_pix,
                        'speed': speed,
                        'direction': direction,
                        'direction_from_displacement': direction_disp,
                        'pos_sampling_rate': pos_sampling_rate
                    }

    def load_lfp(self, trial_iterator, sampling_rate, start_time, end_time, channels, reload_flag = False, bandpass_filter = False):
        """
        Loads the LFP (Local Field Potential) data for a specified trial. Currently from raw Dacq .bin files using the spikeinterface package

        Args:
            trial_iterator (int): The index of the trial for which LFP data is to be loaded.
            sampling_rate (int): The desired sampling rate for the LFP data.
            start_time (int): The start time from which LFP data is to be extracted.
            end_time (int): The end time until which LFP data is to be extracted.
            channels (list of int): A list of channel IDs from which LFP data is to be extracted.

        Populates:
            self.lfp_data (list): A list that stores LFP data for each trial. The LFP data for the specified trial is added at the given index.
        """        
        from spikeinterface.extractors import read_axona
        import spikeinterface.preprocessing as spre
        
        # Check if LFP is already loaded for session:
        if reload_flag == False and self.lfp_data[trial_iterator] != None:
                print(f'LFP data already loaded for trial {trial_iterator}')
                
        else:
            path = f'{self.recording_path}/{self.trial_list[trial_iterator]}.set'
            recording = read_axona(path)

            # Resample
            recording = spre.resample(recording, sampling_rate)

            if bandpass_filter:
                # Bandpass filter
                recording = spre.bandpass_filter(recording, freq_min = 1, freq_max = 300)

            lfp_data = recording.get_traces(start_frame = start_time*sampling_rate, end_frame = end_time*sampling_rate, channel_ids = channels)
            lfp_timestamps = recording.get_times()[start_time*sampling_rate:end_time*sampling_rate]

            self.lfp_data[trial_iterator] = {
            'data': lfp_data,
            'timestamps': lfp_timestamps,
            'sampling_rate': sampling_rate
            }
        
        
    def load_spikes(self, clusters_to_load = None):
        """
        Loads spike data for the entire recording session from phy output. Can select based on phy label. Also loads channel index for each cluters
        
        Args:
            quality (str or array): phy cluster label (if str) OR cluster IDs (if array) to load. Most likely 'good', but 'mua' or 'noise' also possible. If None, loads all clusters

        Populates:
            self.spike_data (dict): A dictionary that stores spike times, spike clusters, and cluster quality for the recording session.
        """
        import spikeinterface.extractors as se
        # Load spike times
        spike_times = np.load(f'{self.sorting_path}/spike_times.npy')

        # Load spike clusters
        spike_clusters = np.load(f'{self.sorting_path}/spike_clusters.npy')
        
        # Load spike templates
        spike_templates = np.load(f'{self.sorting_path}/spike_templates.npy')
        
        # Load cluster info
        cluster_info = pd.read_csv(f'{self.sorting_path}/cluster_info.tsv', sep='\t', index_col = 0)

        if clusters_to_load:
            
            if np.isscalar(clusters_to_load): #If string e.g. 'good'
                # Extract clusters matching quality to load
                cluster_info = cluster_info[cluster_info['group'] == clusters_to_load]
            
            else: #if array of cluster IDs
                # Extract only clusters matching the input array
                cluster_info = cluster_info[np.isin(cluster_info.index, clusters_to_load)]
            
            # Select spike times etc of the included clusters
            mask = np.isin(spike_clusters, cluster_info.index)
            spike_times = spike_times[mask]
            spike_clusters = spike_clusters[mask]
            spike_templates = spike_templates[mask]
            
        
        # Get sampling rate
        sort = se.read_phy(f'{self.sorting_path}').__dict__
        sampling_rate = sort['_sampling_frequency']
        
        ### Add a label for which behavioural trial each included spike is from
        
        # Convert spike times into seconds and flatten into a 1D array
        spike_times = np.divide(spike_times, sampling_rate).flatten()
            
        # Determine the trial for each spike
        spike_trial = np.digitize(spike_times.flatten(), self.trial_offsets) - 1
        

        # Populate spike_data
        self.spike_data = {
            'spike_times': spike_times,
            'spike_clusters': spike_clusters,
            'spike_templates': spike_templates,
            'spike_trial': spike_trial,
            'cluster_info': cluster_info,
            'sampling_rate': sampling_rate
            }
        
        


