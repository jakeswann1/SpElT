import pandas as pd
import numpy as np

class ephys:
    '''
    A class to manage ephys data, including metadata, position, LFP, and spike data recorded from (currently): 
    - raw DACQ recordings sorted with Kilosort 2 and curated with phy
    
    Assumes a basic file structure of path_to_data/animal/YYYY-MM-DD/ for each session

    Usage:
        # Initialize the class with animal name, recording date, and recording type.
        obj = ephys('r0001', '2023-01-01', 'nexus')

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

    def __init__(self, animal, date, recording_type):
        """
        Initializes the ephys object with given animal, date, and recording type.

        Args:
            animal (str): Name or ID of the animal.
            date (str): Date of the recording in 'YYYY-MM-DD' format.
            recording_type (str): Type of the recording (e.g., 'nexus').
        """
        self.recording_type = recording_type
        
        self.base_path = '/home/isabella/Documents/isabella/jake/recording_data' #Hard-coded for now
        self.recording_path = f'{self.base_path}/{animal}/{date}'
        self.sorting_path = f'{self.recording_path}/{date[2:4]}{date[5:7]}{date[8:10]}_sorting_ks2_custom'
        
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

            #Populate basic metdata       
            self.metadata[trial_iterator] = setHeader

    def load_pos(self, trial_iterator):
        
        # Get path of trial to load
        path = f'{self.recording_path}/{self.trial_list[trial_iterator]}'
        
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
        from postprocess_pos_data import process_position_data

        xy_pos, led_pos, led_pix, speed, direction, direction_disp = process_position_data(raw_pos_data, self.max_speed, self.smoothing_window_size)

        # Populate processed pos data
        self.pos_data[trial_iterator] = {
            'XY Position': xy_pos,
            'LED Positions': led_pos,
            'LED Pixel Size': led_pix,
            'Speed': speed,
            'Direction': direction,
            'Direction from displacement': direction_disp
        }

    def load_lfp(self, trial_iterator, sampling_rate, start_time, end_time, channels):
        
        from spikeinterface.extractors import read_axona
        import spikeinterface.preprocessing as spre
        path = f'{self.recording_path}/{self.trial_list[trial_iterator]}.set'
        recording = read_axona(path)

        recording = spre.resample(recording, sampling_rate)
        recording = spre.bandpass_filter(recording, freq_min = 1, freq_max = 300)
        lfp_data = recording.get_traces(start_frame = start_time*sampling_rate, end_frame = end_time*sampling_rate, channel_ids = channels)
        lfp_timestamps = recording.get_times()[start_time*sampling_rate:end_time*sampling_rate]

        self.lfp_data[trial_iterator] = {
        'data': lfp_data,
        'timestamps': lfp_timestamps,
        'sampling rate': sampling_rate
        }
        
        
    def load_spikes(self):

        # Load spike times
        spike_times = np.load(f'{self.sorting_path}/spike_times.npy')

        # Load spike clusters
        spike_clusters = np.load(f'{self.sorting_path}/spike_clusters.npy')

        # Load cluster groups (quality)
        cluster_df = pd.read_csv(f'{self.sorting_path}/cluster_group.tsv', sep='\t')
        cluster_quality = dict(zip(cluster_df['cluster_id'], cluster_df['group']))

        # Populate spike_data
        self.spike_data = {
            'spike times': spike_times,
            'spike clusters': spike_clusters,
            'cluster_quality': cluster_quality
        }
