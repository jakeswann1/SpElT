import pandas as pd
import numpy as np

from .ephys_utils import gs_to_df
from .postprocessing.postprocess_pos_data import process_position_data

class ephys:
    '''
    A class to manage ephys data, including metadata, position, LFP, and spike data recorded from (currently): 
    - raw DACQ recordings sorted with Kilosort 2 and curated with phy
    - Neuropixels 2 recordings acquired with OpenEphys, sorted with Kilosort 4 and curated with phy, with video tracking data from Bonsai
    
    Assumes a basic file structure of path_to_data/animal/YYYY-MM-DD/ for each session

    Usage:
        # Initialize the class with recording type and optional path to session.
        obj = ephys('nexus', 'path/to/recording/data/animalID/date')
        obj = ephys('nexus') will prompt a box to select the recording folder
        

        # Load metadata for a list of trials.
        obj.load_metadata([0, 2, 3])

        # Load position data for a list of trials.
        obj.load_pos([0, 1, 2])

        # Load LFP data for a list of trials with a specific sampling rate and channels.
        obj.load_lfp([0, 1, 2], sampling_rate = 1000, channels = [0, 1, 2])

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

    def __init__(self, recording_type, path, sheet_url = None):
        """
        Initialize the ephys object.

        Parameters:
        recording_type (str): The type of recording.
        path (str): The path to the recording.

        Raises:
        ValueError: If no path is specified.

        """
        
        self.recording_type = recording_type
        
        if path:
            self.recording_path = path
        else:
            raise ValueError('No path specified')
        
        # Get date and animal ID from path
        self.date = self.recording_path.split('/')[-1]
        self.date_short = f'{self.date[2:4]}{self.date[5:7]}{self.date[8:10]}'
        self.animal = self.recording_path.split('/')[-2]
         
        # Get age and probe info from Google Sheet
        try:
            df = gs_to_df(sheet_url)
        except:
            print('Google Sheet not found, please specify a valid URL')
            
        self.age = int(df['Age'].loc[df['Session'] == f'{self.animal}_{self.date_short}'].iloc[0])
        self.probe_type = df['probe_type'].loc[df['Session'] == f'{self.animal}_{self.date_short}'].iloc[0]
        self.probe_channels = df['num_channels'].loc[df['Session'] == f'{self.animal}_{self.date_short}'].iloc[0]

        if self.recording_type == 'nexus':
            self.sorting_path = f'{self.recording_path}/{self.date_short}_sorting_ks2_custom'
        elif self.recording_type == 'NP2_openephys':
            self.sorting_path = f'{self.recording_path}/{self.date_short}_sorting_ks4'
        else:
            raise ValueError('Recording type not recognised, please specify "nexus" or "NP2_openephys"')
        
        # Load session information from session.csv which is within the sorting folder as dataframe
        self.session = pd.read_csv(f'{self.sorting_path}/session.csv', index_col = 0)
        
        # Collect each trial name
        self.trial_list = self.session.iloc[:,1].to_list()
        # Collect each trial number
        self.trial_iterators = [i for i, _ in enumerate(self.trial_list)]
        
        # Collect trial offsets for aligning spike data
        durations = self.session.iloc[:,4].to_list()
        self.trial_offsets = []
        offset = 0
        for duration in durations:
            self.trial_offsets.append(offset)
            offset += duration

        # Initialise data variables
        self.metadata = [None] * len(self.trial_list)
        self.lfp_data = [None] * len(self.trial_list)
        self.pos_data = [None] * len(self.trial_list)
        self.spike_data = [None]
        
        
        # Set constants for position processing
        self.max_speed = 5
        self.smoothing_window_size = 3
    
    def load_metadata(self, trial_list, output_flag = True):
        """
        Loads the metadata for a specified trial. Currently only a subset of lines from the Dacq .set file

        Args:
            trial_iterator (int): The index of the trial for which metadata is to be loaded.
            output_flag (bool): if True, print a statement when loading the set file (default True)

        Populates:
            self.metadata (list): A list that stores metadata for each trial. The metadata for the specified trial is added at the given index.
        """  
        
        # Deal with int trial_list
        if isinstance(trial_list, int):
            trial_list = [trial_list]
            
        for trial_iterator in trial_list:
            if self.recording_type == 'nexus':
                # Get path of trial to load
                path = f'{self.recording_path}/{self.trial_list[trial_iterator]}.set'
                
                if output_flag is True:
                    print(f'Loading set file: {path}')

                with open(path, 'rb') as fid:
                    setHeader = {}

                    # Read the lines of the file up to the specified number (5 in this case) and write into dict
                    while True:
                        line = fid.readline()
                        if not line:
                            break
                        elements = line.decode().strip().split()
                        setHeader[elements[0]] = ' '.join(elements[1:])

                # Add sampling rate - HARDCODED FOR NOW
                setHeader['sampling_rate'] = 48000

                #Populate basic metdata       
                self.metadata[trial_iterator] = setHeader

    def load_pos(self, trial_list, output_flag = True, reload_flag = False):
        """
        Loads  and postprocesses the position data for a specified trial. Currently only from Dacq .pos files

        Args:
            trial_list (int or array): The index of the trial for which position data is to be loaded.

        Populates:
            self.pos_data (list): A list that stores position data for each trial. The position data for the specified trial is added at the given index.
        """  
        # Deal with int trial_list
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

                    if output_flag:
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

                    # TERRIBLE HACK - NEEDS FIXING PROPERLY
                    # If t-maze trial, rescale PPM because it isn't set right in pos file 
                    if 't-maze' in self.trial_list[trial_iterator]:
                        realPPM = 615
                        posHeader['pixels_per_metre'] = 615
                        if output_flag:
                            print(f'Real PPM artifically set to 615 (t-maze default)')

                    
                    posHeader['scaled_ppm'] = 400
                    goalPPM = 400
                    scale_fact = goalPPM / realPPM

                    # Scale area boundaries in place
                    posHeader['min_x'] = int(posHeader['window_min_x']) * scale_fact
                    posHeader['max_x'] = int(posHeader['window_max_x']) * scale_fact
                    posHeader['min_y'] = int(posHeader['window_min_y']) * scale_fact
                    posHeader['max_y'] = int(posHeader['window_max_y']) * scale_fact

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
                    xy_pos, led_pos, led_pix, speed, direction, direction_disp = process_position_data(raw_pos_data, self.max_speed, self.smoothing_window_size)
                    
                    # Divide by sampling rate to give dataframe columns in seconds
                    xy_pos.columns /= pos_sampling_rate
                    led_pos.columns /= pos_sampling_rate
                    led_pix.columns /= pos_sampling_rate
                    
                    # Populate processed pos data
                    self.pos_data[trial_iterator] = {
                        'header': posHeader,
                        'xy_position': xy_pos,
                        'led_positions': led_pos,
                        'led_pixel_size': led_pix,
                        'speed': speed, #in cm/s
                        'direction': direction,
                        'direction_from_displacement': direction_disp,
                        'pos_sampling_rate': pos_sampling_rate
                    }
                    

    def load_lfp(self, trial_list, sampling_rate, channels = None, scale_to_uv = True, reload_flag = False, bandpass_filter = None):
        """
        Loads the LFP (Local Field Potential) data for a specified trial. Currently from raw Dacq .bin files using the spikeinterface package
        Masks clipped values and scales to microvolts based on the gain in the .set file

        Args:
            trial_list (int or array-like): The index of the trial for which LFP data is to be loaded.
            sampling_rate (int): The desired sampling rate for the LFP data.
            channels (list of int, optional): A list of channel IDs from which LFP data is to be extracted. Default is all
            scale_to_uv (bool, optional): choose whether to scale raw LFP trace to microvolts based on the gain in the .set file. Default True
            reload_flag (bool, optional): if true, forces reloading of data. If false, only loads data for trials with no LFP data loaded. Default False
            bandpass_filter (2-element array, optional): apply bandpass filter with min and max frequency. Default None. e.g. [5 100] would bandpass filter @ 5-100Hz

        Populates:
            self.lfp_data (list): A list that stores LFP data for each trial. The LFP data for the specified trial is added at the given index.
        """        
        from spikeinterface.extractors import read_axona
        import spikeinterface.preprocessing as spre
        
        # Deal with int trial_list
        if isinstance(trial_list, int):
            trial_list = [trial_list]
            
        for trial_iterator in trial_list:

            # Check if LFP is already loaded for session:
            if reload_flag == False and self.lfp_data[trial_iterator] != None:
                    print(f'LFP data already loaded for trial {trial_iterator}')

            else:
                path = f'{self.recording_path}/{self.trial_list[trial_iterator]}.set'
                recording = read_axona(path)

                # Resample
                recording = spre.resample(recording, sampling_rate)

                

                if bandpass_filter is not None:
                    # Bandpass filter
                    recording = spre.bandpass_filter(recording, 
                                                     freq_min = bandpass_filter[0], 
                                                     freq_max = bandpass_filter[1])

                # Set channels to load to list of str to match recording object - not ideal but other fixes are harder
                if channels is not None:
                    channels = list(map(str, channels))
                    
                lfp_data = recording.get_traces(start_frame = None, end_frame = None, channel_ids = channels).astype(float)
                
                lfp_timestamps = recording.get_times()

                # Mask clipped values of +- 32000
                clip_mask = np.logical_or(lfp_data > 32000, lfp_data < -32000)
                
                # Scale traces to uv - method taken from getLFPV.m by Roddy Grieves 2018
                # Raw file samples are stored with 16-bit resolution, so range from -32768 to 32767
                if scale_to_uv is True:
                    # Load .set metadata
                    self.load_metadata(trial_iterator, output_flag = False)
                    set_header = self.metadata[trial_iterator]
                    # Get ADC for recording
                    adc = int(set_header['ADC_fullscale_mv'])
                    
                    # Get channel gains
                    gains = np.empty(len(channels))
                    for n, channel in enumerate(channels):
                        gains[n] = set_header[f'gain_ch_{channel}']
                    
                    # Scale traces
                    lfp_data = lfp_data / 32768 * adc * 1000 
                    lfp_data = lfp_data / gains.T
                        
                else:
                    gains = None

                self.lfp_data[trial_iterator] = {
                'data': lfp_data,
                'timestamps': lfp_timestamps,
                'clip_mask': clip_mask,
                'sampling_rate': sampling_rate,
                'channels': channels,
                'gains': gains
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

        if clusters_to_load is not None:
            
            # Case if clusters to load is an empty array:
            if len(clusters_to_load) == 0:
                spike_times = np.nan
                spike_clusters = np.nan
                spike_templates = np.nan
            
            else:
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
        
        
    def load_mean_waveforms(self, clusters_to_load = None, scale = True):
        """
        Load waveforms for specified clusters.

        Args:
            clusters_to_load (list or None): List of cluster IDs to load waveforms for. If None, waveforms will be loaded for all clusters.
            n_spikes (int): Number of spikes to load for each cluster.
            scale (bool): Flag indicating whether to scale the waveforms to microvolts.

        Populates:
            self.waveform_data (dict): A dictionary that stores mean waveforms for specified clusters.
        """
        from phylib.io.model import load_model

        # Initialise waveform_data
        self.mean_waveforms = {}

        # Get path to params.py
        params_path = f'{self.sorting_path}/params.py'

        # Load the TemplateModel.
        model = load_model(params_path)

        # Load metadata for scaling traces (gains will load from the first trial)
        self.load_metadata(0, output_flag=False)
        set_header = self.metadata[0]
        # Get ADC for recording
        adc = int(set_header['ADC_fullscale_mv'])

        if clusters_to_load is None:
            clusters_to_load = self.spike_data['cluster_info'].index
            print('No clusters specified, loading mean waveforms for all clusters')

        for cluster in clusters_to_load:
            
            # Get the best mean waveform for the cluster from phy
            best_mean_waveform = model.get_cluster_mean_waveforms(cluster)['mean_waveforms'][:,0]
            best_channel = model.get_cluster_mean_waveforms(cluster)['channel_ids'][0]
            
            # Scale the waveforms to microvolts.
            if scale is True:
                # Get channel gains
                gain = int(set_header[f'gain_ch_{best_channel}'])
                # Scale traces to uv using logic as in ephys.load_lfp
                best_mean_waveform = (best_mean_waveform / 32768 * adc * 1000) / gain

            # Populate waveform_data
            self.mean_waveforms[cluster] = best_mean_waveform

            