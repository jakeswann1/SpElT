def custom_tetrode_params():
    custom_klusta_params = {'adjacency_radius': None,
                 'threshold_strong_std_factor': 3.5,
                 'threshold_weak_std_factor': 1.5,
                 'detect_sign': -1,
                 'extract_s_before': 16,
                 'extract_s_after': 32,
                 'n_features_per_channel': 3,
                 'pca_n_waveforms_max': 10000,
                 'num_starting_clusters': 50,
                 'n_jobs': -1,
                 'total_memory': None,
                 'chunk_size': None,
                 'chunk_memory': None,
                 'chunk_duration': '1s',
                 'progress_bar': True
                } 
    return custom_klusta_params

def custom_probe_params():
    custom_klusta_params = {'adjacency_radius': None,
                 'threshold_strong_std_factor': 5,
                 'threshold_weak_std_factor': 2,
                 'detect_sign': -1,
                 'extract_s_before': 16,
                 'extract_s_after': 32,
                 'n_features_per_channel': 3,
                 'pca_n_waveforms_max': 10000,
                 'num_starting_clusters': 50,
                 'n_jobs': -1,
                 'total_memory': None,
                 'chunk_size': None,
                 'chunk_memory': None,
                 'chunk_duration': '1s',
                 'progress_bar': True
                } 
    return custom_klusta_params

def custom_ks2_params():
    custom_ks2_params = {'detect_threshold': 6,
                 'projection_threshold': [10, 4], 
                 'preclust_threshold': 8,
                 'car': True,
                 'minFR': 0.002,
                 'minfr_goodchannels': 0,
                 'freq_min': 300,
                 'sigmaMask': 30,
                 'nPCs': 3,
                 'ntbuff': 64,
                 'nfilt_factor': 4,
                 'NT': None,
                 'wave_length': 61,
                 'keep_good_only': False,
                 'n_jobs': -1,
                 'total_memory': None,
                 'chunk_size': None,
                 'chunk_memory': None,
                 'chunk_duration': '1s',
                 'progress_bar': True
                }
    return custom_ks2_params

import numpy as np
import pandas as pd
from probeinterface import generate_tetrode, ProbeGroup
from probeinterface import write_prb
from probeinterface.plotting import plot_probe, plot_probe_group

def generate_tetrodes(n):
    '''
    Returns a spikeinterface ProbeGroup object with n tetrodes spaced 300um apart vertically
    '''
    probegroup = ProbeGroup()
    for i in range(n):
        tetrode = generate_tetrode()
        tetrode.move([0, i * 300])
        tetrode.set_device_channel_indices(np.arange(i*4, i*4+4))
        probegroup.add_probe(tetrode)
    
    
    plot_probe_group(probegroup, with_channel_index = True)
    return probegroup

import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) #gets rid of some annoying warnings
from probeinterface import read_prb
from probeinterface.plotting import plot_probe, plot_probe_group
from pathlib import Path
import probeinterface.probe
import spikeinterface as si
import spikeinterface.preprocessing as spre

def preprocess_axona(recording, recording_name, base_folder, electrode_type, num_channels, force_rerun = False):
    '''
    Adds a Probe object to a Spikeinterface recording object
    Cuts the recording to 'num_channels' channels
    Saves the recording to a preprocessing folder
    '''
    preprocessing_folder = Path(f'{base_folder}/{recording_name}_preprocessed')

    probe_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'probes')

    if 'tetrode' in electrode_type:
        probe = generate_tetrodes(int(num_channels/4))
    #Load probes
    elif electrode_type == 'probe' or electrode_type == '32 ch four shanks':
        probe = read_prb(os.path.join(probe_folder, '4x8_buzsaki_oneshank.prb'))
    elif electrode_type == '5x12_buz':
        probe = read_prb(os.path.join(probe_folder, '5x12-16_buz.prb'))
        
    else:
        raise ValueError('Electrode type is set wrong, please set to either "probe" or "tetrode"')

    #plot_probe_group(probe, with_channel_index = True)
    #plt.savefig(f'{base_folder}/probe_layout.png')
    
    
    if (preprocessing_folder).is_dir() and force_rerun == False:
        recording = si.load_extractor(preprocessing_folder)
        print(f'{preprocessing_folder}\n{electrode_type} recording loaded from previous preprocessing\n {recording}')
        return recording
    else:
        channel_ids = recording.get_channel_ids()
        recording = recording.channel_slice(channel_ids=channel_ids[:num_channels]) #Cut to correct number of channels
#         recording = spre.bandpass_filter(recording, freq_min=360.0, freq_max=7000.0)
#         recording = spre.notch_filter(recording, freq = 50)

        ## Currently necessary as the probe is being treated as a single shank
        ## This turns the probe object from ProbeGroup to Probe
        if electrode_type == 'probe':
            singleProbe = probeinterface.Probe.from_dict(probe.to_dict()['probes'][0])
            recording = recording.set_probe(singleProbe)
        elif 'tetrode' in electrode_type:
            recording = recording.set_probegroup(probe, group_mode='by_probe')
        elif electrode_type == '5x12_buz':
            singleProbe = probeinterface.Probe.from_dict(probe.to_dict()['probes'][0])
            recording = recording.set_probe(singleProbe)

        recording.save(folder=preprocessing_folder, overwrite=True)
        print('Recording preprocessed and saved\n')
        return recording
    
import spikeinterface.sorters as ss
from IPython.core.display import HTML

def sort(recording, recording_name, base_folder, electrode_type, sorting_suffix):
    """
    Takes a preprocessed Spikeinterface recording object, and sorts using Klusta or KS2
    Saves the sorting to a folder in the base folder
    """    
    sorting_path = Path(f'{base_folder}/{recording_name[:6]}_{sorting_suffix}') 
    
    #Restart kernel so ipython can find the newly written files
    HTML("<script>Jupyter.notebook.kernel.restart()</script>")

    if (sorting_path).is_dir():
        try:
            sorting = si.load_extractor(sorting_path / 'sort')
            print(f"Sorting loaded from file {sorting_path}\n")
        except ValueError:
            print(f"Sorting at {sorting_path} failed to load - try deleting the folder and rerun")
            raise ValueError

    else:
        # Run klusta on tetrode recording using custom_tetrode_params above
        if 'tetrode' in electrode_type:
            sorting = ss.run_sorter('klusta', recording, output_folder=f'{sorting_path}',
                        verbose = True, docker_image = False, **custom_tetrode_params())
            print(f'Recording sorted!\n Klusta found {len(sorting.get_unit_ids())} units\n')
 
        # Run klusta on probe recording using custom_probe_params above
        elif electrode_type == 'probe' or electrode_type == '32 ch four shanks' or electrode_type == '5x12_buz':
            sorting = ss.run_sorter('kilosort2', recording, output_folder=f'{sorting_path}',
                                    verbose = True, docker_image = True, **custom_ks2_params()) 
            print(f'Recording sorted!\n KS2 found {len(sorting.get_unit_ids())} units\n')           
        else:
            print('Tetrode type set wrong')
        
        
        sorting = sorting.remove_empty_units()          
        sorting.save(folder=sorting_path / 'sort')
        print(f'Sorting saved to {sorting_path}/sort\n')
                
def get_mode(set_file):
    # Gets recording mode from channel 0 in set file
    # Assumes all channels are recorded in the same mode
    
    f = open(set_file, 'r')
    mode = f.readlines()[14][10]
    return mode
    
def concat_cluster_info(folder):
    # Concatenate cluster info files for tetrode recordings (once curated)
    import os 
    import pandas as pd
    
    # List cluster info files for each tetrode
    tsvs = []
    for i in os.listdir(folder):
        if 'cluster_info_' in i:
            tsvs.append(i)
    tsvs.sort()

    # Concatenate all units, plus which tetrode they were from, and save
    info_all = pd.DataFrame()
    for i in tsvs:
        info = pd.read_csv(f'{folder}/{i}', sep = '\t')
        info['tetrode'] = i[13]
        info_all = pd.concat([info_all, info])

    info_all.to_csv(f'{folder}/cluster_info_all.tsv', sep = '\t')


def pos_from_bin(path):
    '''
    Function for generating an Axona .pos position tracking file directly from raw recording data
    Input required is the path to the .set and .bin file (including the trial name, without the file extension, both files must be in the same directory)
    The .pos file will be written into this directory, and should be the same format as those generated by DacqUSB
    Jake Swann, 2023
    '''
    
    header = []
    with open(f'{path}.set', 'r') as f:
        set_file = f.readlines()

    trial_duration = round(float(set_file[4][9:])) #Default Axona conversion rounds to a whole number of seconds
    header =  [set_file[0][:-1]+'\r\n',
               set_file[1][:-1]+'\r\n',
               set_file[2][:-1]+'\r\n',
               set_file[3][:-1]+'\r\n',
               set_file[4][:9] + str(trial_duration) + '       \r\n',
               set_file[5][:-1]+'\r\n',
               'num_colours 4\r\n',
               'min_x 0\r\n',
               'max_x 768\r\n',
               'min_y 0\r\n',
               'max_y 574\r\n',
               'window_min_x ' + str(set_file[1059][5:-1])+'\r\n',
               'window_max_x ' + str(set_file[1060][5:-1])+'\r\n',
               'window_min_y ' + str(set_file[1061][5:-1])+'\r\n',
               'window_max_y ' + str(set_file[1062][5:-1])+'\r\n',
               'timebase 50 hz\r\n',
               'bytes_per_timestamp 4\r\n',
               'sample_rate 50.0 hz\r\n',
               'EEG_samples_per_position 5\r\n',
               'bearing_colour_1 ' + str(set_file[1420][15:-1]) +'\r\n',
               'bearing_colour_2 ' + str(set_file[1421][15:-1]) +'\r\n',
               'bearing_colour_3 ' + str(set_file[1422][15:-1]) +'\r\n',
               'bearing_colour_4 ' + str(set_file[1423][15:-1]) +'\r\n',
               'pos_format t,x1,y1,x2,y2,numpix1,numpix2\r\n',
               'bytes_per_coord 2\r\n',
               'pixels_per_metre ' + str(set_file[1099][25:-1])+'\r\n',
               'num_pos_samples ' + str(trial_duration*50) + '     \r\n', #Camera sample rate hard coded as 50 Hz
               'data_start'
                ]

    # read in data and extract position data
    with open(f'{path}.bin', 'rb') as f:
        data = f.read()

    position_data = []
    packetnums = []
    timestamps = []
    x1s = []
    y1s = []
    x2s = []
    y2s = []
    numpix1s = []
    numpix2s = []
    totalpixs = []

    for i in range(432, len(data)-432, 432): #skip first packet because it can be bugged (says Jim)
        packet = data[i:i+432]
        packet_id = packet[0:4].decode('ascii')

        if packet_id == 'ADU2':
            pos_sample = packet[12:32]

            #Flip digits in pos data for saving (this is how they are in the .pos relative to the .bin)
            pos_first = pos_sample[::2]
            pos_second = pos_sample[1::2]
            pos_sample = []
            for n, i in enumerate(pos_first):
                pos_sample.append(pos_second[n])
                pos_sample.append(pos_first[n])
                
            packetnums.append(int.from_bytes(bytes(pos_sample[0:2]), 'big'))
            timestamps.append(int.from_bytes(bytes(pos_sample[2:4]), 'big'))
            y1s.append(int.from_bytes(bytes(pos_sample[4:6]), 'big'))
            x1s.append(int.from_bytes(bytes(pos_sample[6:8]), 'big'))
            y2s.append(int.from_bytes(bytes(pos_sample[8:10]), 'big'))
            x2s.append(int.from_bytes(bytes(pos_sample[10:12]), 'big'))
            numpix1s.append(int.from_bytes(bytes(pos_sample[12:14]), 'big'))
            numpix2s.append(int.from_bytes(bytes(pos_sample[14:16]), 'big'))
            totalpixs.append(int.from_bytes(bytes(pos_sample[16:18]), 'big'))
            
            # the X and Y values are reversed piece-wise so lets switch the format from
            # packet #, video timestamp, y1, x1, y2, x2, numpix1, numpix2, total_pix, unused value
            # to
            # packet #, video timestamp, x1, y1, x2, y2, numpix1, numpix2, total_pix, unused value
            y1 = pos_sample[4:6]
            x1 = pos_sample[6:8]
            y2 = pos_sample[8:10]
            x2 = pos_sample[10:12]
            
            pos_sample[4:6] = x1
            pos_sample[6:8] = y1
            pos_sample[8:10] = x2
            pos_sample[10:12] = y2
            
            pos_sample = bytes(pos_sample)

    #         if len(position_data) < (trial_duration*50*2) +1:
            position_data.append(pos_sample)

    # Drop every second sample because pos data is double-counted in the .bin file
    position_data = position_data[::2]

    # save position data to csv
    pos_df = pd.DataFrame([packetnums, timestamps, x1s, x2s, y1s, y2s, numpix1s, numpix2s, totalpixs], 
                          index = ['Packet Number', 'Timestamps', 'X1', 'X2', 'Y1', 'Y2', 'Pixels LED 1', 'Pixels LED 2', 'Total Pixels'])
    pos_df = pos_df.iloc[:, ::2]
    pos_df = pos_df.set_axis(range(pos_df.shape[1]), axis = 1)
    pos_df.to_csv(f'{path}_pos.csv')

    # save position data to binary file
    with open(f'{path}.pos', 'wb') as f:
        for line in header:
            f.write(line.encode())
        for pos_sample in position_data:
            f.write(pos_sample)
        f.write('\r\ndata_end\r\n'.encode())

        f.close()
    