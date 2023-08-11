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

def gs_to_df(url):
    import pandas as pd
    
    csv_export_url = url.replace('/edit#gid=', '/export?format=csv&gid=')
    df = pd.read_csv(csv_export_url, on_bad_lines = 'skip')
    return df

def fullprint(*args, **kwargs):
    from pprint import pprint
    import numpy
    opt = numpy.get_printoptions()
    numpy.set_printoptions(threshold=numpy.inf)
    pprint(*args, **kwargs)
    numpy.set_printoptions(**opt)
    
def pickle_variable(var, var_name):
    """
    This function pickles any python variable and stores it in a file named '[variableName].pkl'.

    Parameters:
    var: Python variable to be pickled
    var_name: Name of the variable as a string. The variable will be stored in a file named '[var_name].pkl'.
    """
    import pickle
    with open(f'{var_name}.pkl', 'wb') as f:
        pickle.dump(var, f)

def load_pickle(file_path):
    """
    This function loads a pickle file from 'file_path' (expected .pkl file).
    """
    import pickle
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def generate_tetrodes(n):
    # Returns a spikeinterface ProbeGroup object with n tetrodes spaced 300um apart vertically
    
    import numpy as np
    from probeinterface import generate_tetrode, ProbeGroup
    from probeinterface import write_prb
    
    probegroup = ProbeGroup()
    for i in range(n):
        tetrode = generate_tetrode()
        tetrode.move([0, i * 300])
        tetrode.set_device_channel_indices(np.arange(i*4, i*4+4))
        probegroup.add_probe(tetrode)
    
    
    from probeinterface.plotting import plot_probe, plot_probe_group
    plot_probe_group(probegroup, with_channel_index = True)
    return probegroup

def preprocess(recording, recording_name, base_folder, electrode_type, num_channels):
    # Adds a Probe object to a Spikeinterface recording object
    # Cuts the recording to 'num_channels' channels
    # Saves the recording to a preprocessing folder
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) #gets rid of some annoying warnings
    from probeinterface import read_prb
    from probeinterface.plotting import plot_probe, plot_probe_group
    from pathlib import Path
    import probeinterface.probe
    import spikeinterface as si
    import spikeinterface.preprocessing as spre
    
    preprocessing_folder = Path(f'{base_folder}/{recording_name}_preprocessed')

    if 'tetrode' in electrode_type:
#         probe = read_prb('/home/isabella/Documents/isabella/klusta_testdata/spikeinterface/probes/8_tetrodes.prb') #Load probe
        probe = generate_tetrodes(int(num_channels/4))

    #Load probe
    elif electrode_type == 'probe' or electrode_type == '32 ch four shanks':
        probe = read_prb('/home/isabella/Documents/isabella/klusta_testdata/spikeinterface/probes/4x8_buzsaki_oneshank.prb') 
    elif electrode_type == '5x12_buz':
        probe = read_prb('/home/isabella/Documents/isabella/klusta_testdata/spikeinterface/probes/5x12-16_buz.prb')
        
    else:
        raise ValueError('Electrode type is set wrong, please set to either "probe" or "tetrode"')

    #plot_probe_group(probe, with_channel_index = True)
    #plt.savefig(f'{base_folder}/probe_layout.png')

    if (preprocessing_folder).is_dir():
        print(preprocessing_folder)
        recording = si.load_extractor(preprocessing_folder)
        print(f'{electrode_type} recording loaded from previous preprocessing\n')
        print(recording)
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

        recording_saved = recording.save(folder=preprocessing_folder)
        print('Recording preprocessed and saved\n')
        return recording
    

def sort(recording, recording_name, base_folder, electrode_type, sorting_suffix):
    # Takes a preprocessed Spikeinterface recording object, and sorts using Klusta
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning) #gets rid of some annoying warnings
    from pathlib import Path
    import spikeinterface as si
    import spikeinterface.sorters as ss
    from probeinterface import read_prb
    
#     recordings = recording.split_by(property='group', outputs='dict')
    
    if 'tetrode' in electrode_type:
        sorter_params = custom_tetrode_params()
        sorter = 'klusta'
    else:
        sorter_params = custom_probe_params()
        sorter = 'klusta'
    
    sorting_path = Path(f'{base_folder}/{recording_name[:6]}_{sorting_suffix}') 
    
    #Restart kernel so ipython can find the newly written files
    from IPython.core.display import HTML
    HTML("<script>Jupyter.notebook.kernel.restart()</script>")

    if (sorting_path).is_dir():
        sorting = si.load_extractor(sorting_path / 'sort')
        print(f"Sorting loaded from file {sorting_path}\n")

    else:
        if 'tetrode' in electrode_type:
        # Run klusta on tetrode recording using custom_tetrode_params above
            sorting = ss.run_sorter('klusta', recording, output_folder=f'{sorting_path}',
                        verbose = True, docker_image = False, **custom_tetrode_params())
 
        elif electrode_type == 'probe' or electrode_type == '32 ch four shanks' or electrode_type == '5x12_buz':
        # Run klusta on probe recording using custom_probe_params above
            sorting = ss.run_sorter('kilosort2', recording, output_folder=f'{sorting_path}',
                                    verbose = True, docker_image = True, **custom_ks2_params())            
        else:
            print('Tetrode type set wrong')
        
        print(f'Recording sorted!\n Klusta found {len(sorting.get_unit_ids())} units\n')
        sorting = sorting.remove_empty_units()          
        sorting_saved = sorting.save(folder=sorting_path / 'sort')
        print(f'Sorting saved to {sorting_path}/sort\n')
            


    #raster = si.widgets.plot_rasters(sorting)
    
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

