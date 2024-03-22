import pandas as pd
import os
import spikeinterface.extractors as se
from .axona_preprocessing import preprocess_axona, pos_from_bin

def collect_sessions(session_list, trial_list, sheet, probe_to_sort):
    recording_list = [[] for _ in range(len(session_list))]
    for i, session in enumerate(session_list):
        for trial in trial_list:
            if session in trial:
                base_folder = sheet[sheet['trial_name'] == trial]['path'].tolist()[0]
                num_channels = int(sheet[sheet['trial_name'] == trial]['num_channels'].tolist()[0])
                electrode_type = sheet[sheet['trial_name'] == trial]['probe_type'].tolist()[0]
                print(f"Loading {base_folder}/{trial}")

                if probe_to_sort == 'NP2_openephys':
                    recording = se.read_openephys(folder_path=f"{base_folder}/{trial}", stream_id = '0')
                    
                elif probe_to_sort == '5x12_buz':
                    recording = se.read_axona(f"{base_folder}/{trial}.set")
                    #Generate .pos file if not already present
                    if os.path.isfile(f'{base_folder}/{trial}.pos') == 0:
                        pos_from_bin(f'{base_folder}/{trial}')
                    #Preprocess recording
                    recording = preprocess_axona(recording = recording,
                                        recording_name = trial,
                                        base_folder = base_folder,
                                        electrode_type = electrode_type,
                                        num_channels = num_channels)
                       
                else:
                    raise ValueError('Probe type not recognized, currently only "NP2_openephys" and "5x12_buz" are supported.')
                
                trial_duration = recording.get_num_samples()
                print(trial_duration)
                recording_list[i].append([recording,
                                    trial, 
                                    base_folder, 
                                    electrode_type,
                                    trial_duration])
                    
    return recording_list