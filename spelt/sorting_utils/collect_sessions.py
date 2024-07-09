import os
import pandas as pd
import spikeinterface.extractors as se
from ..axona_utils.axona_preprocessing import preprocess_axona, pos_from_bin

def parse_session_info(session):
    parts = session.split("_")
    base_session = f"{parts[0]}_{parts[1]}"
    area = parts[2] if len(parts) > 2 else None
    return base_session, area

def load_and_process_recording(trial_info, trial, probe_to_sort, base_folder, area=None):

    if probe_to_sort == 'NP2_openephys':
        return se.read_openephys(folder_path=f"{base_folder}/{trial}/{area}", stream_id='0')

    elif probe_to_sort == '5x12_buz':
        recording = se.read_axona(f"{base_folder}/{trial}.set")
        if not os.path.isfile(f'{base_folder}/{trial}.pos'):
            pos_from_bin(f'{base_folder}/{trial}')
        return preprocess_axona(recording=recording, recording_name=trial, base_folder=base_folder,
                                electrode_type=trial_info['probe_type'])
    else:
        raise ValueError('Probe type not recognized, currently only "NP2_openephys" and "5x12_buz" are supported.')

def collect_trial_info(sheet, trial):

    trial_data = sheet[sheet['trial_name'] == trial].iloc[0]
    trial_info = {
        'path': trial_data['path'],
        'probe_type': trial_data['probe_type']
    }
    # Check for 'Areas' column and include it if present
    trial_info['area'] = trial_data['Areas'] if 'Areas' in sheet.columns else None
    return trial_info

def collect_sessions(session_list, trial_list, sheet, probe_to_sort, area_list):
    """
    Collects recordings from a list of sessions and trials. The function will return a list of lists, where each
    sublist corresponds to a session and contains the recordings for that session.

    Parameters
    ----------
    session_list : list
        List of session names.
    trial_list : list
        List of trial names.
    sheet : pandas.DataFrame
        Dataframe containing the trial information.
    probe_to_sort : str
        Probe type to sort. Currently only 'NP2_openephys' and '5x12_buz' are supported.
    area_list : list
        List of areas to include. If None, all areas will be included.

    Returns
    -------
    recording_list : list
        List of lists containing the recordings for each session.

    """
    recording_list = [[] for _ in session_list]

    for i, session in enumerate(session_list):

        base_session, area = parse_session_info(session)
        for trial in trial_list:
            if area_list is None or (area in area_list[i] and base_session in trial):
                trial_info = collect_trial_info(sheet, trial)
                base_folder = trial_info['path']
                print(f"Loading {base_folder}/{trial}")

                recording = load_and_process_recording(trial_info, trial, probe_to_sort, base_folder, area)
                trial_duration = recording.get_num_samples()

                recording_data = [recording, trial, base_folder, trial_info['probe_type'], trial_duration]
                if area_list is not None:
                    recording_data.append(area)
                recording_list[i].append(recording_data)

    return recording_list
