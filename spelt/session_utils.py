import numpy as np

def gs_to_df(url):
    import pandas as pd
    
    csv_export_url = url.replace('/edit#gid=', '/export?format=csv&gid=')
    df = pd.read_csv(csv_export_url, on_bad_lines = 'skip')
    return df

def find_all_sessions(sheet_path, data_path, sorting_suffix):
    '''
    Function to find all sessions and session paths from Recording Master Spreadsheet
    '''
    
    sheet = gs_to_df(sheet_path)

    sheet['path'] = data_path + sheet['path']

    sheet_inc = sheet[sheet['Include'] == 'Y']
    trial_list = sheet_inc['trial_name'].to_list()
    session_list = np.unique([f"{i.split('_')[0]}_{i.split('_')[1]}" for i in trial_list])
    session_dict = {}
    
    for i in session_list:
        animal = i[-5:]
        date_short = i[:6]
        date_long = f'20{i[:2]}-{i[2:4]}-{i[4:6]}'
        
        path_to_session = f'{data_path}/{animal}/{date_long}'
        
        session_dict[i] = path_to_session
    
    return session_dict

from .ephys import ephys
import pandas as pd

def make_df_all_sessions(session_dict, recording_type = 'nexus'):
    '''
    Function to make a dataframe of all sessions and their paths
    '''
    
    # Initialise DataFrame for ephys objects
    df_all_sessions = pd.DataFrame(data = None, index = session_dict.keys(), columns = ['ephys_object'], dtype='object')

    for i, session_path in enumerate(session_dict.values()):
        # Create ephys object for session and add to dataframe
        obj = ephys(recording_type = recording_type, path = session_path)
        df_all_sessions.at[list(session_dict.keys())[i], 'ephys_object'] = obj

    return df_all_sessions