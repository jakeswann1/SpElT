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


from ipywidgets import interact, widgets
from IPython.display import display

class SessionSelector:
    def __init__(self, session_dict):
        """
        Initialize the SessionSelector class.
        
        Parameters:
        - session_dict (dict): A dictionary where keys are session names and values are corresponding paths.
        """
        self.session_dict = session_dict
        self.path_to_session = None
        self._create_dropdown()
        
    def _create_dropdown(self):
        """
        Create and display a dropdown widget populated with the keys from the session_dict.
        """
        # Create the dropdown widget with keys from the session_dict
        self.dropdown_widget = widgets.Dropdown(
            options=self.session_dict.keys(),
            description='Select Session:',
            disabled=False,
        )
        
        # Connect the widget and the update function
        interact(self._update_path_to_session, selected_key=self.dropdown_widget)
        
    def _update_path_to_session(self, selected_key):
        """
        Update the instance variable path_to_session based on the selected dropdown key.
        
        Parameters:
        - selected_key (str): The key selected from the dropdown widget.
        """
        self.path_to_session = self.session_dict[selected_key]
        print(f"Value of selector.path_to_session has been set to: {self.path_to_session}")

