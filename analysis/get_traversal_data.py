import numpy as np
import pandas as pd

def get_traversal_cycles(arm_times, cycle_numbers, lfp_timestamps, lfp_sampling_rate):
    '''
    Finds theta cycle IDs for individual arm traversals
    Returns an array of arrays, each containing theta cycle IDs for a single traversal
    E.g. [[1,2,3], [6,7,8]]
    '''
    
    # Find continuous segments in central_times and return_times 
    # (ie where there is > 0.02 seconds between one sample and the next)
    # This should correspond to individual arm traversals
    arm_segments = np.split(arm_times, np.where(np.diff(arm_times) > 0.025)[0] + 1)
    # These are then stored as a list of time ranges for interfacing with LFP
    traversals = np.array([(segment[0], segment[-1]) for segment in arm_segments if len(segment) > 1])
    
    # Find whole theta cycles within each traversal
    # For each traversal
    arm_cycle_numbers = []
    for count, traversal in enumerate(traversals):

        # Multiply times by lfp_sampling rate
        traversal = (traversal * lfp_sampling_rate).astype(int)

        # Get timestamps and unique cycle numbers found in this time window
        traversal_cycle_numbers = np.unique(cycle_numbers[traversal[0]:traversal[1]])

        # Discard min and max cycle number from traversal as these will likely be incomplete
        traversal_cycle_numbers = traversal_cycle_numbers[1:-1]

        # Add included cycles to array
        arm_cycle_numbers.append(traversal_cycle_numbers)

    return arm_cycle_numbers

def get_data_for_traversals(arm_traversal_cycles, cycle_numbers, lfp_data, speed_data, channels_to_load, theta_phase, lfp_timestamps):
    '''
    Makes a dataframe of LFP data from all traversals in a given arm.
    Columns are timestamps of LFP data samples
    Indices are:
    - Channel IDs (containing LFP data samples, one row for each channel loaded)
    - Cycle Theta Phase (taken from channel 35 at 0um)
    - Cycle Index (index of theta cycle for each LFP sample
    - Speed data interpolated up to match LFP sampling rate
    - Traversal Index (index of traversal in that arm, counting from 0 for each trial)
    
    '''
    #Initialise variables
    channels_to_load = [str(i) for i in channels_to_load] # Convert channel array into string to preserve ordering by depth
    traversal_df = pd.DataFrame(index = channels_to_load)
    traversal_index = 0

    
    # For each theta cycle for the given arm group, get LFP trace, theta phases and timestamp
    for traversal in arm_traversal_cycles:
        
        mask = np.isin(cycle_numbers, traversal).flatten()
        
        cycle_data = lfp_data[mask, :].T
        cycle_phases = theta_phase[mask, :]
        cycle_timestamps = lfp_timestamps[mask]
        cycle_id = cycle_numbers[mask]
        cycle_speed = speed_data[mask]
        
        # Add traversal data to dataframe:
        traversal_df.loc[channels_to_load, cycle_timestamps] = cycle_data #Add LFP data for each included channel
        traversal_df.loc['Cycle Theta Phase', cycle_timestamps] = cycle_phases.flatten() # Add theta phase for each sample
        traversal_df.loc['Cycle Index', cycle_timestamps] = cycle_id.flatten() # Add theta cycle index
        traversal_df.loc['Traversal Index', cycle_timestamps] = np.ones(len(cycle_timestamps)) * traversal_index # Add traversal index
        traversal_df.loc['Speed', cycle_timestamps] = cycle_speed
        
        traversal_index += 1
        
        # # Make a dict with structure: cycle_number: dataframe of data
        # cycle_dict[tuple(traversal)] = cycle_frame
        
    return traversal_df

def drop_extreme_cycles(df):
    """
    Drops columns with the lowest and highest 'Cycle Index' value for each unique 'Traversal Index'.

    Args:
        df (pd.DataFrame): DataFrame with 'Cycle Index' and 'Traversal Index' as rows.

    Returns:
        pd.DataFrame: A new DataFrame with specified columns dropped.
    """

    # Find the rows for 'Cycle Index' and 'Traversal Index'
    cycle_index_row = df.index[df.index == 'Cycle Index'][0]
    traversal_index_row = df.index[df.index == 'Traversal Index'][0]

    # Convert these rows to columns for easy processing
    df_transposed = df.T
    df_transposed['Cycle Index'] = df_transposed[cycle_index_row]
    df_transposed['Traversal Index'] = df_transposed[traversal_index_row]

    # Group by 'Traversal Index' and find columns to drop
    columns_to_drop = []
    for name, group in df_transposed.groupby('Traversal Index'):
        min_cycle_index_col = group['Cycle Index'].astype(float).idxmin()
        max_cycle_index_col = group['Cycle Index'].astype(float).idxmax()
        columns_to_drop.extend([min_cycle_index_col, max_cycle_index_col])

    # Drop the identified columns
    df_dropped = df_transposed.drop(columns=columns_to_drop).drop(['Cycle Index', 'Traversal Index'], axis=1)

    # Transpose back to the original format
    return df_dropped.T