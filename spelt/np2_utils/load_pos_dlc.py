import numpy as np
import pandas as pd

def load_pos_dlc(path, ppm):
    """
    Load position data from DeepLabCut-generated .csv file
    Assumes points 1 and 2 are on the head and neck for calculating orientation

    Parameters
    ----------
    path : str
        Path to the .csv file to be loaded
    ppm : int, optional
        Pixels per metre for scaling

    """

    # Read position data from csv file
    data = pd.read_csv(path, index_col = 0).T
    # Combine bodyparts and coords into a single label 'bodypart1_x'
    data['Label'] = data.apply(lambda row: f"{row['bodyparts']}_{row['coords']}", axis=1)
    data.drop(columns=['bodyparts', 'coords'], inplace=True)
    data.set_index('Label', inplace=True)

    pos_header = {'pixels_per_metre': ppm}

    # Extract bodypart position FOR FIRST TWO BODY PARTS ONLY
    bodypart_pos = data.loc[:, ['bodypart1_x', 'bodypart1_y', 'bodypart2_x', 'bodypart2_y']]

    #Scale pos data to specific PPM
    real_ppm = ppm
    goal_ppm = 400 #HARD CODED FOR NOW
    pos_header['scaled_ppm'] = goal_ppm
    scale_fact = goal_ppm / real_ppm

    # Scale pos data in place
    bodypart_pos *= scale_fact

    raw_pos_data = {'header': pos_header,
                    'bodypart_pos': bodypart_pos}
    
    return raw_pos_data