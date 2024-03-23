import numpy as np
import pandas as pd

def load_pos_axona(path, override_ppm = None):
    """
    Load position data from .pos file and associated .csv file created during axona_preprocessing.pos_from_bin

    Parameters
    ----------
    path : str
        Path to the .pos file to be loaded
    override_ppm : int, optional
        Override the pixels per metre value in the .pos file, by default None

    Returns
    -------
    dict
        Dictionary containing the header information and position data from the .pos file
    float
        Sampling rate of the position data
    """

    # Read position data from csv file (faster than directly from .pos)
    data = pd.read_csv(f'{path}_pos.csv', index_col = 0).T

    # Read header from pos file into dictionary
    with open(f'{path}.pos', 'rb') as fid:
        pos_header = {}

        # Read the lines of the file up to the specified number (27 in this case) and write into dict
        for _ in range(27):
            line = fid.readline()
            if not line:
                break
            elements = line.decode().strip().split()
            pos_header[elements[0]] = ' '.join(elements[1:])
            
    # Get sampling rate
    pos_sampling_rate = float(pos_header['sample_rate'][:-3])

    # Extract LED position data and tracked pixel size data
    led_pos = data.loc[:, ['X1', 'Y1', 'X2', 'Y2']]
    led_pix = data.loc[:, ['Pixels LED 1', 'Pixels LED 2']]

    # Set missing values to NaN
    led_pos[led_pos == 1023] = np.nan
    led_pix[led_pix == 1023] = np.nan

    ## Scale pos data to specific PPM 
    # Currently hard coded to 400 PPM
    real_ppm = int(pos_header['pixels_per_metre'])

    if override_ppm:
        pos_header['pixels_per_metre'] = override_ppm
        real_ppm = override_ppm
    

    goal_ppm = 400
    pos_header['scaled_ppm'] = goal_ppm
    scale_fact = goal_ppm / real_ppm

    # Scale area boundaries in place
    pos_header['min_x'] = int(pos_header['window_min_x']) * scale_fact
    pos_header['max_x'] = int(pos_header['window_max_x']) * scale_fact
    pos_header['min_y'] = int(pos_header['window_min_y']) * scale_fact
    pos_header['max_y'] = int(pos_header['window_max_y']) * scale_fact

    # Scale pos data in place
    led_pos *= scale_fact

    # Collect header and data into a dict
    raw_pos_data = {'header': pos_header,
                'led_pos': led_pos.T,
                'led_pix': led_pix.T}
    
    return raw_pos_data, pos_sampling_rate