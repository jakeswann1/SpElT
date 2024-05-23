import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

def create_direction_polar_plot(spike_stamps, frame_times, directions, direction_bins):
    """
    Create a direction polar plot for spike data.

    Parameters:
    - spike_stamps (list or array): List or array of timestamps where spikes occur.
    - frame_times (list or array): List or array of timestamps where each frame was taken.
    - directions (list or array): List or array of directions corresponding to each frame.
    - direction_bins (list or array): List or array defining the direction bins in degrees. For example, [0, 30, 60, ..., 360].
    Outputs:
    - A polar plot showing spike count per unit time for each direction bin.
    """
    # Create a DataFrame for the frame data
    frame_df = pd.DataFrame({
        'Frame_Time': frame_times,
        'Direction': directions
    })

    # Bin the direction data
    bin_labels = [f'{direction_bins[i]}-{direction_bins[i+1]}' for i in range(len(direction_bins)-1)]
    print(bin_labels)

    frame_df['Direction_Bin'] = pd.cut(frame_df['Direction'], bins=direction_bins, labels=bin_labels, include_lowest=True, right=False)
    print(frame_df)

    # Ensure spike_stamps and frame_times are numpy arrays for vectorized operations
    spike_stamps = np.array(spike_stamps)
    frame_times = np.array(frame_times)

    # Digitize the spike times to find their corresponding frames
    spike_frames = np.digitize(spike_stamps, frame_times) - 1  # -1 to convert to 0-based index



    # Create a DataFrame for the spike data
    spike_df = pd.DataFrame({'Spike_Frame': spike_frames})
    spike_df = spike_df[spike_df['Spike_Frame'] < len(frame_times)]  # Remove spikes beyond the last frame

    # Merge spike data with frame data to get the direction bin for each spike
    spike_df = spike_df.merge(frame_df, left_on='Spike_Frame', right_index=True, how='left')

    # Calculate the total time and spike count for each bin
    bin_counts = frame_df['Direction_Bin'].value_counts().reset_index(name='Total_Time')
    bin_spike_counts = spike_df['Direction_Bin'].value_counts().reset_index(name='Spike_Count')

    # Rename columns for clarity
    bin_counts.rename(columns={'index': 'Direction_Bin'}, inplace=True)
    bin_spike_counts.rename(columns={'index': 'Direction_Bin'}, inplace=True)

    # Merge the total time and spike count data
    result_df = pd.merge(bin_counts, bin_spike_counts, on='Direction_Bin', how='left').fillna({'Spike_Count': 0})

    # Multiply total time by 0.04 (assuming 0.04 seconds per frame)
    result_df['Total_Time'] *= 0.04

    # Calculate the spike count per unit time
    result_df['Count_Per_Time'] = result_df['Spike_Count'] / result_df['Total_Time']
    result_df

     # Order by direction bins from smallest to biggest angle
    result_df['Direction_Bin'] = pd.Categorical(result_df['Direction_Bin'], categories=bin_labels, ordered=True)
    result_df = result_df.sort_values('Direction_Bin').reset_index(drop=True)
    
    

    # Create the polar plot
    angles = np.deg2rad(np.linspace(0, 360, len(result_df) + 1))
    values = result_df['Count_Per_Time'].tolist()
    values += values[:1]  # close the circle
    angles = angles[:len(values)]  # match angles length

    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, marker='o')
    ax.fill(angles, values, alpha=0.25)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)


    plt.title('Spike Count per Unit Time')
    plt.show()

import numpy as np
import pandas as pd

def create_direction_df(spike_stamps, frame_times, directions, direction_bins):
    """
    Compute the direction data and returns a DataFrame with spike count per unit time for each direction bin.
    
    Parameters:
    spike_stamps (list or array): List or array of timestamps where spikes occur.
    frame_times (list or array): List or array of timestamps corresponding to each frame.
    directions (list or array): List or array of directions corresponding to each frame.
    direction_bins (list or array): List or array defining the direction bins in degrees. For example, [0, 30, 60, ..., 360].
    
    Returns:
    pd.DataFrame: DataFrame showing spike count per unit time for each direction bin.
    """
    
   # Create a DataFrame for the frame data
    frame_df = pd.DataFrame({
        'Frame_Time': frame_times,
        'Direction': directions
    })

    # Bin the direction data
    bin_labels = [f'{direction_bins[i]}-{direction_bins[i+1]}' for i in range(len(direction_bins)-1)]
    

    frame_df['Direction_Bin'] = pd.cut(frame_df['Direction'], bins=direction_bins, labels=bin_labels, include_lowest=True, right=False)

    # Ensure spike_stamps and frame_times are numpy arrays for vectorized operations
    spike_stamps = np.array(spike_stamps)
    frame_times = np.array(frame_times)

    
    # Digitize the spike times to find their corresponding frames
    spike_frames = np.digitize(spike_stamps, frame_times) - 1  # -1 to convert to 0-based index
    
    # Create a DataFrame for the spike data
    spike_df = pd.DataFrame({'Spike Frame': spike_frames})
    spike_df = spike_df[spike_df['Spike Frame'] < len(frame_times)]  # Remove spikes beyond the last frame
    
    
    # Digitize the spike times to find their corresponding frames
    spike_frames = np.digitize(spike_stamps, frame_times) - 1  # -1 to convert to 0-based index

    # Create a DataFrame for the spike data
    spike_df = pd.DataFrame({'Spike_Frame': spike_frames})
    spike_df = spike_df[spike_df['Spike_Frame'] < len(frame_times)]  # Remove spikes beyond the last frame

    # Merge spike data with frame data to get the direction bin for each spike
    spike_df = spike_df.merge(frame_df, left_on='Spike_Frame', right_index=True, how='left')

    # Calculate the total time and spike count for each bin
    bin_counts = frame_df['Direction_Bin'].value_counts().reset_index(name='Total_Time')
    bin_spike_counts = spike_df['Direction_Bin'].value_counts().reset_index(name='Spike_Count')

    # Rename columns for clarity
    bin_counts.rename(columns={'index': 'Direction_Bin'}, inplace=True)
    bin_spike_counts.rename(columns={'index': 'Direction_Bin'}, inplace=True)

    # Merge the total time and spike count data
    result_df = pd.merge(bin_counts, bin_spike_counts, on='Direction_Bin', how='left').fillna({'Spike_Count': 0})

    # Multiply total time by 0.04 (assuming 0.04 seconds per frame)
    result_df['Total_Time'] *= 0.04

    # Calculate the spike count per unit time
    result_df['Count_Per_Time'] = result_df['Spike_Count'] / result_df['Total_Time']
    result_df

     # Order by direction bins from smallest to biggest angle
    result_df['Direction_Bin'] = pd.Categorical(result_df['Direction_Bin'], categories=bin_labels, ordered=True)
    result_df = result_df.sort_values('Direction_Bin').reset_index(drop=True)
    
    return result_df

def plot_direction_polar(result_df):
    """
    Create a direction polar plot for spike data.
    
    Parameters:
    result_df (pd.DataFrame): DataFrame showing spike count per unit time for each direction bin.
    """
    angles = np.deg2rad(np.linspace(0, 360, len(result_df) + 1))
    values = result_df['Count_Per_Time'].tolist()
    values += values[:1]  # close the circle
    angles = angles[:len(values)]  # match angles length

    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, marker='o')
    ax.fill(angles, values, alpha=0.25)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)


    plt.title('Spike Count per Unit Time')
    plt.show()
    