import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def bz_csd(lfp, **kwargs):
    """
    Calculates the 1D approximation of current source density (CSD) from a linear array of LFPs.
    Translated from: https://github.com/buzsakilab/buzcode/blob/master/analysis/lfp/CurrentSourceDensity/bz_CSD.m

    Args:
        lfp (ndarray or dict): LFP data. If ndarray, shape should be (timepoints, channels). If dict, should have fields 'data', 'timestamps', 'sampling_rate'. If pandas DataFrame, should have timestamps as columns, channel IDs as rows
        **kwargs: Optional keyword arguments for customizing the CSD computation and plotting.

    Returns:
        dict: CSD data with fields 'data', 'timestamps', 'sampling_rate', 'channels', 'params'.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import detrend, savgol_filter

    # Parse inputs
    if isinstance(lfp, dict):
        data = lfp['data']
        timestamps = lfp['timestamps']
        sampling_rate = lfp['sampling_rate']

    elif isinstance(lfp, pd.DataFrame):
        data = lfp.to_numpy().T
        timestamps = lfp.columns.to_numpy()
        sampling_rate = kwargs.get('sampling_rate', 1000)

    else:
        data = lfp
        timestamps = np.arange(data.shape[0])
        sampling_rate = kwargs.get('sampling_rate', 1000)

    channels = kwargs.get('channels', np.arange(data.shape[1]))
    spat_sm = kwargs.get('spat_sm', 0)
    temp_sm = kwargs.get('temp_sm', 0)
    do_detrend = kwargs.get('do_detrend', False)
    plot_csd = kwargs.get('plot_csd', True)
    plot_lfp = kwargs.get('plot_lfp', True)
    win = kwargs.get('win', [0, data.shape[0]])

    # Compute CSD
    lfp_frag = data[win[0]:win[1], channels] * -1

    if lfp_frag.size == 0:
        # print("LFP fragment is empty. Skipping processing.")
        return None

    # Detrend
    if do_detrend:
        lfp_frag = detrend(lfp_frag, axis=0)

    # Temporal smoothing
    if temp_sm > 0 and lfp_frag.shape[0] > temp_sm:
        # Ensure window length is odd and appropriate
        temp_sm = min(temp_sm | 1, lfp_frag.shape[0] - 1)
        polyorder_temp = min(3, temp_sm - 1)
        lfp_frag = savgol_filter(lfp_frag, window_length=temp_sm, polyorder=polyorder_temp, axis=0)

    # Spatial smoothing
    if spat_sm > 0 and lfp_frag.shape[1] > spat_sm:
        # Ensure window length is odd and appropriate
        spat_sm = min(spat_sm | 1, lfp_frag.shape[1] - 1)
        polyorder_spat = min(3, spat_sm - 1)
        lfp_frag = savgol_filter(lfp_frag, window_length=spat_sm, polyorder=polyorder_spat, axis=1)


    # Calculate CSD
    CSD = np.diff(lfp_frag, n=2, axis=0)

    # Generate output dictionary
    csd = {}
    csd['data'] = CSD
    csd['timestamps'] = timestamps[win[0] + 2: win[1]]  # Remove the adjustment by 2
    csd['sampling_rate'] = sampling_rate
    csd['channels'] = channels
    csd['params'] = {'spat_sm': spat_sm, 'temp_sm': temp_sm, 'detrend': do_detrend}

    # Plot
    if plot_lfp:
        cmax = np.max(np.abs(data[:, channels]))
        plt.figure()
        plt.imshow(data[win[0]:win[1], channels].T, aspect='auto', cmap='seismic', vmin=-cmax, vmax=cmax)
        plt.colorbar(label='LFP (μV)')
        plt.xlabel('Samples')
        plt.ylabel('Channel')
        plt.title('LFP')
        plt.show()

    if plot_csd:
        cmax = np.max(np.abs(CSD))
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].contourf(csd['timestamps'], np.arange(CSD.shape[1]), CSD.T, 40, cmap='jet', vmin=-cmax, vmax=cmax)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Channel')
        axes[0].set_title('CSD')
        axes[0].invert_yaxis()
        axes[1].plot(csd['timestamps'], np.mean(CSD, axis=1))
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Average CSD')
        axes[1].set_title('Average CSD')
        plt.tight_layout()
        plt.show()

    return csd

def calculate_csd_df(arm_cycle_df):
    '''
    Takes dataframe of LFP data with theta cycles and traversal indices
    Calculates current-source density for each traversal individually and adds to the dataframe
    '''
    # List of index labels which are NOT channel data (hard-coded for now)
    non_ephys_labels = ['Traversal Index', 'Cycle Index', 'Cycle Theta Phase', 'Speed']

    # Add indices to output dataframe
    csd_index = [str(label) + '_csd' for label in arm_cycle_df.drop(non_ephys_labels).index]
    csd_df_empty = pd.DataFrame(np.nan, index=csd_index, columns=arm_cycle_df.columns)
    csd_df = pd.concat([arm_cycle_df, csd_df_empty], axis=0)

    # Get total number of traversals
    traversals = int(max(arm_cycle_df.loc(axis=0)['Traversal Index']))

    # Loop through traversals and calculate CSD
    for traversal in range(traversals + 1):
        # Select traversal data from dataframe
        traversal_df = arm_cycle_df.loc[:, arm_cycle_df.loc['Traversal Index'] == traversal]

        # Calculate CSD
        traversal_csd = bz_csd(traversal_df.drop(non_ephys_labels, axis=0),
                               plot_csd=False, plot_lfp=False)

        # If CSD calculation failed
        if traversal_csd is None:
            pass  # Optionally handle this case
        else:
            # CSD has two fewer samples than LFP, so align CSD to correct timestamps and add to frame
            csd_df.loc[csd_index, traversal_df.columns[1:-1]] = traversal_csd['data'].T

    return csd_df, csd_index

def mean_csd_theta_phase(arm_csd_df, arm_csd_labels):
    '''
    Takes the processed dataframe with CSD calculated, along with labels specifying which rows refer to these data
    Also takes optional string for displaying arm identity on plot

    Returns original dataframe meaned across theta phase bins
    '''

    ## Bin by theta phase
    bins = np.linspace(0, 2 * np.pi, 101)
    binned_theta = np.digitize(arm_csd_df.loc['Cycle Theta Phase'].values, bins)

    csd_data = arm_csd_df.loc[arm_csd_labels]
    csd_data.columns = binned_theta
    csd_data = csd_data.dropna(axis = 1)

    ## Mean CSD for each channel in that bin
    # Extract unique column names
    unique_columns = set(csd_data.columns)

    # Initialize a dictionary to hold our new columns
    mean_columns = {}

    # Calculate mean for each unique column
    for col in unique_columns:
        # Select columns with the current unique name
        selected_columns = csd_data.loc[:, col]

        # Calculate the mean for these columns
        mean_value = selected_columns.mean(axis=1)

        # Add the mean values to our dictionary
        mean_columns[col] = mean_value

    # Create a new DataFrame from the dictionary
    mean_csd_data = pd.DataFrame(mean_columns)

    return mean_csd_data

def plot_csd_theta_phase(mean_csd_data, title='', save_path=None, padding=2):
    '''
    Takes output of the previous function, and plots mean CSD across theta phase using pltcontourf
    arm
    '''
    # Plot using contourf as in bz_csd function
    # Preparing the meshgrid for plotting
    channels = np.arange(mean_csd_data.shape[0])
    theta_phases = np.arange(mean_csd_data.shape[1]) * (2 * np.pi / mean_csd_data.shape[1])
    # Cut off the first and last two theta phases to avoid edge effects
    X, Y = np.meshgrid(theta_phases[2:-2], channels)
    mean_csd_data = mean_csd_data.iloc[:, 2:-2]

    cmax = np.nanmax(np.abs(mean_csd_data))
    # Plotting using contourf
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(X, Y, mean_csd_data, 40, cmap='jet', vmin=-cmax, vmax=cmax)

    plt.colorbar(contour)
    # Adding labels and title
    plt.xticks(np.linspace(0, 2 * np.pi, 5),
           ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    channel_depths = channels * -100
    plt.yticks(channels, channel_depths)
    plt.gca().invert_yaxis()
    plt.xlabel('Theta Phase')
    plt.ylabel('Depth (μm)')
    plt.title(f'Current Source Density across Theta Phase - {title}')

    if save_path is not None:
        plt.savefig(save_path)

    # Show the plot
    plt.show()
