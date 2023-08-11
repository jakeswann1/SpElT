def bz_csd(lfp, **kwargs):
    """
    Calculates the 1D approximation of current source density (CSD) from a linear array of LFPs.
    Translated from: https://github.com/buzsakilab/buzcode/blob/master/analysis/lfp/CurrentSourceDensity/bz_CSD.m

    Args:
        lfp (ndarray or dict): LFP data. If ndarray, shape should be (timepoints, channels). If dict, should have fields 'data', 'timestamps', 'samplingRate'.
        **kwargs: Optional keyword arguments for customizing the CSD computation and plotting.

    Returns:
        dict: CSD data with fields 'data', 'timestamps', 'samplingRate', 'channels', 'params'.
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import detrend, savgol_filter
    
    # Parse inputs
    if isinstance(lfp, dict):
        data = lfp['data']
        timestamps = lfp['timestamps']
        samplingRate = lfp['samplingRate']
    else:
        data = lfp
        timestamps = np.arange(data.shape[0])
        samplingRate = kwargs.get('samplingRate', 1250)

    channels = kwargs.get('channels', np.arange(data.shape[1]))
    spat_sm = kwargs.get('spat_sm', 11)
    temp_sm = kwargs.get('temp_sm', 7)
    doDetrend = kwargs.get('doDetrend', False)
    plotCSD = kwargs.get('plotCSD', True)
    plotLFP = kwargs.get('plotLFP', True)
    win = kwargs.get('win', [0, data.shape[0]])

    # Compute CSD
    lfp_frag = data[win[0]:win[1], channels] * -1

    # Detrend
    if doDetrend:
        lfp_frag = detrend(lfp_frag, axis=0)

    # Temporal smoothing
    if temp_sm > 0:
        lfp_frag = savgol_filter(lfp_frag, window_length=temp_sm, polyorder=3, axis=0)

    # Spatial smoothing
    if spat_sm > 0:
        lfp_frag = savgol_filter(lfp_frag, window_length=spat_sm, polyorder=3, axis=1)

    # Calculate CSD
    CSD = np.diff(lfp_frag, n=2, axis=0)

    # Generate output dictionary
    csd = {}
    csd['data'] = CSD
    csd['timestamps'] = timestamps[win[0] + 2: win[1]]  # Remove the adjustment by 2
    csd['samplingRate'] = samplingRate
    csd['channels'] = channels
    csd['params'] = {'spat_sm': spat_sm, 'temp_sm': temp_sm, 'detrend': doDetrend}

    # Plot
    if plotLFP:
        cmax = np.max(np.abs(data[:, channels]))
        plt.figure()
        plt.imshow(data[win[0]:win[1], channels].T, aspect='auto', cmap='seismic', vmin=-cmax, vmax=cmax)
        plt.colorbar(label='LFP (μV)')
        plt.xlabel('Samples')
        plt.ylabel('Channel')
        plt.title('LFP')
        plt.show()

    if plotCSD:
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