import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _compute_csd(
    lfp_fragment: np.ndarray,
    spat_sm: int = 0,
    temp_sm: int = 0,
    do_detrend: bool = False,
) -> np.ndarray:
    """
    Core CSD computation without plotting or I/O.

    Computes the 1D current source density using second spatial derivative.
    Extracted from bz_csd() to enable reuse in event-locked analysis.

    Parameters
    ----------
    lfp_fragment : np.ndarray
        LFP data fragment, shape (n_samples, n_channels)
        Will be multiplied by -1 as per convention
    spat_sm : int, optional
        Spatial smoothing window length (Savitzky-Golay filter)
        If 0, no spatial smoothing is applied
    temp_sm : int, optional
        Temporal smoothing window length (Savitzky-Golay filter)
        If 0, no temporal smoothing is applied
    do_detrend : bool, optional
        Whether to detrend LFP along time axis before CSD computation

    Returns
    -------
    np.ndarray
        CSD values, shape (n_samples, n_channels-2)
        Note: 2 channels are lost due to second spatial derivative

    Notes
    -----
    - LFP is multiplied by -1 following convention
    - Second spatial derivative (across channels) removes 2 edge channels
    - CSD formula: CSD_n = -(LFP_{n-1} - 2*LFP_n + LFP_{n+1})
    - Smoothing uses Savitzky-Golay filter with appropriate polynomial order
    - Units: Same as input LFP (typically µV) without spatial normalization
    """
    from scipy.signal import detrend, savgol_filter

    # Multiply by -1 following convention
    lfp_proc = lfp_fragment * -1

    # Return early if empty
    if lfp_proc.size == 0:
        return np.array([])

    # Detrend along time axis
    if do_detrend:
        lfp_proc = detrend(lfp_proc, axis=0)

    # Temporal smoothing (along time axis)
    if temp_sm > 0 and lfp_proc.shape[0] > temp_sm:
        # Ensure window length is odd and appropriate
        temp_sm = min(temp_sm | 1, lfp_proc.shape[0] - 1)
        polyorder_temp = min(3, temp_sm - 1)
        lfp_proc = savgol_filter(
            lfp_proc, window_length=temp_sm, polyorder=polyorder_temp, axis=0
        )

    # Spatial smoothing (along channel axis)
    if spat_sm > 0 and lfp_proc.shape[1] > spat_sm:
        # Ensure window length is odd and appropriate
        spat_sm = min(spat_sm | 1, lfp_proc.shape[1] - 1)
        polyorder_spat = min(3, spat_sm - 1)
        lfp_proc = savgol_filter(
            lfp_proc, window_length=spat_sm, polyorder=polyorder_spat, axis=1
        )

    # Calculate CSD using second SPATIAL derivative (across channels)
    # This removes 2 edge channels (top and bottom)
    csd_values = np.diff(lfp_proc, n=2, axis=1)

    return csd_values


def bz_csd(lfp: np.ndarray | dict | pd.DataFrame, **kwargs):
    """
    Calculates the 1D approximation of current source density (CSD)
        from a linear array of LFPs.
    Translated from: https://github.com/buzsakilab/buzcode/blob/master/analysis/lfp/CurrentSourceDensity/bz_CSD.m

    Args:
        lfp: LFP data. If ndarray, shape should be (timepoints, channels).
            If dict, should have fields 'data', 'timestamps', 'sampling_rate'.
            If pandas DataFrame, should have timestamps as columns, channel IDs as rows
        **kwargs: keyword arguments for customizing the CSD computation and plotting.

    Returns:
        dict: CSD data with fields
            'data', 'timestamps', 'sampling_rate', 'channels', 'params'.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # Parse inputs
    if isinstance(lfp, dict):
        data = lfp["data"]
        timestamps = lfp["timestamps"]
        sampling_rate = lfp["sampling_rate"]

    elif isinstance(lfp, pd.DataFrame):
        data = lfp.to_numpy().T
        timestamps = lfp.columns.to_numpy()
        sampling_rate = kwargs.get("sampling_rate", 1000)

    else:
        data = lfp
        timestamps = np.arange(data.shape[0])
        sampling_rate = kwargs.get("sampling_rate", 1000)

    channels = kwargs.get("channels", np.arange(data.shape[1]))
    spat_sm = kwargs.get("spat_sm", 0)
    temp_sm = kwargs.get("temp_sm", 0)
    do_detrend = kwargs.get("do_detrend", False)
    plot_csd = kwargs.get("plot_csd", True)
    plot_lfp = kwargs.get("plot_lfp", True)
    win = kwargs.get("win", [0, data.shape[0]])

    # Compute CSD using extracted helper function
    lfp_frag = data[win[0] : win[1], channels]

    if lfp_frag.size == 0:
        # print("LFP fragment is empty. Skipping processing.")
        return None

    # Use helper function for CSD computation
    csd_values = _compute_csd(
        lfp_frag, spat_sm=spat_sm, temp_sm=temp_sm, do_detrend=do_detrend
    )

    # Generate output dictionary
    csd = {}
    csd["data"] = csd_values
    csd["timestamps"] = timestamps[
        win[0] : win[1]
    ]  # No adjustment - CSD loses channels, not timepoints
    csd["sampling_rate"] = sampling_rate
    csd["channels"] = channels[1:-1]  # CSD loses 2 edge channels
    csd["params"] = {"spat_sm": spat_sm, "temp_sm": temp_sm, "detrend": do_detrend}

    # Plot
    if plot_lfp:
        cmax = np.max(np.abs(data[:, channels]))
        plt.figure()
        plt.imshow(
            data[win[0] : win[1], channels].T,
            aspect="auto",
            cmap="seismic",
            vmin=-cmax,
            vmax=cmax,
        )
        plt.colorbar(label="LFP (μV)")
        plt.xlabel("Samples")
        plt.ylabel("Channel")
        plt.title("LFP")
        plt.show()

    if plot_csd:
        cmax = np.max(np.abs(csd_values))
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].contourf(
            csd["timestamps"],
            np.arange(csd_values.shape[1]),
            csd_values.T,
            40,
            cmap="jet",
            vmin=-cmax,
            vmax=cmax,
        )
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Channel")
        axes[0].set_title("CSD")
        axes[0].invert_yaxis()
        axes[1].plot(csd["timestamps"], np.mean(csd_values, axis=1))
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Average CSD")
        axes[1].set_title("Average CSD")
        plt.tight_layout()
        plt.show()

    return csd


def calculate_csd_df(arm_cycle_df):
    """
    Takes dataframe of LFP data with theta cycles and traversal indices
    Calculates current-source density for each traversal individually
        and adds to the dataframe
    """
    # List of index labels which are NOT channel data (hard-coded for now)
    non_ephys_labels = ["Traversal Index", "Cycle Index", "Cycle Theta Phase", "Speed"]

    # Add indices to output dataframe
    csd_index = [
        str(label) + "_csd" for label in arm_cycle_df.drop(non_ephys_labels).index
    ]
    csd_df_empty = pd.DataFrame(np.nan, index=csd_index, columns=arm_cycle_df.columns)
    csd_df = pd.concat([arm_cycle_df, csd_df_empty], axis=0)

    # Get total number of traversals
    traversals = int(max(arm_cycle_df.loc(axis=0)["Traversal Index"]))

    # Loop through traversals and calculate CSD
    for traversal in range(traversals + 1):
        # Select traversal data from dataframe
        traversal_df = arm_cycle_df.loc[
            :, arm_cycle_df.loc["Traversal Index"] == traversal
        ]

        # Calculate CSD
        traversal_csd = bz_csd(
            traversal_df.drop(non_ephys_labels, axis=0), plot_csd=False, plot_lfp=False
        )

        # If CSD calculation failed
        if traversal_csd is None:
            pass  # Optionally handle this case
        else:
            # CSD has two fewer samples than LFP,
            # so align CSD to correct timestamps and add to frame
            try:
                csd_df.loc[csd_index, traversal_df.columns[1:-1]] = traversal_csd[
                    "data"
                ].T
            except ValueError as e:
                print("CSD calculation failed for traversal", traversal)
                print("Error:", e)
                continue

    return csd_df, csd_index


def plot_mean_csd(
    data: pd.DataFrame | pd.Series | dict[str, pd.DataFrame | pd.Series],
    channel_depths: list[float],
    title: str,
    save_path: str,
    plot_sem: bool = False,
) -> None:
    """
    Plot CSD data - handles both single dataset and comparison between datasets.

    Parameters:
    -----------
    data : pd.DataFrame, pd.Series, or dict[str, pd.DataFrame | pd.Series]
        Single DataFrame/Series for mean CSD plot, or dict with dataset names as keys
        for comparison plot (e.g., {"Centre": central_data, "Return": return_data})
    channel_depths : list[float]
        Channel depth values for y-tick labels
    title : str
        Plot title
    save_path : str
        Path to save the figure
    plot_sem : bool, optional
        Whether to plot standard error of the mean as shaded area
    """
    plt.figure()

    if isinstance(data, pd.DataFrame):
        # Single dataset plot
        mean_values = data.mean(axis=1)
        plt.plot(mean_values.values, mean_values.index, c="orange")

        if plot_sem:
            sem_values = data.sem(axis=1)
            plt.fill_betweenx(
                mean_values.index,
                mean_values.values - sem_values.values,
                mean_values.values + sem_values.values,
                alpha=0.2,
                color="orange",
            )

    else:
        # Comparison plot
        colors = ["orange", "blue", "green", "red", "purple"]

        for i, (label, df) in enumerate(data.items()):
            color = colors[i % len(colors)]

            # Handle both Series and DataFrame inputs
            if isinstance(df, pd.Series):
                mean_values = df
                has_multiple_columns = False
            else:
                mean_values = df.mean(axis=1) if df.shape[1] > 1 else df.iloc[:, 0]
                has_multiple_columns = df.shape[1] > 1

            plt.plot(mean_values.values, mean_values.index, label=label, c=color)

            if plot_sem and has_multiple_columns:
                sem_values = df.sem(axis=1)
                plt.fill_betweenx(
                    mean_values.index,
                    mean_values.values - sem_values.values,
                    mean_values.values + sem_values.values,
                    alpha=0.2,
                    color=color,
                )

        plt.legend()

    # Common formatting
    plt.gca().invert_yaxis()
    plt.yticks(
        list(data.values())[0].index if isinstance(data, dict) else data.index,
        channel_depths,
    )
    plt.ylabel("Channel Depth (μm)")
    plt.xlabel("CSD")
    plt.axvline(x=0, color="grey", linestyle="--", dashes=[5, 5])
    plt.title(title)
    plt.savefig(save_path)
    plt.show()
