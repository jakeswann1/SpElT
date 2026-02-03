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
        CSD values, shape (n_samples-2, n_channels)
        Note: 2 time samples are lost due to second derivative operation

    Notes
    -----
    - LFP is multiplied by -1 following convention
    - Second derivative along time axis removes 2 time samples
    - Smoothing uses Savitzky-Golay filter with appropriate polynomial order
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

    # Calculate CSD using second derivative
    # This removes 2 time samples
    csd_values = np.diff(lfp_proc, n=2, axis=0)

    return csd_values


def compute_event_locked_csd(
    lfp_data: np.ndarray,
    timestamps: np.ndarray,
    event_times: np.ndarray,
    time_window: tuple[float, float],
    sampling_rate: float,
    channels: np.ndarray | list | None = None,
    spat_sm: int = 0,
    temp_sm: int = 0,
    do_detrend: bool = False,
) -> dict:
    """
    Compute event-locked CSD around specified event times.

    Implementation follows standard methodology: LFP signals are first averaged
    across events (event-triggered average), then CSD is computed on the averaged
    LFP signal. This approach reduces noise before differentiation and matches
    published methods.

    Algorithm:
    1. Extract LFP segments aligned to event times
    2. Average LFP across all events → event-triggered average
    3. Compute CSD on the averaged LFP using second spatial derivative

    Parameters
    ----------
    lfp_data : np.ndarray
        LFP data, shape (n_samples, n_channels)
    timestamps : np.ndarray
        LFP timestamps in seconds, shape (n_samples,)
    event_times : np.ndarray
        Event timestamps in seconds (e.g., ripple peak times), shape (n_events,)
    time_window : tuple[float, float]
        Time window around events as (pre_time, post_time) in seconds
        Example: (-0.1, 0.1) for ±100ms window
    sampling_rate : float
        Sampling frequency in Hz
    channels : np.ndarray | list | None
        Channel indices to use. If None, use all channels.
    spat_sm : int
        Spatial smoothing window (Savitzky-Golay)
    temp_sm : int
        Temporal smoothing window (Savitzky-Golay)
    do_detrend : bool
        Whether to detrend LFP before CSD computation

    Returns
    -------
    dict
        'csd': np.ndarray, CSD values, shape (n_timepoints, n_channels-2)
        'lfp_mean': np.ndarray, mean LFP across events, shape (n_timepoints, n_channels)
        'timestamps_relative': np.ndarray, time relative to event (0 = event time)
        'n_events': int, number of valid events included
        'n_events_excluded': int, number of events excluded (out of bounds)
        'sampling_rate': float
        'channels': np.ndarray, channel indices used
        'params': dict, parameters used for computation

    Notes
    -----
    - Events too close to recording boundaries are excluded
    - All event windows are aligned to relative time (t=0 at event)
    - NaN handling: events with NaN values are excluded from averaging
    - CSD formula: CSD_n = -(LFP_{n-1} - 2*LFP_n + LFP_{n+1})

    Examples
    --------
    >>> # Compute ripple-triggered CSD
    >>> results = compute_event_locked_csd(
    ...     lfp_data=lfp_trial_data,
    ...     timestamps=lfp_timestamps,
    ...     event_times=ripple_results['ripple_timestamps'],
    ...     time_window=(-0.1, 0.1),  # ±100ms
    ...     sampling_rate=1000,
    ...     spat_sm=3,
    ...     temp_sm=5
    ... )
    >>> print(f"Averaged {results['n_events']} ripple events")
    >>> plt.imshow(results['csd'].T, aspect='auto')
    """
    # Handle channel selection
    if channels is None:
        channels = np.arange(lfp_data.shape[1])
    else:
        channels = np.asarray(channels)

    # Select requested channels
    lfp_data = lfp_data[:, channels]

    # Calculate time window in samples
    pre_samples = int(time_window[0] * sampling_rate)  # Negative value
    post_samples = int(time_window[1] * sampling_rate)  # Positive value
    window_length_samples = post_samples - pre_samples

    # Create relative time axis
    timestamps_relative = np.linspace(
        time_window[0], time_window[1], window_length_samples
    )

    # Initialize list to store individual LFP windows
    lfp_windows = []
    n_events_excluded = 0

    # Extract LFP windows for all events
    for event_time in event_times:
        # Find nearest timestamp index for this event
        event_idx = np.argmin(np.abs(timestamps - event_time))

        # Calculate window indices
        start_idx = event_idx + pre_samples  # pre_samples is negative
        end_idx = event_idx + post_samples

        # Check bounds
        if start_idx < 0 or end_idx >= len(timestamps):
            n_events_excluded += 1
            continue

        # Extract LFP segment
        lfp_segment = lfp_data[start_idx:end_idx, :]

        # Check for NaN values
        if np.any(np.isnan(lfp_segment)):
            n_events_excluded += 1
            continue

        # Check segment has correct length
        if lfp_segment.shape[0] != window_length_samples:
            n_events_excluded += 1
            continue

        lfp_windows.append(lfp_segment)

    # Check if we have any valid events
    if len(lfp_windows) == 0:
        raise ValueError(
            "No valid events found. All events were excluded "
            "(boundary issues or NaN values)."
        )

    # Convert list to array: shape (n_events, n_timepoints, n_channels)
    lfp_windows_array = np.array(lfp_windows)

    # Average LFP across all events (event-triggered average)
    lfp_mean = np.nanmean(lfp_windows_array, axis=0)

    # Compute CSD on the averaged LFP
    csd = _compute_csd(
        lfp_mean, spat_sm=spat_sm, temp_sm=temp_sm, do_detrend=do_detrend
    )

    # Adjust timestamps_relative to account for samples lost in CSD computation
    # CSD loses 2 samples due to second derivative
    timestamps_relative_adjusted = timestamps_relative[1:-1]

    # Build results dictionary
    results = {
        "csd": csd,
        "lfp_mean": lfp_mean,
        "timestamps_relative": timestamps_relative_adjusted,
        "n_events": len(lfp_windows),
        "n_events_excluded": n_events_excluded,
        "sampling_rate": sampling_rate,
        "channels": channels,
        "params": {
            "spat_sm": spat_sm,
            "temp_sm": temp_sm,
            "do_detrend": do_detrend,
            "time_window": time_window,
        },
    }

    return results


def compute_phase_binned_csd(
    csd_data: np.ndarray | pd.DataFrame,
    phase_values: np.ndarray,
    n_bins: int = 100,
    phase_range: tuple[float, float] = (0, 2 * np.pi),
) -> pd.DataFrame:
    """
    Bin CSD by phase and compute mean within each bin.

    Generic version of mean_csd_theta_phase() that works with any phase
    (theta, gamma, ripple phase, etc.)

    Parameters
    ----------
    csd_data : np.ndarray or pd.DataFrame
        CSD data. If ndarray, shape (n_channels, n_samples).
        If DataFrame, rows are channels, columns are samples.
    phase_values : np.ndarray
        Phase values for each sample, shape (n_samples,)
        Should be in radians within phase_range
    n_bins : int, optional
        Number of phase bins. Default: 100
    phase_range : tuple[float, float], optional
        Phase range in radians, default (0, 2π) for full cycle

    Returns
    -------
    pd.DataFrame
        Mean CSD for each phase bin
        Rows: channels, Columns: phase bins

    Examples
    --------
    >>> # Theta phase binning (backwards compatible)
    >>> binned = compute_phase_binned_csd(
    ...     csd_data=arm_csd_df.loc[arm_csd_labels],
    ...     phase_values=arm_csd_df.loc["Cycle Theta Phase"].values,
    ...     n_bins=100
    ... )

    >>> # Gamma phase binning (new capability)
    >>> binned = compute_phase_binned_csd(
    ...     csd_data=csd_array,
    ...     phase_values=gamma_phase,
    ...     n_bins=50
    ... )
    """
    # Convert to DataFrame if needed
    if isinstance(csd_data, np.ndarray):
        csd_df = pd.DataFrame(csd_data)
    else:
        csd_df = csd_data.copy()

    # Create phase bins
    bins = np.linspace(phase_range[0], phase_range[1], n_bins + 1)

    # Digitize phase values
    binned_phase = np.digitize(phase_values, bins)

    # Assign binned phase as columns
    csd_df.columns = binned_phase

    # Drop NaN columns
    csd_df = csd_df.dropna(axis=1)

    # Get unique bin indices
    unique_bins = set(csd_df.columns)

    # Initialize dictionary to hold mean values for each bin
    mean_columns = {}

    # Calculate mean for each unique bin
    for bin_idx in unique_bins:
        # Select columns with the current bin index
        selected_columns = csd_df.loc[:, bin_idx]

        # Calculate the mean for these columns
        if isinstance(selected_columns, pd.Series):
            # Only one sample in this bin
            mean_value = selected_columns
        else:
            # Multiple samples in this bin
            mean_value = selected_columns.mean(axis=1)

        # Add the mean values to our dictionary
        mean_columns[bin_idx] = mean_value

    # Create a new DataFrame from the dictionary
    mean_csd_data = pd.DataFrame(mean_columns)

    # Sort columns by bin index
    mean_csd_data = mean_csd_data.sort_index(axis=1)

    return mean_csd_data


def plot_event_locked_csd(
    csd_result: dict,
    channel_depths: np.ndarray | None = None,
    title: str = "Event-Locked CSD",
    cmap: str = "RdBu_r",
    plot_mean: bool = True,
    save_path: str | None = None,
) -> None:
    """
    Plot event-locked CSD from compute_event_locked_csd() output.

    Creates 2-panel figure:
    - Left: CSD heatmap (time × depth)
    - Right: Mean CSD profile across time (optional)

    Parameters
    ----------
    csd_result : dict
        Output from compute_event_locked_csd()
    channel_depths : np.ndarray | None, optional
        Channel depths in μm for y-axis labels
        If None, use channel indices
    title : str, optional
        Plot title. Default: "Event-Locked CSD"
    cmap : str, optional
        Colormap for heatmap. Default: "RdBu_r"
    plot_mean : bool, optional
        If True, plot mean CSD profile on right panel. Default: True
    save_path : str | None, optional
        If provided, save figure to this path

    Examples
    --------
    >>> results = compute_event_locked_csd(...)
    >>> plot_event_locked_csd(
    ...     results,
    ...     channel_depths=channel_depths_array,
    ...     title="Ripple-Triggered CSD",
    ...     save_path="ripple_csd.png"
    ... )
    """
    # Extract data from results
    csd = csd_result["csd"]
    timestamps_relative = csd_result["timestamps_relative"]
    n_events = csd_result["n_events"]

    # Set up channel labels
    n_channels = csd.shape[1]
    if channel_depths is None:
        channel_labels = np.arange(n_channels)
        ylabel = "Channel"
    else:
        # CSD loses 2 channels, so adjust depth array
        if len(channel_depths) > n_channels:
            channel_labels = channel_depths[1:-1]
        else:
            channel_labels = channel_depths
        ylabel = "Depth (μm)"

    # Determine colormap limits (symmetric around zero)
    cmax = np.nanmax(np.abs(csd))

    # Create figure
    if plot_mean:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]

    # Left panel: CSD heatmap
    im = axes[0].contourf(
        timestamps_relative,
        np.arange(n_channels),
        csd.T,
        40,
        cmap=cmap,
        vmin=-cmax,
        vmax=cmax,
    )
    axes[0].axvline(x=0, color="k", linestyle="--", alpha=0.5, linewidth=2)

    # Add horizontal line at depth=0 if channel depths provided
    if channel_depths is not None and 0 in channel_labels:
        # Find channel index corresponding to depth=0
        depth_zero_idx = np.argmin(np.abs(channel_labels))
        axes[0].axhline(
            y=depth_zero_idx,
            color="r",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label="CA1 Pyr Layer",
        )
        axes[0].legend(loc="upper right")

    axes[0].set_xlabel("Time relative to event (s)")
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(f"{title}\n(n={n_events} events)")
    axes[0].set_yticks(np.arange(n_channels))
    axes[0].set_yticklabels([f"{int(d)}" for d in channel_labels])
    plt.colorbar(im, ax=axes[0], label="CSD")

    # Right panel: Mean CSD profile (across time)
    if plot_mean:
        mean_across_time = np.nanmean(csd, axis=0)
        axes[1].plot(mean_across_time, np.arange(n_channels), linewidth=2)
        axes[1].axvline(x=0, color="grey", linestyle="--", alpha=0.5)

        # Add horizontal line at depth=0 if channel depths provided
        if channel_depths is not None and 0 in channel_labels:
            depth_zero_idx = np.argmin(np.abs(channel_labels))
            axes[1].axhline(
                y=depth_zero_idx, color="r", linestyle="--", alpha=0.7, linewidth=2
            )

        axes[1].set_xlabel("Mean CSD")
        axes[1].set_ylabel(ylabel)
        axes[1].set_title("Time-Averaged CSD")
        axes[1].set_yticks(np.arange(n_channels))
        axes[1].set_yticklabels([f"{int(d)}" for d in channel_labels])
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


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
    csd["timestamps"] = timestamps[win[0] + 2 : win[1]]  # Remove the adjustment by 2
    csd["sampling_rate"] = sampling_rate
    csd["channels"] = channels
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


# =============================================================================
# Backwards Compatibility Wrappers
# These functions maintain the original API for existing code
# =============================================================================


def mean_csd_theta_phase(arm_csd_df, arm_csd_labels):
    """
    Takes the processed dataframe with CSD calculated,
    bins by theta phase, and returns mean CSD per bin.

    DEPRECATED: Use compute_phase_binned_csd() for new code.
    This wrapper maintains backwards compatibility.

    Parameters
    ----------
    arm_csd_df : pd.DataFrame
        DataFrame with CSD data and "Cycle Theta Phase" row
    arm_csd_labels : list or array
        Row labels that contain CSD data

    Returns
    -------
    pd.DataFrame
        Mean CSD for each theta phase bin
    """
    # Extract theta phase
    theta_phase = arm_csd_df.loc["Cycle Theta Phase"].values

    # Extract CSD data
    csd_data = arm_csd_df.loc[arm_csd_labels]

    # Call generic function with theta-specific parameters
    return compute_phase_binned_csd(
        csd_data=csd_data,
        phase_values=theta_phase,
        n_bins=100,
        phase_range=(0, 2 * np.pi),
    )


def plot_csd_theta_phase(mean_csd_data, title="", save_path=None, padding=2):
    """
    Takes output of the previous function,
    and plots mean CSD across theta phase using pltcontourf
    """
    # Plot using contourf as in bz_csd function
    # Preparing the meshgrid for plotting
    channels = np.arange(mean_csd_data.shape[0])
    theta_phases = np.arange(mean_csd_data.shape[1]) * (
        2 * np.pi / mean_csd_data.shape[1]
    )
    # Cut off the first and last two theta phases to avoid edge effects
    x, y = np.meshgrid(theta_phases[2:-2], channels)
    mean_csd_data = mean_csd_data.iloc[:, 2:-2]

    cmax = np.nanmax(np.abs(mean_csd_data))
    # Plotting using contourf
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(x, y, mean_csd_data, 40, cmap="RdBu", vmin=-cmax, vmax=cmax)

    plt.colorbar(contour)
    # Adding labels and title
    plt.xticks(
        np.linspace(0, 2 * np.pi, 5),
        ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"],
    )
    channel_depths = channels * -100
    plt.yticks(channels, channel_depths)
    plt.gca().invert_yaxis()
    plt.xlabel("Theta Phase")
    plt.ylabel("Depth (μm)")
    plt.title(f"Current Source Density across Theta Phase - {title}")

    if save_path is not None:
        plt.savefig(save_path)

    # Show the plot
    plt.show()


def process_arm_csd(
    arm_type,
    arm_times,
    cycle_numbers,
    theta_phase,
    lfp_timestamps,
    lfp_data,
    lfp_sampling_rate,
    resampled_speed_data,
    channels_to_load,
    trial,
    freq_band_name,
    csd_path,
):
    from spelt.analysis.behavioral import (
        drop_extreme_cycles,
        get_data_for_traversals,
        get_traversal_cycles,
    )

    # Calculate individual arm traversals and get identity of whole theta cycles
    arm_cycles = get_traversal_cycles(
        arm_times, cycle_numbers, lfp_timestamps, lfp_sampling_rate
    )
    print(len(arm_cycles), "total", arm_type, "arm traversals found")

    # Get lfp trace, theta phase, timestamps, speed, cycle index and traversal index
    arm_cycle_df = get_data_for_traversals(
        arm_cycles,
        cycle_numbers,
        lfp_data,
        resampled_speed_data,
        channels_to_load,
        theta_phase,
        lfp_timestamps,
    )

    # # Drop cycles where any lfp data >= |32000| (clip value for axona)
    # def filter_func(x):
    #     # If any value in the group is greater than 32000, exclude the whole group
    #     if (x.drop("Cycle Index", axis=1).abs() >= 32000).any(axis=None):
    #         return False
    #     else:
    #         return True

    arm_cycle_df = arm_cycle_df.T.groupby("Cycle Index")  # .filter(filter_func)
    # print(arm_cycle_df[arm_cycle_df["Traversal Index"] == 37].shape)
    print(arm_cycle_df)

    # Calculate CSD
    print(
        f"""Calculating {arm_type} CSD using
        {len(np.unique(arm_cycle_df["Cycle Index"]))} theta cycles"""
    )
    arm_csd_df, arm_csd_labels = calculate_csd_df(arm_cycle_df.T)

    # Drop the first and last theta cycle for each traversal
    # to remove calculation artifacts
    arm_csd_df = drop_extreme_cycles(arm_csd_df)

    # Filter dataframe for speed > 2.5 cm/s
    arm_csd_df = arm_csd_df.loc[:, arm_csd_df.loc["Speed"] > 2.5]

    # Compute mean CSD vs theta phase
    arm_csd_theta_phase = mean_csd_theta_phase(arm_csd_df, arm_csd_labels)

    # Plot
    plot_csd_theta_phase(
        arm_csd_theta_phase,
        title=f"{arm_type} Arm - {freq_band_name}",
        save_path=f"{csd_path}/{trial}_CSD_Theta_Phase_{arm_type}_{freq_band_name}.png",
    )

    # Calculate mean csd for each channel
    mean_arm_csd = arm_csd_df.mean(axis=1)[arm_csd_labels]
    mean_arm_csd.columns = [trial]

    return arm_cycle_df, arm_csd_df, arm_csd_labels, arm_csd_theta_phase, mean_arm_csd


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
