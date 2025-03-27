import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_direction_polar_plot(spike_times, frame_times, directions, n_bins):
    """
    Create a direction polar plot for spike data. Currently no smoothing operations are performed.

    Parameters:
    - spike_times (list or array): List or array of timestamps where spikes occur.
    - frame_times (list or array): List or array of timestamps where each frame was taken.
    - directions (list or array): List or array of directions corresponding to each frame.
    - n_bins (int): Number of direction bins to use.

    Outputs:
    - A polar plot showing spike count per unit time for each direction bin.
    """
    if not isinstance(spike_times, (np.ndarray, list)):
        raise TypeError("spike_times must be a list or numpy array")
    if not isinstance(frame_times, (np.ndarray, list)):
        raise TypeError("frame_times must be a list or numpy array")
    if not isinstance(directions, (np.ndarray, list)):
        raise TypeError("directions must be a list or numpy array")
    if not isinstance(n_bins, int) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")

    result_df = make_direction_rate_map(spike_times, frame_times, directions, n_bins)
    plot_direction_rate_map(result_df)


def make_direction_rate_map(spike_times, frame_times, directions, n_bins):
    """
    Create a directional rate map DataFrame.

    Parameters:
    - spike_times (list or array): List or array of timestamps where spikes occur.
    - frame_times (list or array): List or array of timestamps where each frame was taken.
    - directions (list or array): List or array of directions corresponding to each frame.
    - n_bins (int): Number of direction bins to use.

    Returns:
    - result_df (DataFrame): A DataFrame containing the direction bins, total time, spike count,
      and spike count per unit time.
    """
    # Check input types and values
    spike_times = np.asarray(spike_times)
    frame_times = np.asarray(frame_times)
    directions = np.asarray(directions)
    if spike_times.ndim != 1 or frame_times.ndim != 1 or directions.ndim != 1:
        raise ValueError(
            "spike_times, frame_times, and directions must be 1-dimensional"
        )
    if len(frame_times) != len(directions):
        raise ValueError("frame_times and directions must have the same length")

    frame_df = pd.DataFrame({"Frame_Time": frame_times, "Direction": directions})

    # Calculate framerate from frame times
    frame_rate = np.diff(frame_times).mean()

    # Bin directions into n_bins
    direction_bins = np.linspace(0, 360, n_bins + 1)
    bin_labels = [
        f"{direction_bins[i]:.1f}-{direction_bins[i+1]:.1f}"
        for i in range(len(direction_bins) - 1)
    ]
    frame_df["Direction_Bin"] = pd.cut(
        frame_df["Direction"],
        bins=direction_bins,
        labels=bin_labels,
        include_lowest=True,
        right=False,
    )

    # Assign spike times to frame bins
    spike_frames = np.digitize(spike_times, frame_times) - 1
    spike_df = pd.DataFrame({"Spike_Frame": spike_frames})
    spike_df = spike_df[spike_df["Spike_Frame"] < len(frame_times)]

    spike_df = spike_df.merge(
        frame_df, left_on="Spike_Frame", right_index=True, how="left"
    )

    # Calculate spike count per unit time for each direction bin
    bin_counts = frame_df["Direction_Bin"].value_counts().reset_index(name="Total_Time")
    bin_spike_counts = (
        spike_df["Direction_Bin"].value_counts().reset_index(name="Spike_Count")
    )

    bin_counts.rename(columns={"index": "Direction_Bin"}, inplace=True)
    bin_spike_counts.rename(columns={"index": "Direction_Bin"}, inplace=True)

    # Merge the two DataFrames and calculate spike count per unit time
    result_df = pd.merge(
        bin_counts, bin_spike_counts, on="Direction_Bin", how="left"
    ).fillna({"Spike_Count": 0})

    result_df["Total_Time"] /= frame_rate
    result_df["Count_Per_Time"] = result_df["Spike_Count"] / result_df["Total_Time"]

    result_df["Direction_Bin"] = pd.Categorical(
        result_df["Direction_Bin"], categories=bin_labels, ordered=True
    )
    result_df = result_df.sort_values("Direction_Bin").reset_index(drop=True)

    return result_df


def plot_direction_rate_map(result_df):
    """
    Plot the directional rate map as a polar plot.

    Parameters:
    - result_df (DataFrame): DataFrame containing the direction bins, total time, spike count,
      and spike count per unit time.

    Outputs:
    - A polar plot showing spike count per unit time for each direction bin.
    """
    theta = np.linspace(0.0, 2 * np.pi, len(result_df))
    radii = result_df["Count_Per_Time"].values

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    bars = ax.bar(theta, radii, width=2 * np.pi / len(result_df), bottom=0.0)

    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)
    plt.show()
