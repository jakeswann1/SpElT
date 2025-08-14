import struct
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from scipy.interpolate import interp1d


def write_csv_from_pos(file_path):
    file_path = str(file_path)
    framecounter = []
    x1s = []
    y1s = []
    x2s = []
    y2s = []
    numpix1s = []
    numpix2s = []
    totalpixs = []
    empty = []

    with open(file_path, "rb") as file:
        content = file.read()

    start = content.find(b"data_start")
    end = content.find(b"data_end")

    if start != -1 and end != -1 and start < end:
        data = content[start + len("data_start") : end]
        chunks = [data[i : i + 20] for i in range(0, len(data), 20)]

    for pos_sample in chunks:
        framecounter.append(int.from_bytes(bytes(pos_sample[0:4]), "big"))
        x1s.append(int.from_bytes(bytes(pos_sample[4:6]), "big"))
        y1s.append(int.from_bytes(bytes(pos_sample[6:8]), "big"))
        x2s.append(int.from_bytes(bytes(pos_sample[8:10]), "big"))
        y2s.append(int.from_bytes(bytes(pos_sample[10:12]), "big"))
        numpix1s.append(int.from_bytes(bytes(pos_sample[12:14]), "big"))
        numpix2s.append(int.from_bytes(bytes(pos_sample[14:16]), "big"))
        totalpixs.append(int.from_bytes(bytes(pos_sample[16:18]), "big"))

    pos_df = pd.DataFrame(
        [framecounter, x1s, x2s, y1s, y2s, numpix1s, numpix2s, totalpixs],
        index=[
            "Framecounter",
            "X1",
            "X2",
            "Y1",
            "Y2",
            "Pixels LED 1",
            "Pixels LED 2",
            "Total Pixels",
        ],
    )
    pos_df = pos_df.set_axis(range(pos_df.shape[1]), axis=1)
    pos_df.to_csv(f"{file_path[:-4]}_pos.csv")


def led_swap_filter(led_pos, led_pix, thresh=5):
    """
    This function checks for instances of two LEDs swapping or the big one
    replacing the little one when the big one gets obscured.

    Parameters:
    led_pos: numpy array of shape (4, n_pos) containing the x, y position of each LED, in the order [X1, Y1, X2, Y2]
    led_pix: numpy array of shape (2, n_pos) containing the number of pixels in each LED
    thresh: threshold for determining if a swap has occurred

    Returns:
    swap_list: list of indices where a swap has occurred
    """

    # Calculate mean and standard deviation of led_pix, ignoring NaNs
    mean_npix = np.nanmean(led_pix, axis=1)
    std_npix = np.nanstd(led_pix, axis=1)

    pos = np.arange(1, led_pix.shape[1])

    # Calculate Euclidean distances
    dist12 = np.sqrt(
        (led_pos[0, pos] - led_pos[2, pos - 1]) ** 2
        + (led_pos[1, pos] - led_pos[3, pos - 1]) ** 2
    )
    dist11 = np.sqrt(
        (led_pos[0, pos] - led_pos[0, pos - 1]) ** 2
        + (led_pos[1, pos] - led_pos[1, pos - 1]) ** 2
    )
    dist21 = np.sqrt(
        (led_pos[2, pos] - led_pos[0, pos - 1]) ** 2
        + (led_pos[3, pos] - led_pos[1, pos - 1]) ** 2
    )
    dist22 = np.sqrt(
        (led_pos[2, pos] - led_pos[2, pos - 1]) ** 2
        + (led_pos[3, pos] - led_pos[3, pos - 1]) ** 2
    )

    # Determine if a swap has occurred
    switched = (dist12 < dist11 - thresh) & (
        np.isnan(led_pos[2, pos]) | (dist21 < dist22 - thresh)
    )

    # Check if size of big light has shrunk to be closer to that of small light (as Z score)
    z11 = (mean_npix[0] - led_pix[0, pos]) / std_npix[0]
    z12 = (led_pix[0, pos] - mean_npix[1]) / std_npix[1]
    shrunk = z11 > z12

    # Find indices where a swap has occurred
    swap_list = np.where(switched & shrunk)[0] + 1

    # Return to DataFrame format
    led_pos = pd.DataFrame(led_pos)
    led_pix = pd.DataFrame(led_pix)

    # Apply swaps to led_pos
    led_pos_swap = led_pos.copy()
    led_pos_swap.iloc[:, swap_list] = led_pos.iloc[[0, 1, 2, 3], swap_list]

    # Apply swaps to led_pix
    led_pix_swap = led_pix.copy()
    led_pix_swap.iloc[:, swap_list] = led_pix.iloc[[0, 1], swap_list]

    # print(f'{len(swap_list)} LED swaps detected and fixed')

    return led_pos_swap, led_pix_swap


def led_speed_filter(led_pos, max_pix_per_sample):
    """
    This function filters out short runs of data caused by the tracker picking up an incorrect distant point.
    It resets the led_pos to NaN for those positions.

    Parameters:
    led_pos: pandas DataFrame of shape (4, n_pos) containing the x, y position of each LED, in the order [X1, Y1, X2, Y2]
    max_pix_per_sample: maximum allowed speed of the LED in pixels per sample

    Returns:
    n_jumpy: number of positions that were filtered out
    led_pos: filtered led_pos DataFrame
    """
    n_jumpy = 0
    mpps_sqd = max_pix_per_sample**2

    # Loop over the two sets of coordinates (X1, Y1) and (X2, Y2)
    for i in range(0, led_pos.shape[0], 2):
        # Create a mask for valid (non-NaN) positions
        valid_mask = ~(led_pos.iloc[i, :].isna() | led_pos.iloc[i + 1, :].isna())
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) < 2:
            print(f"Warning: < 2 tracked points for LED {i // 2 + 1}")
            continue

        # Calculate differences between successive valid positions
        diff_indices = np.diff(valid_indices)
        diff_x = np.diff(led_pos.iloc[i, valid_indices].values)
        diff_y = np.diff(led_pos.iloc[i + 1, valid_indices].values)

        # Compute squared speed in pixels per sample
        pix_per_sample_sqd = (diff_x**2 + diff_y**2) / (diff_indices**2)

        # Find positions that exceed the speed threshold
        jumpy_positions_mask = pix_per_sample_sqd > mpps_sqd
        jumpy_positions = valid_indices[1:][jumpy_positions_mask]

        # Update the number of jumpy positions
        n_jumpy += len(jumpy_positions)

        # Set jumpy positions to NaN
        led_pos.iloc[i : i + 2, jumpy_positions] = np.nan

    return n_jumpy, led_pos


def interpolate_nan_values(df):
    """
    This function identifies the time points when any LEDs have a NaN value, sets all LEDs at that time point to NaN,
    and then interpolates the position for all LEDs.

    Parameters:
    df: pandas DataFrame of shape (4, n_pos) containing the x, y position of each LED, in the order [X1, Y1, X2, Y2]

    Returns:
    df_interpolated: DataFrame with interpolated values for NaNs
    """

    # Identify time points where any values are NaN
    partial_nan_cols = df.isna().any()

    # Set all values in these columns to NaN
    df.loc[:, partial_nan_cols] = np.nan

    # Interpolate NaN values
    df_interpolated = df.interpolate(axis=1)

    return df_interpolated


def boxcar_smooth(df, window_size):
    """
    This function applies a boxcar (moving average) smoothing to the DataFrame using scipy's boxcar function.
    It also interpolates between existing values to handle NaN values and fills remaining NaNs.

    Parameters:
    df: pandas DataFrame of shape (4, n_pos) containing the x, y position of each LED, in the order [X1, Y1, X2, Y2]
    window_size: size of the smoothing window

    Returns:
    df_smoothed: DataFrame with smoothed values
    """
    from scipy.signal import convolve
    from scipy.signal.windows import boxcar

    # Create a boxcar window
    window = boxcar(window_size)

    # Apply boxcar smoothing
    df_smoothed = df.apply(
        lambda x: pd.Series(convolve(x, window, mode="same") / window_size), axis=1
    )

    # Interpolate between existing values to handle NaN values
    df_smoothed = df_smoothed.interpolate(method="linear", axis=0)

    # Fill remaining NaNs with the first and last valid observation
    df_smoothed = df_smoothed.bfill(axis=0).ffill(axis=0)

    return df_smoothed


def calculate_position(led_pos, proportion):
    """
    This function calculates a single position some proportion of the way between the two points X1, Y1 and X2, Y2.

    Parameters:
    df_increasing: pandas DataFrame of shape (4, n_pos) containing the x, y position of each point, in the order [X1, Y1, X2, Y2]
    proportion: proportion of the way between the two points for which to calculate the position

    Returns:
    pos: calculated position
    """
    pos = (
        led_pos.iloc[:2, :].values * (1 - proportion)
        + led_pos.iloc[2:, :].values * proportion
    )
    return pd.DataFrame(pos, index=["X", "Y"])


def calculate_heading_direction(pos):
    """
    This function calculates the heading direction from position displacement.

    Parameters:
    pos: pandas DataFrame of shape (2, n_pos) containing the x, y position

    Returns:
    dir_disp: heading direction from position displacement
    """
    dir_disp = np.mod(
        np.arctan2(
            -pos.iloc[1, 1:].values + pos.iloc[1, :-1].values,
            pos.iloc[0, 1:].values - pos.iloc[0, :-1].values,
        )
        * 180
        / np.pi,
        360,
    )
    dir_disp = np.append(dir_disp, dir_disp[-1])  # Duplicate last value

    return dir_disp


def calculate_speed(pos, pos_sample_rate, pix_per_metre):
    """
    This function calculates the speed of the movement based on the position displacement.

    Parameters:
    pos: pandas DataFrame of shape (2, n_pos) containing the x, y position
    pos_sample_rate: sample rate of the position data
    pix_per_metre: conversion factor from pixels to metres

    Returns:
    speed: speed of the movement in cm/s
    """
    speed = np.sqrt(
        (pos.iloc[0, 1:].values - pos.iloc[0, :-1].values) ** 2
        + (pos.iloc[1, 1:].values - pos.iloc[1, :-1].values) ** 2
    ) * (100 * pos_sample_rate / pix_per_metre)
    speed = np.append(speed, speed[-1])  # Duplicate last value

    return speed


# Function to process position data
def postprocess_pos_data(posdata, max_speed, smoothing_window_size):
    # Check for LED swaps and apply LED swap filter
    # Check if 'led_pix' is in the posdata
    if "led_pix" in posdata:
        led_pos, led_pix = led_swap_filter(
            posdata["led_pos"].to_numpy(), posdata["led_pix"].to_numpy()
        )
    else:
        led_pos, led_pix = led_swap_filter(
            posdata["led_pos"].to_numpy(), np.zeros((2, posdata["led_pos"].shape[1]))
        )

    # Filter points for those moving impossibly fast and set led_pos_x to NaN
    ppm = int(posdata["header"]["scaled_ppm"])
    pos_sample_rate = float(posdata["header"]["sample_rate"][:-3])

    max_pix_per_sample = max_speed * ppm / pos_sample_rate
    n_jumpy, led_pos = led_speed_filter(led_pos, max_pix_per_sample * 100)

    # Interpolate NaN values
    led_pos = interpolate_nan_values(led_pos)

    # Smooth data
    led_pos = boxcar_smooth(led_pos, window_size=smoothing_window_size)

    # Calculate direction
    # To correct for light pos relative to rat subtract angle of large light
    correction = int(posdata["header"]["bearing_colour_1"])

    direction = np.mod(
        (180 / np.pi)
        * (
            np.arctan2(
                -led_pos.iloc[1, :] + led_pos.iloc[3, :],
                led_pos.iloc[0, :] - led_pos.iloc[2, :],
            )
        )
        - correction,
        360,
    )

    # Get position from smoothed individual lights
    headPos = 0.5  # Hard-coded for now, not included in current metadata
    # Proportional distance of the rat's head between the LEDs, 0.5 being halfway
    xy_pos = calculate_position(led_pos, headPos)

    # Calculate heading from displacement
    direction_disp = calculate_heading_direction(xy_pos)

    # Calculate speed
    speed = calculate_speed(xy_pos, pos_sample_rate=pos_sample_rate, pix_per_metre=ppm)

    return xy_pos, led_pos, led_pix, speed, direction, direction_disp
