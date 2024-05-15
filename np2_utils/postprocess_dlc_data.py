import struct
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from scipy.interpolate import interp1d

def led_speed_filter(led_pos, max_pix_per_sample):
    """
    This function filters out short runs of data caused by the tracker picking up an incorrect distant point.
    It resets the led_pos to NaN for those positions.

    Parameters:
    led_pos: pandas DataFrame of shape (4, n_pos) containing the x, y position of each LED, in the order [X1, Y1, X2, Y2]
    max_pix_per_sample: maximum allowed speed of the LED in pixels per sample
    led: index of the LED to filter (1 for LED 1, 2 for LED 2)

    Returns:
    n_jumpy: number of positions that were filtered out
    led_pos: filtered led_pos DataFrame
    """
    # For each LED (should be 2)
    for i in range(0,np.shape(led_pos)[0],2):
        
        # Get indices of positions where the LED was tracked
        ok_pos = np.where(~led_pos.iloc[i, :].isna() | ~led_pos.iloc[i+1, :].isna())[0]

        if len(ok_pos) < 2:
            print(f"Warning: < 2 tracked points for LED {i // 2 + 1}")

        mpps_sqd = max_pix_per_sample**2

        n_jumpy = 0
        prev_pos = ok_pos[0]
        
        for pos in ok_pos[1:]:
            # Calculate speed of shift from prev_pos in pixels per sample (squared)
            
            #Calculation:
            #Euclidean distance (in pixels) between current and previous position per LED squared (so positive),
            #divided by the number of samples between this position and the previous valid position all squared
            
            pix_per_sample_sqd = (led_pos.iloc[i, pos] - led_pos.iloc[i, prev_pos])**2 + (led_pos.iloc[i+1, pos] - led_pos.iloc[i+1, prev_pos])**2 / (pos-prev_pos)**2
            
            #Compare with threshold and add 1 to n_jumpy counter if too fast, or set as new valid prev_pos
            if pix_per_sample_sqd > mpps_sqd:
                led_pos.iloc[i:i+2, pos] = np.nan
                n_jumpy += 1
            else:
                prev_pos = pos

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
    from scipy.signal import convolve, boxcar

    # Create a boxcar window
    window = boxcar(window_size)

    # Apply boxcar smoothing
    df_smoothed = df.apply(lambda x: pd.Series(convolve(x, window, mode='same') / window_size), axis=1)

    # Interpolate between existing values to handle NaN values
    df_smoothed = df_smoothed.interpolate(method='linear', axis=0)

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
    pos = led_pos.iloc[:2, :].values * (1 - proportion) + led_pos.iloc[2:, :].values * proportion
    return pd.DataFrame(pos, index=['X', 'Y'])

def calculate_heading_direction(pos):
    """
    This function calculates the heading direction from position displacement.

    Parameters:
    pos: pandas DataFrame of shape (2, n_pos) containing the x, y position

    Returns:
    dir_disp: heading direction from position displacement
    """
    dir_disp = np.mod(np.arctan2(-pos.iloc[1, 1:].values + pos.iloc[1, :-1].values, pos.iloc[0, 1:].values - pos.iloc[0, :-1].values) * 180 / np.pi, 360)
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
    speed = np.sqrt((pos.iloc[0, 1:].values - pos.iloc[0, :-1].values)**2 + (pos.iloc[1, 1:].values - pos.iloc[1, :-1].values)**2) * (100 * pos_sample_rate / pix_per_metre)
    speed = np.append(speed, speed[-1])  # Duplicate last value

    return speed

# Function to process position data
def postprocess_dlc_data(posdata, max_speed, smoothing_window_size):

    tracked_points = posdata['bodypart_pos']
    
    # Filter points for those moving impossibly fast and set tracked_points_x to NaN
    ppm = int(posdata['header']['scaled_ppm'])
    pos_sample_rate = float(posdata['header']['sample_rate'])
    
    max_pix_per_sample = max_speed*ppm/pos_sample_rate
    n_jumpy, tracked_points = led_speed_filter(tracked_points, max_pix_per_sample*100)
    
    # Interpolate NaN values
    tracked_points = interpolate_nan_values(tracked_points)
    
    # Smooth data
    tracked_points = boxcar_smooth(tracked_points, window_size=smoothing_window_size)
        
    # Calculate direction
    # To correct for tracked point orientation, we need to subtract the head point angle value from the direction
    correction = int(posdata['header']['tracked_point_angle_1'])
    
    direction = np.mod((180/np.pi) * (np.arctan2(-tracked_points.iloc[1, :] + tracked_points.iloc[3, :], tracked_points.iloc[0, :] - tracked_points.iloc[2, :])) - correction, 360)

    # Get position from smoothed individual lights
    headPos = 0.5 # Hard-coded for now, not included in current metadata
                  # Proportional distance of the rat's head between the LEDs, 0.5 being halfway
    xy_pos = calculate_position(tracked_points, headPos)
    
    # Calculate heading from displacement
    direction_disp = calculate_heading_direction(xy_pos)
    
    # Calculate speed
    speed = calculate_speed(xy_pos, pos_sample_rate=pos_sample_rate, pix_per_metre=ppm)

    return xy_pos, tracked_points, speed, direction, direction_disp
