import numpy as np

from ..axona_utils.postprocess_pos_data import (
    boxcar_smooth,
    calculate_heading_direction,
    calculate_position,
    calculate_speed,
    interpolate_nan_values,
    led_speed_filter,
)


# Function to process position data
def postprocess_dlc_data(posdata, max_speed, smoothing_window_size):

    if "bodypart_pos" in posdata:
        tracked_points = posdata["bodypart_pos"]
        # To correct for tracked point orientation,
        # we need to subtract the head point angle value from the direction
        correction = int(posdata["header"]["tracked_point_angle_1"])
    elif "led_pos" in posdata:
        tracked_points = posdata["led_pos"]
        # To correct for tracked point orientation,
        # we need to subtract position of LED 1 from the direction
        correction = int(posdata["header"]["bearing_colour_1"])

    # Filter points for those moving impossibly fast and set tracked_points_x to NaN
    ppm = int(posdata["header"]["scaled_ppm"])
    pos_sample_rate = float(posdata["header"]["sample_rate"])

    max_pix_per_sample = max_speed * ppm / pos_sample_rate

    # Interpolate NaN values
    tracked_points = interpolate_nan_values(tracked_points)

    n_jumpy, tracked_points = led_speed_filter(tracked_points, max_pix_per_sample * 100)

    # Smooth data
    tracked_points = boxcar_smooth(tracked_points, window_size=smoothing_window_size)

    # Calculate direction
    direction = np.mod(
        (180 / np.pi)
        * (
            np.arctan2(
                -tracked_points.iloc[1, :] + tracked_points.iloc[3, :],
                tracked_points.iloc[0, :] - tracked_points.iloc[2, :],
            )
        )
        - correction,
        360,
    )

    # Get position from smoothed individual lights
    head_pos = 0.5  # Hard-coded for now, not included in current metadata
    # Proportional distance of the rat's head between the LEDs, 0.5 being halfway
    xy_pos = calculate_position(tracked_points, head_pos)

    # Calculate heading from displacement
    direction_disp = calculate_heading_direction(xy_pos)

    # Calculate speed
    speed = calculate_speed(xy_pos, pos_sample_rate=pos_sample_rate, pix_per_metre=ppm)

    return xy_pos, tracked_points, speed, direction, direction_disp


def postprocess_bonsai_jake(posdata, max_speed, smoothing_window_size):

    raw_pos = posdata["pos"]
    ppm = posdata["scaled_ppm"]
    sampling_rate = posdata["sampling_rate"]

    raw_pos = interpolate_nan_values(raw_pos)

    smoothed_pos = boxcar_smooth(raw_pos, window_size=smoothing_window_size)
    smoothed_pos.index = ["X", "Y"]

    # Calculate heading from displacement,
    # can't calculate true direction as only a single point is tracked
    direction_disp = calculate_heading_direction(smoothed_pos)

    speed = calculate_speed(
        smoothed_pos, pos_sample_rate=sampling_rate, pix_per_metre=ppm
    )

    return smoothed_pos, speed, direction_disp


def sync_bonsai_jake(xy_pos, ttl_times, pos_sampling_rate, speed, direction_disp):
    if len(xy_pos.columns) == len(ttl_times):
        xy_pos.columns = ttl_times
    elif len(xy_pos.columns) < len(ttl_times):
        xy_pos.columns = ttl_times[: len(xy_pos.columns)]
        print(
            f"""
            WARNING: Bonsai position data has
            more TTL pulses than position samples times.
            Position samples: {len(xy_pos.columns)}
            vs TTL times: {len(ttl_times)}
            Removing {len(ttl_times) - len(xy_pos.columns)}
            TTL pulses from the end of TTL times.
            Data may be up to
            {(len(ttl_times) - len(xy_pos.columns))/pos_sampling_rate}
                seconds out of sync.
            """
        )
    elif len(xy_pos.columns) > len(ttl_times):
        print(
            f"""
            WARNING: Bonsai position data has
            more position samples than TTL pulses.
            Position samples: {len(xy_pos.columns)}
            vs TTL times: {len(ttl_times)}
            Removing {len(xy_pos.columns) - len(ttl_times)}
            position samples from the end of position data.
            Data may be up to
            {(len(xy_pos.columns) - len(ttl_times))/pos_sampling_rate}
                seconds out of sync.
            """
        )
        xy_pos = xy_pos.iloc[:, : len(ttl_times)]
        xy_pos.columns = ttl_times
        speed = speed[: len(ttl_times)]
        direction_disp = direction_disp[: len(ttl_times)]
