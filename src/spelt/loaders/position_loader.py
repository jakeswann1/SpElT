"""Position data loading functions for various tracking systems."""

from pathlib import Path

import numpy as np


def load_position_data(
    tracking_type: str,
    path: Path,
    trial_name: str,
    trial_type: str,
    sync_data: dict | None = None,
    max_speed: float = 5.0,
    smoothing_window: int = 3,
    output_flag: bool = True,
) -> dict:
    """
    Load position data for a single trial based on tracking type.

    Parameters
    ----------
    tracking_type : str
        Type of tracking system ('axona', 'bonsai_roi', 'bonsai_leds', 'dlc')
    path : Path
        Path to the trial data
    trial_name : str
        Name of the trial
    trial_type : str
        Type of trial (e.g., 'open-field', 't-maze')
    sync_data : dict, optional
        TTL timestamps for Bonsai synchronization (required for bonsai_roi)
    max_speed : float
        Maximum speed for position filtering (m/s)
    smoothing_window : int
        Window size for position smoothing
    output_flag : bool
        Print loading messages

    Returns
    -------
    dict
        Position data with keys:
        - 'xy_position': DataFrame with x, y coordinates
        - 'speed': Speed array
        - 'direction': Direction array (if available)
        - 'direction_from_displacement': Direction from position changes
        - 'pos_sampling_rate': Sampling rate
        - 'scaled_ppm': Pixels per meter
        - Additional keys depending on tracking type
    """
    if tracking_type == "axona":
        return load_axona_position(
            path, trial_name, max_speed, smoothing_window, output_flag
        )
    elif tracking_type == "bonsai_roi":
        return load_bonsai_roi_position(
            path, trial_type, sync_data, max_speed, smoothing_window, output_flag
        )
    elif tracking_type == "bonsai_leds":
        return load_bonsai_leds_position(
            path, trial_type, max_speed, smoothing_window, output_flag
        )
    elif (path / "dlc.csv").exists():
        return load_dlc_position(path, max_speed, smoothing_window, output_flag)
    else:
        raise ValueError(f"Unsupported tracking type: {tracking_type}")


def load_axona_position(
    path: Path,
    trial_name: str,
    max_speed: float,
    smoothing_window: int,
    output_flag: bool,
) -> dict:
    """
    Load and process Axona position data.

    Parameters
    ----------
    path : Path
        Path to trial directory
    trial_name : str
        Name of the trial
    max_speed : float
        Maximum speed for filtering
    smoothing_window : int
        Window size for smoothing
    output_flag : bool
        Print loading messages

    Returns
    -------
    dict
        Position data dictionary
    """
    from spelt.axona_utils.axona_preprocessing import pos_from_bin
    from spelt.axona_utils.load_pos_axona import load_pos_axona
    from spelt.axona_utils.postprocess_pos_data import (
        postprocess_pos_data,
        write_csv_from_pos,
    )

    # Special PPM override for t-maze trials
    override_ppm = 615 if "t-maze" in trial_name else None
    if override_ppm and output_flag:
        print("Real PPM artificially set to 615 (t-maze default)")

    # Try loading from various file formats with fallbacks
    try:
        raw_pos_data, pos_sampling_rate = load_pos_axona(path, override_ppm)
    except FileNotFoundError:
        if output_flag:
            print("No .csv file found, trying to load from .bin file")
        try:
            pos_from_bin(path)
            raw_pos_data, pos_sampling_rate = load_pos_axona(
                path / trial_name, override_ppm
            )
        except FileNotFoundError:
            if output_flag:
                print("No .csv or .bin file found, trying to load from .pos file")
            write_csv_from_pos(path.with_suffix(".pos"))
            raw_pos_data, pos_sampling_rate = load_pos_axona(path, override_ppm)

    # Postprocess position data
    xy_pos, led_pos, led_pix, speed, direction, direction_disp = postprocess_pos_data(
        raw_pos_data, max_speed, smoothing_window
    )

    # Rescale timestamps to seconds
    xy_pos.columns /= pos_sampling_rate
    led_pos.columns /= pos_sampling_rate
    led_pix.columns /= pos_sampling_rate

    return {
        "header": raw_pos_data.get("header"),
        "xy_position": xy_pos,
        "led_positions": led_pos,
        "led_pixel_size": led_pix,
        "speed": speed,
        "direction": direction,
        "direction_from_displacement": direction_disp,
        "pos_sampling_rate": pos_sampling_rate,
        "scaled_ppm": 400,
    }


def load_bonsai_roi_position(
    path: Path,
    trial_type: str,
    sync_data: dict | None,
    max_speed: float,
    smoothing_window: int,
    output_flag: bool,
) -> dict:
    """
    Load and process Bonsai ROI tracking data.

    Parameters
    ----------
    path : Path
        Path to trial CSV file
    trial_type : str
        Type of trial
    sync_data : dict
        Dictionary with 'ttl_timestamps' key
    max_speed : float
        Maximum speed for filtering
    smoothing_window : int
        Window size for smoothing
    output_flag : bool
        Print loading messages

    Returns
    -------
    dict
        Position data dictionary
    """
    from spelt.np2_utils.load_pos_bonsai import load_pos_bonsai_roi
    from spelt.np2_utils.postprocess_pos_data_np2 import (
        postprocess_bonsai_jake,
        sync_bonsai_jake,
    )

    if sync_data is None:
        raise ValueError(
            "sync_data with TTL timestamps required for bonsai_roi loading"
        )

    ttl_times = sync_data.get("ttl_timestamps")
    if ttl_times is None:
        raise ValueError("TTL timestamps not found in sync_data")

    ttl_freq = 1 / np.mean(np.diff(ttl_times[2:]))

    if output_flag:
        print(f"Loading raw Bonsai position data from path {path}")

    # Try loading with standard path, fallback to T-maze capitalization
    try:
        raw_pos_data = load_pos_bonsai_roi(path.with_suffix(".csv"), 400, trial_type)
    except FileNotFoundError:
        path = path.with_suffix(".csv").replace("t-maze", "T-maze")
        if output_flag:
            print(f"Looking for Bonsai file with name {path}")
        raw_pos_data = load_pos_bonsai_roi(path, 400, trial_type)

    # Postprocess position data
    xy_pos, speed, direction_disp = postprocess_bonsai_jake(
        raw_pos_data, max_speed, smoothing_window
    )

    # Synchronize with TTL pulses
    pos_sampling_rate = 1 / np.mean(np.diff(ttl_times))
    xy_pos, speed, direction_disp = sync_bonsai_jake(
        xy_pos, ttl_times, pos_sampling_rate, speed, direction_disp
    )

    return {
        "xy_position": xy_pos,
        "speed": speed,
        "direction_from_displacement": direction_disp,
        "ttl_times": ttl_times,
        "ttl_freq": ttl_freq,
        "pos_sampling_rate": pos_sampling_rate,
        "scaled_ppm": 400,
        "header": raw_pos_data["header"],
        "maze_roi": raw_pos_data["maze_roi"],
        "maze_state": raw_pos_data["maze_state"],
    }


def load_bonsai_leds_position(
    path: Path,
    trial_type: str,
    max_speed: float,
    smoothing_window: int,
    output_flag: bool,
) -> dict:
    """
    Load and process Bonsai LED tracking data.

    Parameters
    ----------
    path : Path
        Path to trial CSV file
    trial_type : str
        Type of trial
    max_speed : float
        Maximum speed for filtering
    smoothing_window : int
        Window size for smoothing
    output_flag : bool
        Print loading messages

    Returns
    -------
    dict
        Position data dictionary
    """
    from spelt.np2_utils.load_pos_bonsai import load_pos_bonsai_leds
    from spelt.np2_utils.postprocess_pos_data_np2 import postprocess_bonsai_jake

    if output_flag:
        print("Loading raw Bonsai position data (LED tracking)")

    raw_pos_data = load_pos_bonsai_leds(path.with_suffix(".csv"), 400, trial_type)

    xy_pos, speed, direction_disp = postprocess_bonsai_jake(
        raw_pos_data, max_speed, smoothing_window
    )

    return {
        "xy_position": xy_pos,
        "speed": speed,
        "direction_from_displacement": direction_disp,
        "pos_sampling_rate": raw_pos_data["sampling_rate"],
        "scaled_ppm": 400,
    }


def load_dlc_position(
    path: Path, max_speed: float, smoothing_window: int, output_flag: bool
) -> dict:
    """
    Load and process DeepLabCut tracking data.

    Parameters
    ----------
    path : Path
        Path to trial directory containing dlc.csv
    max_speed : float
        Maximum speed for filtering
    smoothing_window : int
        Window size for smoothing
    output_flag : bool
        Print loading messages

    Returns
    -------
    dict
        Position data dictionary
    """
    from spelt.np2_utils.load_pos_dlc import load_pos_dlc
    from spelt.np2_utils.postprocess_pos_data_np2 import postprocess_dlc_data

    if output_flag:
        print("Loading DLC position data")

    raw_pos_data = load_pos_dlc(path, 400)
    raw_pos_data["header"]["tracked_point_angle_1"] = 0

    xy_pos, tracked_points, speed, direction, direction_disp = postprocess_dlc_data(
        raw_pos_data, max_speed, smoothing_window
    )

    return {
        "header": raw_pos_data["header"],
        "xy_position": xy_pos,
        "tracked_points": tracked_points,
        "speed": speed,
        "direction": direction,
        "direction_from_displacement": direction_disp,
        "bonsai_timestamps": raw_pos_data["bonsai_timestamps"],
        "camera_timestamps": raw_pos_data["camera_timestamps"],
        "scaled_ppm": 400,
    }
