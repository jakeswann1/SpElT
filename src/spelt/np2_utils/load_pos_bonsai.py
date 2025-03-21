import numpy as np
import pandas as pd


def load_pos_bonsai_laurenz(path, ppm):
    """
    Load position data from a .csv file generated by Bonsai
    Assumes columns are structured as follows:
    -

    CURRENTLY ASSUMES TWO LEDs ARE BEING TRACKED
    Required columns: Item2.X, Item2.Y, Item3.X, Item3.Y

    Parameters
    ----------
    path : str
        Path to the .csv file to be loaded
    ppm : int, optional
        Pixels per metre for scaling

    """

    # Read position data from csv file
    data = pd.read_csv(f"{path}/bonsai.csv", index_col=0)

    pos_header = {"pixels_per_metre": ppm}

    # Extract LED position data and tracked pixel size data
    led_pos = data.loc[:, ["Item2.X", "Item2.Y", "Item3.X", "Item3.Y"]].T

    # Scale pos data to specific PPM
    real_ppm = ppm
    goal_ppm = 400  # HARD CODED FOR NOW
    pos_header["scaled_ppm"] = goal_ppm
    scale_fact = goal_ppm / real_ppm

    # Scale pos data in place
    led_pos *= scale_fact

    raw_pos_data = {"header": pos_header, "led_pos": led_pos}

    return raw_pos_data


from dateutil import parser


def load_pos_bonsai_jake(path, ppm, trial_type):
    """
    Load position data from a .csv file generated by Bonsai
    Assumes columns are structured as follows:
    -

    CURRENTLY ASSUMES A SINGLE POINT IS BEING TRACKED
    Required columns: Item2.X, Item2.Y, Item3.X, Item3.Y

    Parameters
    ----------
    path : str
        Path to the .csv file to be loaded
    ppm : int, optional
        Pixels per metre for scaling

    """

    # Read position data from csv file
    data = pd.read_csv(path)
    # Drop first row as it may be a dodgy ttl pulse
    data = data.iloc[1:]

    if trial_type == "open-field":
        pointgrey_timestamps = data.loc[:, "Value.Item1"]
        frame_count = data.loc[:, "Value.Item2"]
        position = data.loc[:, ["Value.Item3.X", "Value.Item3.Y"]].T
        bonsai_timestamps = data.loc[:, "Timestamp"]
        ppm = 500  # TODO: make dynamic from sheet
        print("Estimating PPM for open-field at 800")

    elif trial_type == "t-maze":
        frame_count = data.loc[:, "Value.Item1"]
        pointgrey_timestamps = data.loc[:, "Value.Item2"]
        position = data.loc[:, ["Value.Item3.X", "Value.Item3.Y"]].T
        bonsai_timestamps = data.loc[:, "Timestamp"]
        ppm = 900  # TODO: make dynamic from sheet
        print("Estimating PPM for t-maze at 900")
    else:
        raise ValueError(f'Trial type "{trial_type}" not recognised')

    data.set_index(frame_count, inplace=True)
    # TODO add extracting maze state information, conditionally on trial type

    # Scale pos data to specific PPM
    goal_ppm = 400  # HARD CODED FOR NOW
    scale_fact = goal_ppm / ppm
    position *= scale_fact

    # Parse pointgrey timestamps: from https://groups.google.com/g/bonsai-users/c/WD6mV94KAQs
    time = pointgrey_timestamps.to_numpy()
    cycle1 = (time >> 12) & 0x1FFF
    cycle2 = (time >> 25) & 0x7F
    time = cycle2 + cycle1 / 8000.0
    cycles = np.insert(np.diff(time) < 0, 0, False)
    cycleindex = np.cumsum(cycles)
    pointgrey_timestamps = time + cycleindex * 128
    # offset so time starts at 0
    pointgrey_timestamps = pointgrey_timestamps - min(pointgrey_timestamps)

    # Estimate sampling rate
    sampling_rate = 1 / np.mean(np.diff(pointgrey_timestamps))

    # Parse bonsai timestamps - convert to seconds where 0 is the start of the recording
    bonsai_timestamps = bonsai_timestamps.apply(lambda x: parser.isoparse(x))
    bonsai_timestamps = bonsai_timestamps - bonsai_timestamps.iloc[0]
    bonsai_timestamps = bonsai_timestamps.apply(lambda x: x.total_seconds())
    start_time = bonsai_timestamps.iloc[0]

    position.columns = pointgrey_timestamps

    raw_pos_data = {
        "pos": position,
        "camera_timestamps": pointgrey_timestamps,
        "bonsai_timestamps": bonsai_timestamps,
        "sampling_rate": sampling_rate,
        "pixels_per_metre": ppm,
        "scaled_ppm": goal_ppm,
        "start_time": start_time,
    }

    return raw_pos_data
