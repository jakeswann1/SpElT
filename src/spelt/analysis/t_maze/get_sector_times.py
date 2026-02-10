import numpy as np

from spelt.analysis.t_maze.assign_sectors import assign_sectors
from spelt.ephys import ephys


def get_sector_times(obj: ephys, sector_list: list[int], trial_name) -> ephys:
    trial_iterator = obj.trial_list.index(trial_name)
    if "t-maze" in trial_name:
        # Assign sectors to positions samples
        sector_numbers = np.array(
            assign_sectors(
                obj.pos_data[trial_iterator]["xy_position"].T,
                pos_header=obj.pos_data[trial_iterator]["header"],
            )
        )

        # Find position samples where the animal is in each arm type
        sector_samples = np.where(np.isin(sector_numbers, sector_list))[0]

        # Use actual position timestamps instead of calculated times
        # This ensures alignment with LFP/ephys timestamps
        # Different tracking systems use different timestamp keys
        pos_data = obj.pos_data[trial_iterator]
        if "bonsai_timestamps" in pos_data:
            timestamps = pos_data["bonsai_timestamps"]
        elif "ttl_times" in pos_data:
            timestamps = pos_data["ttl_times"]
        elif "camera_timestamps" in pos_data:
            timestamps = pos_data["camera_timestamps"]
        else:
            # Fallback to calculated times from sample rate
            pos_sample_rate = pos_data["pos_sampling_rate"]
            timestamps = np.arange(len(sector_numbers)) / pos_sample_rate

        # Position timestamps are relative to trial start (0-based)
        # LFP timestamps are now also relative via timestamps_relative field
        # No offset needed - both use same time frame
        sector_times = timestamps[sector_samples]
    else:
        raise ValueError(
            f"Trial name {trial_name} does not correspond to a t-maze trial."
        )

    return sector_times
