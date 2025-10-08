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

        # Get position sampling rate
        pos_sample_rate = obj.pos_data[trial_iterator]["pos_sampling_rate"]

        sector_times = sector_samples / pos_sample_rate
    else:
        raise ValueError(
            f"Trial name {trial_name} does not correspond to a t-maze trial."
        )

    return sector_times
