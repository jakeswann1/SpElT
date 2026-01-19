"""Functions for identifying choice trajectories in T-maze recordings."""

import numpy as np

from spelt.analysis.t_maze.arm_state_detection import find_choice_cycles
from spelt.analysis.t_maze.assign_sectors import assign_sectors


def identify_choice_trajectories_from_sectors(
    sectors: np.ndarray,
    pos_timestamps: np.ndarray,
    trial_idx: int = 0,
    validate_spatial_exclusivity: bool = True,
) -> tuple[list[tuple], list[tuple], dict]:
    """
    Identify left and right choice trajectories from sector assignments.

    Uses the same logic as calculate_choices.py: a choice trajectory is a complete
    cycle from center (sector 8) → goal arm (sector 1 or 9) → back to center (sector 8).

    Parameters
    ----------
    sectors : np.ndarray
        Array of sector numbers for each position sample (1-12)
    pos_timestamps : np.ndarray
        Timestamps corresponding to each position sample (in seconds)
    trial_idx : int, optional
        Trial index to include in output tuples (default: 0)
    validate_spatial_exclusivity : bool, optional
        If True, reject trajectories where animal enters opposite arm.
        For left trajectories: reject if sectors 9-11 appear.
        For right trajectories: reject if sectors 1-3 appear.
        Default: True (validation enabled).

    Returns
    -------
    left_windows : list of tuples
        Each tuple is (trial_idx, start_time, end_time) for left-choice trajectories
    right_windows : list of tuples
        Each tuple is (trial_idx, start_time, end_time) for right-choice trajectories
    rejection_stats : dict
        Statistics about rejected trajectories:
        - 'n_left_rejected': number of left trajectories rejected
        - 'n_right_rejected': number of right trajectories rejected
        - 'n_left_accepted': number of left trajectories accepted
        - 'n_right_accepted': number of right trajectories accepted

    Notes
    -----
    T-maze sector layout (4×3 grid):
    - Sectors 1-4: Left goal arm (top row)
    - Sectors 5-8: Center stem (middle row, 8 = start box)
    - Sectors 9-12: Right goal arm (bottom row)

    Choice detection uses state machine with trigger sectors:
    - Sector 8: Center/start box (beginning and end of choice cycle)
    - Sector 1: Left arm entry (choice made)
    - Sector 9: Right arm entry (choice made)

    A complete choice trajectory:
    center (8) → left (1) or right (9) → back to center (8)
    """
    # Use shared logic to find choice cycles
    cycles, rejection_stats = find_choice_cycles(
        sectors,
        include_incomplete=False,
        validate_spatial_exclusivity=validate_spatial_exclusivity,
    )

    # Convert cycles to time windows
    left_windows = []
    right_windows = []

    for cycle in cycles:
        if cycle["choice"] == "left":
            left_windows.append(
                (
                    trial_idx,
                    pos_timestamps[cycle["start_idx"]],
                    pos_timestamps[cycle["end_idx"]],
                )
            )
        elif cycle["choice"] == "right":
            right_windows.append(
                (
                    trial_idx,
                    pos_timestamps[cycle["start_idx"]],
                    pos_timestamps[cycle["end_idx"]],
                )
            )

    return left_windows, right_windows, rejection_stats


def identify_choice_trajectories_single_trial(
    pos_data: dict,
    trial_idx: int,
    pos_header: dict | None = None,
    validate_spatial_exclusivity: bool = True,
) -> tuple[list[tuple], list[tuple], dict]:
    """
    Identify choice trajectories from position data for a single trial.

    Convenience wrapper that handles sector assignment internally.

    Parameters
    ----------
    pos_data : dict
        Position data dictionary with 'xy_position' key containing a DataFrame
        with x and y coords as rows and time as columns
    trial_idx : int
        Trial index to include in output tuples
    pos_header : dict, optional
        Position header with min/max x/y boundaries. If None, will be
        calculated from position data.
    validate_spatial_exclusivity : bool, optional
        If True, reject trajectories where animal enters opposite arm.
        Default: True (validation enabled).

    Returns
    -------
    left_windows : list of tuples
        Each tuple is (trial_idx, start_time, end_time) for left-choice trajectories
    right_windows : list of tuples
        Each tuple is (trial_idx, start_time, end_time) for right-choice trajectories
    rejection_stats : dict
        Statistics about rejected trajectories
    """
    xy_pos = pos_data["xy_position"]
    pos_timestamps = xy_pos.columns.to_numpy()

    # Assign sectors
    sectors = assign_sectors(xy_pos.T, pos_header=pos_header)

    # Identify trajectories
    return identify_choice_trajectories_from_sectors(
        sectors, pos_timestamps, trial_idx, validate_spatial_exclusivity
    )


def identify_choice_trajectories_batch(
    pos_data_list: list[dict],
    trial_indices: list[int],
    pos_headers: list[dict | None] | None = None,
    validate_spatial_exclusivity: bool = True,
) -> tuple[list[tuple], list[tuple], dict]:
    """
    Identify choice trajectories across multiple trials.

    Parameters
    ----------
    pos_data_list : list of dict
        List of position data dictionaries (one per trial)
    trial_indices : list of int
        List of trial indices corresponding to each pos_data
    pos_headers : list of dict or None, optional
        List of position headers (one per trial). If None, will calculate
        from position data for each trial.
    validate_spatial_exclusivity : bool, optional
        If True, reject trajectories where animal enters opposite arm.
        Default: True (validation enabled).

    Returns
    -------
    left_windows : list of tuples
        All left-choice trajectory windows pooled across trials
        Each tuple is (trial_idx, start_time, end_time)
    right_windows : list of tuples
        All right-choice trajectory windows pooled across trials
        Each tuple is (trial_idx, start_time, end_time)
    rejection_stats : dict
        Cumulative statistics about rejected trajectories across all trials
    """
    all_left_windows = []
    all_right_windows = []

    # Initialize cumulative rejection statistics
    cumulative_rejection_stats = {
        "n_left_rejected": 0,
        "n_right_rejected": 0,
        "n_left_accepted": 0,
        "n_right_accepted": 0,
    }

    if pos_headers is None:
        pos_headers = [None] * len(pos_data_list)

    for pos_data, trial_idx, pos_header in zip(
        pos_data_list, trial_indices, pos_headers
    ):
        if pos_data is None:
            continue

        left_wins, right_wins, rejection_stats = (
            identify_choice_trajectories_single_trial(
                pos_data, trial_idx, pos_header, validate_spatial_exclusivity
            )
        )

        all_left_windows.extend(left_wins)
        all_right_windows.extend(right_wins)

        # Accumulate rejection statistics
        for key in cumulative_rejection_stats:
            cumulative_rejection_stats[key] += rejection_stats[key]

    return all_left_windows, all_right_windows, cumulative_rejection_stats


def identify_choice_trajectories_from_ephys(
    obj, trial_list: list[int] | None = None, validate_spatial_exclusivity: bool = True
) -> tuple[list[tuple], list[tuple], dict]:
    """
    Identify choice trajectories from an ephys object.

    Convenience function that handles position loading and trial iteration.

    Parameters
    ----------
    obj : ephys
        ephys object with position data loaded or loadable
    trial_list : list of int, optional
        List of trial indices to process. If None, processes all trials
        with 't-maze' in their name.
    validate_spatial_exclusivity : bool, optional
        If True, reject trajectories where animal enters opposite arm.
        Default: True (validation enabled).

    Returns
    -------
    left_windows : list of tuples
        All left-choice trajectory windows pooled across trials
        Each tuple is (trial_idx, start_time, end_time)
    right_windows : list of tuples
        All right-choice trajectory windows pooled across trials
        Each tuple is (trial_idx, start_time, end_time)
    rejection_stats : dict
        Cumulative statistics about rejected trajectories across all trials

    Notes
    -----
    If position data is not already loaded for the specified trials,
    this function will load it.
    """
    # Find T-maze trials if not specified
    if trial_list is None:
        trial_list = [
            idx for idx, trial in enumerate(obj.trial_list) if "t-maze" in trial.lower()
        ]

    # Ensure position data is loaded
    obj.load_pos(trial_list=trial_list, reload_flag=False, output_flag=False)

    # Collect position data and headers
    pos_data_list = []
    pos_headers = []
    valid_trial_indices = []

    for trial_idx in trial_list:
        if obj.pos_data[trial_idx] is not None:
            pos_data_list.append(obj.pos_data[trial_idx])
            pos_headers.append(obj.pos_data[trial_idx].get("header", None))
            valid_trial_indices.append(trial_idx)

    # Identify trajectories
    return identify_choice_trajectories_batch(
        pos_data_list, valid_trial_indices, pos_headers, validate_spatial_exclusivity
    )
