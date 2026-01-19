"""Shared utilities for detecting arm states and choice cycles in T-maze data."""

import numpy as np


def identify_arm_states(sectors: np.ndarray) -> np.ndarray:
    """
    Convert sector numbers to arm states using trigger sectors.

    Uses state machine with trigger sectors:
    - Sector 8: Center/start box → state = 0
    - Sector 1: Left arm entry → state = 1
    - Sector 9: Right arm entry → state = 2

    Parameters
    ----------
    sectors : np.ndarray
        Array of sector numbers (1-12) for each position sample

    Returns
    -------
    arm_states : np.ndarray
        Array of arm states for each position sample:
        - 0: Centre
        - 1: Left
        - 2: Right
        - -1: Unknown (before any trigger sector hit)
    """
    arm_states = np.full(len(sectors), -1, dtype=int)
    choice = ""

    for i in range(len(sectors)):
        # Update state based on trigger sectors
        if sectors[i] == 8:
            choice = "centre"
        elif sectors[i] == 1:
            choice = "left"
        elif sectors[i] == 9:
            choice = "right"

        # Encode state
        if choice == "centre":
            arm_states[i] = 0
        elif choice == "left":
            arm_states[i] = 1
        elif choice == "right":
            arm_states[i] = 2

    return arm_states


def get_arm_visit_sequence(arm_states: np.ndarray) -> np.ndarray:
    """
    Get sequence of arm visits by removing consecutive duplicates.

    Parameters
    ----------
    arm_states : np.ndarray
        Array of arm states (0=centre, 1=left, 2=right, -1=unknown)

    Returns
    -------
    arm_visit_order : np.ndarray
        Sequence of unique arm visits (consecutive duplicates removed)
    """
    return arm_states[np.concatenate([[True], np.diff(arm_states) != 0])]


def validate_trajectory_spatial_exclusivity(
    sectors: np.ndarray, start_idx: int, end_idx: int, choice: str
) -> bool:
    """
    Validate that a trajectory doesn't enter the opposite arm.

    Rejects trajectories where the animal enters the opposite arm's entry
    or proximal regions. Terminal sectors (4, 12) are excluded from validation
    as animals may legitimately pass through these boundary regions.

    Parameters
    ----------
    sectors : np.ndarray
        Full array of sector numbers (1-12)
    start_idx : int
        Start index of trajectory
    end_idx : int
        End index of trajectory
    choice : str
        'left' or 'right'

    Returns
    -------
    bool
        True if trajectory is spatially valid (no opposite arm entry)
        False if trajectory enters opposite arm

    Notes
    -----
    Sector layout (4×3 grid):
    - Sectors 1-4: Left arm (1=entry trigger, 4=terminal)
    - Sectors 5-8: Center stem (8=start box)
    - Sectors 9-12: Right arm (9=entry trigger, 12=terminal)

    Validation criteria:
    - Left trajectories: reject if sectors 9, 10, 11 appear
    - Right trajectories: reject if sectors 1, 2, 3 appear
    """
    # Extract trajectory sector sequence
    traj_sectors = sectors[start_idx : end_idx + 1]

    # Filter out NaN values
    traj_sectors = traj_sectors[~np.isnan(traj_sectors)]

    if len(traj_sectors) == 0:
        return True  # Empty trajectory is valid

    if choice == "left":
        # Reject if right arm entry/proximal sectors appear (excluding terminal 12)
        opposite_sectors = [9, 10, 11]
        return not np.any(np.isin(traj_sectors, opposite_sectors))
    elif choice == "right":
        # Reject if left arm entry/proximal sectors appear (excluding terminal 4)
        opposite_sectors = [1, 2, 3]
        return not np.any(np.isin(traj_sectors, opposite_sectors))
    else:
        return True  # Unknown choice type, pass validation


def find_choice_cycles(
    sectors: np.ndarray,
    include_incomplete: bool = False,
    validate_spatial_exclusivity: bool = True,
) -> tuple[list[dict], dict]:
    """
    Find complete choice cycles in sector data.

    A complete choice cycle: centre (8) → left (1) or right (9) → centre (8)

    Parameters
    ----------
    sectors : np.ndarray
        Array of sector numbers (1-12)
    include_incomplete : bool, optional
        If True, include incomplete cycles (no return to center)
    validate_spatial_exclusivity : bool, optional
        If True, reject trajectories where animal enters opposite arm.
        For left trajectories: reject if sectors 9-11 appear.
        For right trajectories: reject if sectors 1-3 appear.
        Default: True (validation enabled).

    Returns
    -------
    cycles : list of dict
        Each dict contains:
        - 'choice': 'left' or 'right'
        - 'start_idx': index where cycle starts (entering sector 8)
        - 'end_idx': index where cycle ends (returning to sector 8)
        - 'arm_entry_idx': index where arm was entered (sector 1 or 9)
        - 'complete': True if returned to center, False otherwise
    rejection_stats : dict
        Statistics about rejected trajectories:
        - 'n_left_rejected': number of left trajectories rejected
        - 'n_right_rejected': number of right trajectories rejected
        - 'n_left_accepted': number of left trajectories accepted
        - 'n_right_accepted': number of right trajectories accepted
    """
    cycles = []
    choice = ""

    # Initialize rejection statistics
    rejection_stats = {
        "n_left_rejected": 0,
        "n_right_rejected": 0,
        "n_left_accepted": 0,
        "n_right_accepted": 0,
    }

    i = 0
    while i < len(sectors):
        current_sector = sectors[i]

        # Track center entry
        if current_sector == 8:
            choice = "centre"
            centre_start_idx = i

        # Track left arm entry
        elif current_sector == 1:
            if choice == "centre":
                # Look ahead for return to center
                j = i + 1
                while j < len(sectors) and sectors[j] != 8:
                    j += 1

                if j < len(sectors):
                    # Complete cycle - validate before appending
                    cycle = {
                        "choice": "left",
                        "start_idx": centre_start_idx,
                        "end_idx": j,
                        "arm_entry_idx": i,
                        "complete": True,
                    }

                    # Apply spatial validation if enabled
                    if validate_spatial_exclusivity:
                        is_valid = validate_trajectory_spatial_exclusivity(
                            sectors, centre_start_idx, j, "left"
                        )
                        if is_valid:
                            cycles.append(cycle)
                            rejection_stats["n_left_accepted"] += 1
                        else:
                            rejection_stats["n_left_rejected"] += 1
                    else:
                        cycles.append(cycle)

                    i = j - 1
                elif include_incomplete:
                    # Incomplete cycle - validate before appending
                    cycle = {
                        "choice": "left",
                        "start_idx": centre_start_idx,
                        "end_idx": len(sectors) - 1,
                        "arm_entry_idx": i,
                        "complete": False,
                    }

                    # Apply spatial validation if enabled
                    if validate_spatial_exclusivity:
                        is_valid = validate_trajectory_spatial_exclusivity(
                            sectors, centre_start_idx, len(sectors) - 1, "left"
                        )
                        if is_valid:
                            cycles.append(cycle)
                            rejection_stats["n_left_accepted"] += 1
                        else:
                            rejection_stats["n_left_rejected"] += 1
                    else:
                        cycles.append(cycle)
                choice = "left"

        # Track right arm entry
        elif current_sector == 9:
            if choice == "centre":
                # Look ahead for return to center
                j = i + 1
                while j < len(sectors) and sectors[j] != 8:
                    j += 1

                if j < len(sectors):
                    # Complete cycle - validate before appending
                    cycle = {
                        "choice": "right",
                        "start_idx": centre_start_idx,
                        "end_idx": j,
                        "arm_entry_idx": i,
                        "complete": True,
                    }

                    # Apply spatial validation if enabled
                    if validate_spatial_exclusivity:
                        is_valid = validate_trajectory_spatial_exclusivity(
                            sectors, centre_start_idx, j, "right"
                        )
                        if is_valid:
                            cycles.append(cycle)
                            rejection_stats["n_right_accepted"] += 1
                        else:
                            rejection_stats["n_right_rejected"] += 1
                    else:
                        cycles.append(cycle)

                    i = j - 1
                elif include_incomplete:
                    # Incomplete cycle - validate before appending
                    cycle = {
                        "choice": "right",
                        "start_idx": centre_start_idx,
                        "end_idx": len(sectors) - 1,
                        "arm_entry_idx": i,
                        "complete": False,
                    }

                    # Apply spatial validation if enabled
                    if validate_spatial_exclusivity:
                        is_valid = validate_trajectory_spatial_exclusivity(
                            sectors, centre_start_idx, len(sectors) - 1, "right"
                        )
                        if is_valid:
                            cycles.append(cycle)
                            rejection_stats["n_right_accepted"] += 1
                        else:
                            rejection_stats["n_right_rejected"] += 1
                    else:
                        cycles.append(cycle)
                choice = "right"

        i += 1

    return cycles, rejection_stats
