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


def find_choice_cycles(
    sectors: np.ndarray, include_incomplete: bool = False
) -> list[dict]:
    """
    Find complete choice cycles in sector data.

    A complete choice cycle: centre (8) → left (1) or right (9) → centre (8)

    Parameters
    ----------
    sectors : np.ndarray
        Array of sector numbers (1-12)
    include_incomplete : bool, optional
        If True, include incomplete cycles (no return to center)

    Returns
    -------
    cycles : list of dict
        Each dict contains:
        - 'choice': 'left' or 'right'
        - 'start_idx': index where cycle starts (entering sector 8)
        - 'end_idx': index where cycle ends (returning to sector 8)
        - 'arm_entry_idx': index where arm was entered (sector 1 or 9)
        - 'complete': True if returned to center, False otherwise
    """
    cycles = []
    choice = ""

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
                    # Complete cycle
                    cycles.append(
                        {
                            "choice": "left",
                            "start_idx": centre_start_idx,
                            "end_idx": j,
                            "arm_entry_idx": i,
                            "complete": True,
                        }
                    )
                    i = j - 1
                elif include_incomplete:
                    # Incomplete cycle
                    cycles.append(
                        {
                            "choice": "left",
                            "start_idx": centre_start_idx,
                            "end_idx": len(sectors) - 1,
                            "arm_entry_idx": i,
                            "complete": False,
                        }
                    )
                choice = "left"

        # Track right arm entry
        elif current_sector == 9:
            if choice == "centre":
                # Look ahead for return to center
                j = i + 1
                while j < len(sectors) and sectors[j] != 8:
                    j += 1

                if j < len(sectors):
                    # Complete cycle
                    cycles.append(
                        {
                            "choice": "right",
                            "start_idx": centre_start_idx,
                            "end_idx": j,
                            "arm_entry_idx": i,
                            "complete": True,
                        }
                    )
                    i = j - 1
                elif include_incomplete:
                    # Incomplete cycle
                    cycles.append(
                        {
                            "choice": "right",
                            "start_idx": centre_start_idx,
                            "end_idx": len(sectors) - 1,
                            "arm_entry_idx": i,
                            "complete": False,
                        }
                    )
                choice = "right"

        i += 1

    return cycles
