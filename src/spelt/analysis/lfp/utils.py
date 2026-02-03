import numpy as np


def find_evenly_spaced_channels(channel_depths, target_spacing, tolerance=2.0):
    """
    Find longest subsequence of channels with target spacing.

    Uses dynamic programming to find the longest chain where each
    selected channel is target_spacing away from the previous one.

    Parameters
    ----------
    channel_depths : np.ndarray
        Array of channel depths (in μm)
    target_spacing : float
        Target spacing in μm
    tolerance : float
        Acceptable deviation from target spacing (default: 2.0 μm)

    Returns
    -------
    dict
        Selected channels forming longest evenly-spaced subsequence
    """
    # Sort channels by depth
    sorted_idx = np.argsort(channel_depths)
    sorted_depths = channel_depths[sorted_idx]
    n = len(sorted_depths)

    # Dynamic programming approach
    # dp[i] = (length of longest sequence ending at i, previous index)
    dp = [(1, -1) for _ in range(n)]

    for i in range(1, n):
        for j in range(i):
            spacing = sorted_depths[i] - sorted_depths[j]

            # Check if spacing matches target
            if abs(spacing - target_spacing) <= tolerance:
                # Can extend sequence from j to i
                new_length = dp[j][0] + 1
                if new_length > dp[i][0]:
                    dp[i] = (new_length, j)

    # Find the longest sequence
    max_length = max(length for length, _ in dp)
    end_idx = next(i for i, (length, _) in enumerate(dp) if length == max_length)

    # Backtrack to get the full sequence
    sequence = []
    idx = end_idx
    while idx != -1:
        sequence.append(idx)
        idx = dp[idx][1]

    sequence.reverse()

    # Convert to original indices
    selected_sorted_idx = np.array(sequence)
    selected_original_idx = sorted_idx[selected_sorted_idx]
    selected_depths = sorted_depths[selected_sorted_idx]

    # Compute actual spacing
    if len(selected_depths) > 1:
        actual_spacings = np.diff(selected_depths)
        mean_spacing = np.mean(actual_spacings)
        spacing_std = np.std(actual_spacings)
    else:
        mean_spacing = None
        spacing_std = None

    return {
        "indices": selected_original_idx,
        "depths": selected_depths,
        "n_channels": len(selected_original_idx),
        "spacing": mean_spacing,
        "spacing_std": spacing_std,
        "depth_range": (
            (selected_depths[0], selected_depths[-1])
            if len(selected_depths) > 0
            else None
        ),
    }
