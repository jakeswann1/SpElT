import numpy as np


def apply_common_reference(
    lfp_data: np.ndarray, method: str = "median", axis: int = 1
) -> np.ndarray:
    """
    Apply common reference to multi-channel LFP data.

    Computes a reference signal from all channels and subtracts it from
    each channel. Common referencing removes shared noise and artifacts
    while preserving local signals.

    Parameters
    ----------
    lfp_data : np.ndarray
        Multi-channel LFP data, shape (n_samples, n_channels)
    method : str, optional
        Reference method: "median" or "mean", by default "median"
        - "median": More robust to outliers and artifacts
        - "mean": Standard average reference
    axis : int, optional
        Axis along which to compute reference (default: 1 for channels)

    Returns
    -------
    np.ndarray
        Referenced LFP data with same shape as input

    Raises
    ------
    ValueError
        If method is not "median" or "mean"

    Notes
    -----
    Common average reference (CAR) is a standard preprocessing step in multi-channel
    neural recordings. It assumes that the reference signal (shared noise) affects
    all channels equally and can be estimated by averaging across channels.

    The median method is generally preferred for neural recordings as it is more
    robust to artifacts or bad channels that may have extreme values.

    Examples
    --------
    >>> # Apply median reference
    >>> lfp_referenced = apply_common_reference(lfp_data, method="median")
    >>>
    >>> # Apply mean reference
    >>> lfp_referenced = apply_common_reference(lfp_data, method="mean")
    """
    if method not in ["median", "mean"]:
        raise ValueError(f"method must be 'median' or 'mean', got '{method}'")

    # Compute reference signal
    if method == "median":
        reference = np.median(lfp_data, axis=axis, keepdims=True)
    else:  # method == "mean"
        reference = np.mean(lfp_data, axis=axis, keepdims=True)

    # Subtract reference from each channel
    lfp_referenced = lfp_data - reference

    return lfp_referenced


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
