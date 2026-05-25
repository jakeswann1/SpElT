"""Linearise figure-of-eight T-maze position to a 1D coordinate per loop direction.

The maze is a continuous figure-of-eight with the layout:

    Top row (left arm):    [1*REWARD] [2]  [3]  [4]
    Middle row (stem):     [5]        [6]  [7]  [8*START]
    Bottom row (right arm):[9*REWARD] [10] [11] [12]

A LEFT loop traverses sectors 8 → 7 → 6 → 5 → 1 → 2 → 3 → 4 → 8.
A RIGHT loop traverses sectors 8 → 7 → 6 → 5 → 9 → 10 → 11 → 12 → 8.

Within both stem and arm corridors, X is monotonic along the rat's path:
- Stem (middle row): rat moves leftward, so X decreases as the rat progresses.
- Arms (top/bottom row): rat moves rightward, so X increases as the rat progresses.

We can therefore build a single continuous 1D coordinate per direction by:
- Stem segment:  coord_bin = (n_x_bins - 1) - x_bin
- Arm segment:   coord_bin = n_x_bins + x_bin

The brief Y-only return at the right edge (sector 4 → 8 or 12 → 8) puts the rat
back into the stem with high X, mapping to coord ≈ 0 — the same physical location
as the loop start. Speed-filtering the spikes upstream eliminates any double-
counting of immobility-related activity at this transition.

The output is a 1D bin index suitable for use with
`spelt.analysis.t_maze.collapse_position_to_1d.make_1d_rate_maps`.
"""

import numpy as np

from spelt.analysis.t_maze.assign_sectors import bin_indices_to_sectors

LEFT_DIRECTION = "left"
RIGHT_DIRECTION = "right"

STEM_SECTORS = {5, 6, 7, 8}
LEFT_ARM_SECTORS = {1, 2, 3, 4}
RIGHT_ARM_SECTORS = {9, 10, 11, 12}


def linearise_figure8_position(
    pos_bin_idx: tuple[np.ndarray, np.ndarray],
    pos_sample_times: np.ndarray,
    pos_sampling_rate: float,
    direction: str,
    pos_header: dict,
    bin_size_cm: float = 2.5,
    n_x_bins: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Linearise figure-of-eight T-maze position to a 1D bin index.

    Parameters
    ----------
    pos_bin_idx : tuple of (x_bins, y_bins)
        Position bin indices, as returned by `bin_pos_data_dlc`.
    pos_sample_times : np.ndarray
        Timestamp for each position sample.
    pos_sampling_rate : float
        Position sampling rate in Hz.
    direction : {'left', 'right'}
        Which loop to linearise.
    pos_header : dict
        Position header (`obj.pos_data[trial]['header']`) — must contain
        `min_x`, `max_x`, `min_y`, `max_y`, and `scaled_ppm`.
    bin_size_cm : float, default 2.5
        Spatial bin size in cm. Must match the value used by `bin_pos_data_dlc`.
    n_x_bins : int, optional
        Total number of X bins in the original 2D binning. If None, inferred
        from the data as `max(x_bins) + 1`. Pass explicitly to keep the
        linearised loop length consistent across sessions / trials where the
        rat doesn't reach the full FoV extent.

    Returns
    -------
    linearised_bin_idx : np.ndarray
        1D bin index along the figure-of-eight loop. NaN for poses outside the
        chosen direction's loop. Range: [0, total_loop_bins - 1].
    pos_sample_times : np.ndarray
        Unchanged timestamps (returned for API consistency with
        `collapse_position_bins_to_x`).
    pos_sampling_rate : float
        Unchanged sampling rate.
    total_loop_bins : int
        Total number of bins in the linearised loop (2 * n_x_bins).
    """
    if direction not in (LEFT_DIRECTION, RIGHT_DIRECTION):
        raise ValueError(f"direction must be 'left' or 'right', got {direction!r}")

    x_bins, y_bins = pos_bin_idx
    x_bins = np.asarray(x_bins)
    y_bins = np.asarray(y_bins)

    if x_bins.shape != y_bins.shape:
        raise ValueError(
            f"x_bins and y_bins shape mismatch: {x_bins.shape} vs {y_bins.shape}"
        )

    if len(x_bins) == 0:
        return np.array([]), pos_sample_times, pos_sampling_rate, 0

    if n_x_bins is None:
        n_x_bins = int(np.nanmax(x_bins)) + 1
    total_loop_bins = 2 * n_x_bins

    sectors = bin_indices_to_sectors(
        x_bins.astype(float), y_bins.astype(float), pos_header, bin_size_cm
    )

    arm_sectors = LEFT_ARM_SECTORS if direction == LEFT_DIRECTION else RIGHT_ARM_SECTORS

    is_stem = np.isin(sectors, list(STEM_SECTORS))
    is_arm = np.isin(sectors, list(arm_sectors))

    linearised = np.full(len(x_bins), np.nan, dtype=float)
    linearised[is_stem] = (n_x_bins - 1) - x_bins[is_stem]
    linearised[is_arm] = n_x_bins + x_bins[is_arm]

    return linearised, pos_sample_times, pos_sampling_rate, total_loop_bins


def figure8_loop_length_cm(total_loop_bins: int, bin_size_cm: float = 2.5) -> float:
    """Total linearised loop length in cm (n_bins * bin_size_cm)."""
    return total_loop_bins * bin_size_cm
