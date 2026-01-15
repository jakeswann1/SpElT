"""Functions for masking rate maps by T-maze sectors."""

import numpy as np


def mask_rate_map_by_sectors(
    rate_map: np.ndarray,
    pos_header: dict,
    sectors_to_keep: list[int],
    bin_size: float = 2.5,
) -> np.ndarray:
    """
    Mask a rate map to only include specified T-maze sectors.

    Sets bins outside the specified sectors to NaN, allowing for
    sector-specific analyses (e.g., correlation in center stem only).

    Parameters
    ----------
    rate_map : np.ndarray
        2D array of firing rates (shape: y_bins x x_bins)
    pos_header : dict
        Position header with min/max x/y boundaries:
        - 'min_x', 'max_x', 'min_y', 'max_y': FOV boundaries
        - 'scaled_ppm': pixels per meter (for bin size conversion)
    sectors_to_keep : list of int
        Sector numbers to keep (1-12). All other bins set to NaN.
        Sectors: 1-4 (left arm), 5-8 (center stem), 9-12 (right arm)
    bin_size : float, optional
        Spatial bin size in cm (default: 2.5)

    Returns
    -------
    masked_rate_map : np.ndarray
        Rate map with bins outside specified sectors set to NaN

    Examples
    --------
    >>> # Keep only center stem (sectors 5-8)
    >>> masked_map = mask_rate_map_by_sectors(
    ...     rate_map, pos_header, sectors_to_keep=[5, 6, 7, 8]
    ... )

    >>> # Keep only choice point (sectors 6, 7)
    >>> masked_map = mask_rate_map_by_sectors(
    ...     rate_map, pos_header, sectors_to_keep=[6, 7]
    ... )

    Notes
    -----
    - T-maze sectors are arranged in a 4x3 grid:
        Row 1 (top):    1  2  3  4  (left arm)
        Row 2 (middle): 5  6  7  8  (center stem)
        Row 3 (bottom): 9 10 11 12  (right arm)
    - The function assumes the rate map uses the same coordinate system
      as the position data
    """
    # Get field of view dimensions
    scaled_ppm = pos_header.get("scaled_ppm", 400)
    min_x = pos_header.get("min_x", 0)
    max_x = pos_header.get("max_x")
    min_y = pos_header.get("min_y", 0)
    max_y = pos_header.get("max_y")

    # Convert bin size to pixels
    bin_length = bin_size * scaled_ppm / 100  # cm -> m -> pixels

    # Calculate sector boundaries (4 columns x 3 rows)
    num_cols = 4
    num_rows = 3

    sector_width = (max_x - min_x) / num_cols
    sector_height = (max_y - min_y) / num_rows

    # Calculate bin edges for the rate map
    y_bins, x_bins = rate_map.shape

    # Create coordinate grids for bin centers
    x_coords = np.arange(x_bins) * bin_length + min_x + bin_length / 2
    y_coords = np.arange(y_bins) * bin_length + min_y + bin_length / 2

    # Create 2D coordinate grids
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # Assign sectors to each bin
    col_indices = np.floor((x_grid - min_x) / sector_width).astype(int) + 1
    row_indices = np.floor((y_grid - min_y) / sector_height).astype(int) + 1

    # Clip to valid range
    col_indices = np.clip(col_indices, 1, num_cols)
    row_indices = np.clip(row_indices, 1, num_rows)

    # Calculate sector numbers (1-12)
    sector_map = (row_indices - 1) * num_cols + col_indices

    # Create mask for bins in specified sectors
    mask = np.isin(sector_map, sectors_to_keep)

    # Apply mask to rate map
    masked_rate_map = rate_map.copy()
    masked_rate_map[~mask] = np.nan

    return masked_rate_map


def mask_rate_maps_by_sectors(
    rate_maps: dict[int, np.ndarray],
    pos_header: dict,
    sectors_to_keep: list[int],
    bin_size: float = 2.5,
) -> dict[int, np.ndarray]:
    """
    Mask multiple rate maps by T-maze sectors.

    Convenience function to apply sector masking to a dictionary of rate maps.

    Parameters
    ----------
    rate_maps : dict
        {unit_id: rate_map} dictionary of 2D rate maps
    pos_header : dict
        Position header with min/max x/y boundaries
    sectors_to_keep : list of int
        Sector numbers to keep (1-12)
    bin_size : float, optional
        Spatial bin size in cm (default: 2.5)

    Returns
    -------
    masked_rate_maps : dict
        {unit_id: masked_rate_map} dictionary with bins outside
        specified sectors set to NaN

    Examples
    --------
    >>> # Mask all units to center stem only
    >>> center_maps = mask_rate_maps_by_sectors(
    ...     all_rate_maps, pos_header, sectors_to_keep=[5, 6, 7, 8]
    ... )
    """
    masked_maps = {}

    for unit_id, rate_map in rate_maps.items():
        masked_maps[unit_id] = mask_rate_map_by_sectors(
            rate_map, pos_header, sectors_to_keep, bin_size
        )

    return masked_maps
