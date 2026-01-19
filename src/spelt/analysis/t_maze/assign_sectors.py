import numpy as np
import pandas as pd


def assign_sectors(xy_positions, pos_header=None):
    """
    Assign sectors to given xy_positions based on a grid layout.

    Parameters:
    - xy_positions (DataFrame): DataFrame containing x and y coordinates.
    - pos_header (dict): position header dictionary from ephys object

    Returns:
    - sector_numbers (ndarray): Array containing sector numbers for each coordinate.
    """

    # Validate input
    if not isinstance(xy_positions, pd.DataFrame):
        raise TypeError("Input xy_positions must be a pandas DataFrame.")

    if xy_positions.shape[1] != 2:
        xy_positions = xy_positions.T
        if xy_positions.shape[1] != 2:
            raise ValueError("Invalid input dimensions.")

    # Drop rows where either x or y is NaN only for min/max calculation
    xy_positions_filtered = xy_positions.dropna(
        subset=[xy_positions.columns[0], xy_positions.columns[1]]
    )

    # Check if DataFrame is empty after removing NaNs
    if xy_positions_filtered.empty:
        raise ValueError("Input DataFrame is empty after removing NaN values.")

    # Calculate the minimum and maximum x, y coordinates to determine the field of view
    # These coordinates are rounded to the lower and upper bin edges, respectively
    if pos_header is not None:
        min_x = pos_header["min_x"]
        max_x = pos_header["max_x"]
        min_y = pos_header["min_y"]
        max_y = pos_header["max_y"]
    else:
        min_x = np.floor(xy_positions_filtered.iloc[:, 0].min())
        max_x = np.ceil(xy_positions_filtered.iloc[:, 0].max())
        min_y = np.floor(xy_positions_filtered.iloc[:, 1].min())
        max_y = np.ceil(xy_positions_filtered.iloc[:, 1].max())
        print("Position header not provided. Using min/max values from xy_positions.")

    # Define grid dimensions
    num_cols = 4
    num_rows = 3

    # Calculate sector dimensions
    sector_width = (max_x - min_x) / num_cols
    sector_height = (max_y - min_y) / num_rows

    # Handle NaN values explicitly by assigning them to a specific sector or NaN
    col_indices = (
        np.floor((xy_positions.iloc[:, 0] - min_x) / sector_width).astype(float) + 1
    )
    row_indices = (
        np.floor((xy_positions.iloc[:, 1] - min_y) / sector_height).astype(float) + 1
    )

    # Replace NaNs in col_indices and row_indices with a valid index or leave them NaN
    col_indices = np.clip(col_indices, 1, num_cols)
    row_indices = np.clip(row_indices, 1, num_rows)

    # Calculate sector numbers
    sector_numbers = (row_indices - 1) * num_cols + col_indices

    # Convert to integers, handling NaNs by keeping them as NaNs
    sector_numbers = sector_numbers.where(~sector_numbers.isna(), np.nan)

    return sector_numbers.values


def bin_indices_to_sectors(
    x_bins: np.ndarray, y_bins: np.ndarray, pos_header: dict, bin_size: float = 2.5
) -> np.ndarray:
    """
    Convert position bin indices to sector numbers.

    Takes spatial bin indices and converts them to T-maze sector numbers (1-12)
    using the same grid layout as assign_sectors().

    Parameters
    ----------
    x_bins : np.ndarray
        X bin indices (samples,)
    y_bins : np.ndarray
        Y bin indices (samples,)
    pos_header : dict
        Position header with min/max x/y boundaries and scaled_ppm
    bin_size : float
        Spatial bin size in cm (default: 2.5)

    Returns
    -------
    sector_numbers : np.ndarray
        Sector numbers (1-12) for each sample, same shape as x_bins
        NaN values preserved for invalid positions

    Notes
    -----
    - Uses 4x3 grid layout matching assign_sectors()
    - Sectors numbered 1-12: Row 1 (1-4), Row 2 (5-8), Row 3 (9-12)
    - Converts bin indices → spatial coordinates → sector numbers
    """
    # Get spatial parameters
    scaled_ppm = pos_header.get("scaled_ppm", 400)
    min_x = pos_header["min_x"]
    max_x = pos_header["max_x"]
    min_y = pos_header["min_y"]
    max_y = pos_header["max_y"]

    # Convert bin size to pixels
    bin_length = bin_size * scaled_ppm / 100  # cm -> m -> pixels

    # Convert bin indices to spatial coordinates (pixel centers)
    x_coords = min_x + x_bins * bin_length + bin_length / 2
    y_coords = min_y + y_bins * bin_length + bin_length / 2

    # Grid dimensions
    num_cols = 4
    num_rows = 3

    # Calculate sector dimensions
    sector_width = (max_x - min_x) / num_cols
    sector_height = (max_y - min_y) / num_rows

    # Calculate column and row indices (1-indexed)
    col_indices = np.floor((x_coords - min_x) / sector_width).astype(float) + 1
    row_indices = np.floor((y_coords - min_y) / sector_height).astype(float) + 1

    # Clip to valid range
    col_indices = np.clip(col_indices, 1, num_cols)
    row_indices = np.clip(row_indices, 1, num_rows)

    # Calculate sector numbers: (row - 1) * cols + col
    sector_numbers = (row_indices - 1) * num_cols + col_indices

    # Preserve NaN values
    sector_numbers = np.where(
        np.isnan(x_bins) | np.isnan(y_bins), np.nan, sector_numbers
    )

    return sector_numbers


def filter_position_by_sectors(
    pos_bin_idx: tuple[np.ndarray, np.ndarray],
    pos_sample_times: np.ndarray,
    pos_sampling_rate: float,
    sectors: list[int],
    pos_header: dict,
    bin_size: float = 2.5,
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, float]:
    """
    Filter position samples to only those within specified sectors.

    Takes position bin data and returns only samples where the animal
    was located in one of the specified T-maze sectors. This is used
    to restrict splitter cell analysis to specific regions (e.g., the
    center stem choice point).

    Parameters
    ----------
    pos_bin_idx : tuple of (x_bins, y_bins)
        Position bin indices for each sample
    pos_sample_times : np.ndarray
        Timestamps for each position sample
    pos_sampling_rate : float
        Position sampling rate in Hz
    sectors : list of int
        List of sector numbers to include (e.g., [6, 7])
    pos_header : dict
        Position header with spatial boundaries
    bin_size : float
        Spatial bin size in cm (default: 2.5)

    Returns
    -------
    filtered_pos_bin_idx : tuple of (x_bins, y_bins)
        Position bin indices only for samples in specified sectors
    filtered_pos_sample_times : np.ndarray
        Timestamps only for samples in specified sectors
    pos_sampling_rate : float
        Unchanged sampling rate (returned for consistency)

    Notes
    -----
    - Uses bin_indices_to_sectors() to determine which samples to keep
    - Maintains temporal order of samples
    - Empty arrays returned if no samples in specified sectors

    Examples
    --------
    >>> # Filter to only center stem (sectors 6, 7)
    >>> pos_bins_filtered, times_filtered, rate = filter_position_by_sectors(
    ...     pos_bin_idx=(x_bins, y_bins),
    ...     pos_sample_times=times,
    ...     pos_sampling_rate=50.0,
    ...     sectors=[6, 7],
    ...     pos_header=header,
    ...     bin_size=2.5
    ... )
    """
    x_bins, y_bins = pos_bin_idx

    # Handle empty input
    if len(x_bins) == 0:
        return (np.array([]), np.array([])), np.array([]), pos_sampling_rate

    # Convert bin indices to sector numbers
    sector_numbers = bin_indices_to_sectors(x_bins, y_bins, pos_header, bin_size)

    # Create mask for samples in target sectors
    # Use isin to handle multiple sectors efficiently
    in_target_sectors = np.isin(sector_numbers, sectors)

    # Also exclude NaN positions
    valid_mask = in_target_sectors & ~np.isnan(sector_numbers)

    # Filter all arrays
    filtered_x_bins = x_bins[valid_mask]
    filtered_y_bins = y_bins[valid_mask]
    filtered_times = pos_sample_times[valid_mask]

    return ((filtered_x_bins, filtered_y_bins), filtered_times, pos_sampling_rate)
