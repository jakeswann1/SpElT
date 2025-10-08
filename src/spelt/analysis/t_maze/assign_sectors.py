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
