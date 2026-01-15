import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import QhullError
from scipy.stats import pearsonr


def interpolate_map(map, max_dim):
    """
    Interpolates a given map to a new set of dimensions.

    Parameters:
    map (ndarray): The input map to be interpolated.
    max_dim (tuple): The maximum dimensions of the interpolated map.

    Returns:
    ndarray: The interpolated map.

    Notes:
    If interpolation fails (e.g., due to collinear data points), returns
    a map filled with NaN values.
    """

    # If map is already the target size, return it directly
    if map.shape == (max_dim[0], max_dim[1]):
        return map.copy()

    x = np.arange(map.shape[1])
    y = np.arange(map.shape[0])
    xx, yy = np.meshgrid(x, y)

    # Get valid entries
    valid_entries = ~np.isnan(map)
    coords = np.array((xx[valid_entries], yy[valid_entries])).T
    values = map[valid_entries]

    # Check if we have enough non-collinear points for interpolation
    if len(values) < 3:
        # Not enough points for interpolation
        return np.full((max_dim[0], max_dim[1]), np.nan)

    # Create grid for new dimensions
    grid_x, grid_y = np.mgrid[0 : max_dim[1], 0 : max_dim[0]]

    # Interpolate using griddata
    try:
        interpolated_map = griddata(
            coords, values, (grid_x, grid_y), method="linear", fill_value=np.nan
        )
    except QhullError:
        # Interpolation failed (likely due to collinear points)
        # Return NaN-filled map
        interpolated_map = np.full((max_dim[0], max_dim[1]), np.nan)

    return interpolated_map


def pearson_corr(map1, map2):
    """Calculate Pearson correlation for two flattened maps."""
    valid_mask = ~np.isnan(map1) & ~np.isnan(map2)
    if np.any(valid_mask):
        return pearsonr(map1[valid_mask], map2[valid_mask])[0]
    else:
        return np.nan


def spatial_correlation(map_list1, map_list2=None):
    """
    Calculate spatial correlation between 2D rate maps, interpolating to common size.

    The common size is determined as the maximum size among all maps.

    Parameters:
    - map_list1: A list of 2D numpy arrays representing rate maps.
    - map_list2: (Optional) Another list of 2D numpy arrays representing
                 rate maps.

    Returns:
    - A matrix of Pearson correlation coefficients if only map_list1 is
      provided.
    - An array of Pearson correlation coefficients for corresponding maps if
      map_list1 and map_list2 are provided.
    """

    # Determine the maximum dimensions among all maps
    all_maps = map_list1 + (map_list2 if map_list2 else [])
    max_dim = max(
        [(m.shape[0], m.shape[1]) for m in all_maps], key=lambda x: x[0] * x[1]
    )

    # Interpolate maps to the maximum dimensions
    interpolated_maps1 = [interpolate_map(m, max_dim) for m in map_list1]
    interpolated_maps2 = (
        [interpolate_map(m, max_dim) for m in map_list2] if map_list2 else None
    )

    if interpolated_maps2 is None:
        # Calculate correlation matrix for maps in a single list
        n_maps = len(interpolated_maps1)
        corr_matrix = np.full((n_maps, n_maps), np.nan)

        for i in range(n_maps):
            for j in range(i, n_maps):
                corr = pearson_corr(
                    interpolated_maps1[i].flatten(), interpolated_maps1[j].flatten()
                )
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        return corr_matrix
    else:
        # Calculate correlation for corresponding maps in two lists
        n_maps = min(len(interpolated_maps1), len(interpolated_maps2))
        corr_array = np.full(n_maps, np.nan)

        for i in range(n_maps):
            corr_array[i] = pearson_corr(
                interpolated_maps1[i].flatten(), interpolated_maps2[i].flatten()
            )

        return corr_array
