import numpy as np
from scipy.ndimage import convolve


def create_circular_kernel(radius):
    """
    Create a flat, circular kernel.

    Args:
        radius (int): Radius of the circular kernel.

    Returns:
        ndarray: A 2D array representing the circular kernel.
    """
    diameter = 2 * radius + 1
    x, y = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x**2 + y**2 <= radius**2
    kernel = np.zeros((diameter, diameter))
    kernel[mask] = 1
    return kernel


def update_smoothed_maps(smoothed_spk, smoothed_pos, f_spk, f_pos, f_vis, bins_passed):
    """
    Update the smoothed spike and position maps.

    Args:
        smoothed_spk (ndarray): Smoothed spike map to be updated.
        smoothed_pos (ndarray): Smoothed position map to be updated.
        f_spk (ndarray): Filtered spike map.
        f_pos (ndarray): Filtered position map.
        f_vis (ndarray): Filtered visited template.
        bins_passed (ndarray): Boolean array indicating which bins passed the adaptive smoothing criteria.
    """
    smoothed_spk[bins_passed] = f_spk[bins_passed] / f_vis[bins_passed]
    smoothed_pos[bins_passed] = f_pos[bins_passed] / f_vis[bins_passed]


def finalize_smoothed_rate_map(smoothed_spk, smoothed_pos, pos_map):
    """
    Finalize the smoothed rate map.

    Args:
        smoothed_spk (ndarray): Smoothed spike map.
        smoothed_pos (ndarray): Smoothed position map.
        pos_map (ndarray): Original position map.

    Returns:
        ndarray: Smoothed rate map.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        smoothed_rate = smoothed_spk / smoothed_pos
    smoothed_rate[pos_map == 0] = np.nan
    return smoothed_rate


def handle_empty_maps(spk_map, pos_map):
    """
    Handle the case when the spike map is empty.

    Args:
        spk_map (ndarray): Spike map.
        pos_map (ndarray): Position map.

    Returns:
        tuple: Tuple containing NaN-filled arrays for smoothed spike, position, and rate maps,
               and NaN for the median radius.
    """
    nan_map = np.full_like(pos_map, np.nan)
    return nan_map, nan_map, nan_map, np.nan


def adaptive_smooth(spk_map, pos_map, alpha, max_radius=None):
    """
    Apply adaptive smoothing to rate maps using a flat, circular kernel.
    Built to match the logic of scan(pix).maps.adaptiveSmooth.m by Thomas Wills

    Args:
        spk_map (ndarray): 2D array representing the spike map.
        pos_map (ndarray): 2D array representing the position map.
        alpha (float): Alpha parameter for adaptive smoothing.
        max_radius (int, optional): Maximum allowed radius for the smoothing kernel.
                                    Defaults to half of the smallest map dimension.

    Returns:
        smoothed_spk (ndarray): Smoothed spike map.
        smoothed_pos (ndarray): Smoothed position map.
        smoothed_rate (ndarray): Smoothed rate map.
        median_radius (float): Median radius used in the smoothing process.
    """
    if max_radius is None:
        max_radius = min(spk_map.shape) // 2

    # Check for empty spike map
    if np.sum(spk_map) == 0:
        return handle_empty_maps(spk_map, pos_map)

    # Initializations
    smoothed_spk = np.zeros_like(spk_map)
    smoothed_pos = np.zeros_like(pos_map)
    visited_template = (pos_map > 0).astype(int)
    smoothed_check = pos_map == 0  # True for unvisited bins
    radii_used_list = []

    # Main adaptive smoothing loop
    radius = 1
    while not np.all(smoothed_check):
        # Check if radius is getting too big
        if radius > max_radius:
            break

        # Construct filter kernel
        kernel = create_circular_kernel(radius)

        # Filter maps to get number of spikes and sum of positions within the kernel
        f_spk = convolve(spk_map, kernel, mode="constant", cval=0.0)
        f_pos = convolve(pos_map, kernel, mode="constant", cval=0.0)
        f_vis = convolve(visited_template, kernel, mode="constant", cval=0.0)

        # Determine which bins meet the criteria at this radius
        with np.errstate(divide="ignore"):
            bins_passed = np.logical_and(
                f_pos != 0, (alpha / (np.sqrt(f_spk) * f_pos)) <= radius
            )
        bins_passed &= ~smoothed_check

        # Update smoothed maps and record bins
        update_smoothed_maps(
            smoothed_spk, smoothed_pos, f_spk, f_pos, f_vis, bins_passed
        )
        smoothed_check[bins_passed] = True

        # Record radii used
        radii_used_list.extend([radius] * np.sum(bins_passed))

        # Increase circle radius
        radius += 1

    # Finalizing smoothed rate map
    smoothed_rate = finalize_smoothed_rate_map(smoothed_spk, smoothed_pos, pos_map)

    # Compute median filter radius
    median_radius = np.nanmedian(radii_used_list)

    return smoothed_spk, smoothed_pos, smoothed_rate, median_radius
