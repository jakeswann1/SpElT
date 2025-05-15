import numpy as np
from scipy import ndimage, signal


def detect_linear_place_fields(
    rate_map: np.ndarray,
    min_peak_height: float = 0.3,  # fraction of max firing rate
    min_width_cm: float = 5.0,  # minimum field width in cm
    max_width_proportion: float = 0.25,  # maximum field width as proportion of track
    smoothing_sigma: float = 2.0,  # smoothing factor for rate map
    spatial_bin_size_cm: float = 1.0,  # size of each bin in cm
    field_threshold: float = 0.2,  # threshold relative to peak height
    peak_distance_bins: int = 10,  # minimum distance between peaks in bins
    prevent_overlaps: bool = True,  # prevent field overlaps
) -> tuple[list[tuple[int, int]], np.ndarray, dict]:
    """
    Detect place field boundaries from a 1D linear rate map

    Args:
        rate_map: 1D numpy array containing firing rates across positions
        min_peak_height: minimum peak height as fraction of maximum rate
        min_width_cm: minimum field width in cm
        max_width_proportion: maximum field width as proportion of track length
        smoothing_sigma: sigma for Gaussian smoothing of rate map
        spatial_bin_size_cm: size of each bin in cm
        field_threshold: threshold relative to peak height for field boundaries
        peak_distance_bins: minimum distance between peaks in bins
        prevent_overlaps: prevent fields from overlapping

    Returns:
        tuple containing:
            - list of (start, end) indices for each detected field
            - smoothed rate map used for detection
            - dictionary with additional metrics about the fields
    """
    # Convert minimum width from cm to bins
    min_width_bins = int(min_width_cm / spatial_bin_size_cm)

    # Calculate maximum width in bins
    max_width_bins = int(max_width_proportion * len(rate_map))

    # Handle empty or invalid input
    if len(rate_map) == 0 or np.all(np.isnan(rate_map)) or np.max(rate_map) == 0:
        return [], np.zeros_like(rate_map), {"num_fields": 0, "metrics": []}

    # Replace NaN values with zeros
    rate_map_clean = np.copy(rate_map)
    rate_map_clean[np.isnan(rate_map_clean)] = 0

    # Apply smoothing
    smoothed_map = ndimage.gaussian_filter1d(rate_map_clean, smoothing_sigma)

    # Find peaks
    max_rate = np.max(smoothed_map)
    peak_height_threshold = max_rate * min_peak_height
    peaks, _ = signal.find_peaks(
        smoothed_map, height=peak_height_threshold, distance=peak_distance_bins
    )

    # Process each peak to find field boundaries
    candidates = []
    for peak_idx in peaks:
        peak_height = smoothed_map[peak_idx]
        field_cutoff = peak_height * field_threshold

        # Find left boundary
        left_idx = peak_idx
        while left_idx > 0 and smoothed_map[left_idx] >= field_cutoff:
            left_idx -= 1

        # Find right boundary
        right_idx = peak_idx
        while (
            right_idx < len(smoothed_map) - 1
            and smoothed_map[right_idx] >= field_cutoff
        ):
            right_idx += 1

        # Check field width constraints
        field_width = right_idx - left_idx
        if min_width_bins <= field_width <= max_width_bins:
            candidates.append(
                {
                    "peak_idx": peak_idx,
                    "left_idx": left_idx,
                    "right_idx": right_idx,
                    "peak_height": peak_height,
                    "width_bins": field_width,
                }
            )

    # Sort candidates by peak height (strongest first)
    candidates.sort(key=lambda x: x["peak_height"], reverse=True)

    # Process candidates with overlap prevention if needed
    field_boundaries = []
    field_metrics = []

    # Track which bins are already occupied if preventing overlaps
    occupied = np.zeros(len(smoothed_map), dtype=bool) if prevent_overlaps else None

    for candidate in candidates:
        peak_idx = candidate["peak_idx"]
        left_idx = candidate["left_idx"]
        right_idx = candidate["right_idx"]

        # Handle overlap prevention if needed
        if prevent_overlaps:
            # Check if this field overlaps with existing fields
            if np.any(occupied[left_idx : right_idx + 1]):
                # Skip if peak is already in another field
                if occupied[peak_idx]:
                    continue

                # Trim field to avoid overlaps
                # Find new left boundary
                new_left_idx = left_idx
                while new_left_idx < peak_idx and occupied[new_left_idx]:
                    new_left_idx += 1

                # Find new right boundary
                new_right_idx = right_idx
                while new_right_idx > peak_idx and occupied[new_right_idx]:
                    new_right_idx -= 1

                # Check if trimmed field still meets width constraints
                field_width = new_right_idx - new_left_idx
                if not (min_width_bins <= field_width <= max_width_bins):
                    continue  # Skip field if width is outside allowed range

                left_idx = new_left_idx
                right_idx = new_right_idx

            # Mark bins as occupied
            occupied[left_idx : right_idx + 1] = True

        # Add the field
        field_boundaries.append((left_idx, right_idx))

        # Calculate field metrics
        field_slice = smoothed_map[left_idx : right_idx + 1]
        field_area = np.sum(field_slice)
        positions = np.arange(left_idx, right_idx + 1)
        center_of_mass = np.sum(positions * field_slice) / field_area
        width_cm = (right_idx - left_idx) * spatial_bin_size_cm
        width_proportion = (right_idx - left_idx) / len(smoothed_map)

        field_metrics.append(
            {
                "peak_idx": peak_idx,
                "peak_rate": candidate["peak_height"],
                "left_idx": left_idx,
                "right_idx": right_idx,
                "width_bins": right_idx - left_idx,
                "width_cm": width_cm,
                "width_proportion": width_proportion,
                "area": field_area,
                "center_of_mass": center_of_mass,
            }
        )

    return (
        field_boundaries,
        smoothed_map,
        {"num_fields": len(field_boundaries), "metrics": field_metrics},
    )


def merge_nearby_fields(
    fields: list[tuple[int, int]],
    smoothed_map: np.ndarray,
    min_separation_bins: int = 5,
    max_width_proportion: float = 0.25,
) -> list[tuple[int, int]]:
    """
    Merge nearby fields that are likely part of the same place field.

    Args:
        fields: List of (start, end) indices for each field
        smoothed_map: Smoothed rate map
        min_separation_bins: Minimum separation between fields
        max_width_proportion: Maximum allowed width as proportion of track length

    Returns:
        List of merged field boundaries
    """
    if len(fields) <= 1:
        return fields

    # Calculate maximum width in bins
    max_width_bins = int(max_width_proportion * len(smoothed_map))

    # Sort fields by position
    sorted_fields = sorted(fields, key=lambda x: x[0])

    merged_fields = []
    current_field = sorted_fields[0]

    for next_field in sorted_fields[1:]:
        current_start, current_end = current_field
        next_start, next_end = next_field

        # Calculate the potential merged field width
        merged_width = next_end - current_start

        # Only consider merging if the resulting field won't be too wide
        if merged_width <= max_width_bins:
            # Check if fields should be merged
            if next_start - current_end <= min_separation_bins:
                # Handle case where fields are adjacent or overlapping
                if next_start <= current_end + 1:
                    # Simply merge the fields without valley check
                    current_field = (current_start, next_end)
                else:
                    # Check depth of valley between fields
                    valley_depth = np.min(smoothed_map[current_end : next_start + 1])
                    peak1 = np.max(smoothed_map[current_start : current_end + 1])
                    peak2 = np.max(smoothed_map[next_start : next_end + 1])
                    min_peak = min(peak1, peak2)

                    # Merge if valley is not deep enough
                    if valley_depth > min_peak * 0.5:
                        current_field = (current_start, next_end)
                    else:
                        merged_fields.append(current_field)
                        current_field = next_field
            else:
                merged_fields.append(current_field)
                current_field = next_field
        else:
            # Don't merge if the field would be too wide
            merged_fields.append(current_field)
            current_field = next_field

    # Add the last field
    merged_fields.append(current_field)

    return merged_fields
