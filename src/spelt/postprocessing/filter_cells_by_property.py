from typing import Any

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.postprocessing as spost


def filter_cells_by_property(
    analyzer: si.SortingAnalyzer,
    depth_range: tuple[float, float] | None = None,
    fr_range: tuple[float, float] | None = None,
    spike_width_range_us: tuple[float, float] | None = None,
    burst_params: tuple[float, float, float] | None = None,
) -> tuple[Any, list]:
    """
    Filter cells based on various properties and return the filtered analyzer object.

    Parameters:
    -----------
    analyzer : Any
        The analyzer object with sorting data
    depth_range : Optional[Tuple[float, float]], optional
        Range of depth in micrometers as (min, max)
    fr_range : Optional[Tuple[float, float]], optional
        Range of firing rate in Hz as (min, max)
    spike_width_threshold : Optional[float], optional
        Minimum spike width in SECONDS (so 500 microseconds = 0.0005)
    burst_params : Optional[Tuple[float, float, float]], optional
        Bursting parameters as (window_ms, bin_ms, threshold_ms)
    include_spike_width : bool, optional
        Whether to include spike width filtering (default: True)

    Returns:
    --------
    analyzer : Any
        The analyzer object with filtered units selected
    units_to_keep : List
        List of unit IDs that were kept after filtering
    """
    sorting: si.BaseSorting = analyzer.sorting
    # Get all unit IDs
    all_units = sorting.get_unit_ids()

    # Apply depth filter if specified
    if depth_range is not None:
        depths = sorting.get_property("depth")
        depth_mask = (depths >= depth_range[0]) & (depths <= depth_range[1])
    else:
        depth_mask = np.ones_like(all_units, dtype=bool)

    # Apply firing rate filter if specified
    if fr_range is not None:
        fr = sorting.get_property("fr")
        fr_mask = (fr >= fr_range[0]) & (fr <= fr_range[1])
    else:
        fr_mask = np.ones_like(all_units, dtype=bool)

    # Combine masks
    mask = depth_mask & fr_mask
    units_to_keep = sorting.get_unit_ids()[mask]

    # Update analyzer with current filter
    analyzer.select_units(units_to_keep)

    # Apply spike width filter if specified and if any units remain
    if spike_width_range_us is not None and len(units_to_keep) > 0:
        # Compute spike templates
        analyzer.compute(
            {"random_spikes": {}, "templates": {"ms_before": 1.5, "ms_after": 2.5}}
        )

        # Calculate spike widths
        widths_df: pd.DataFrame = spost.compute_template_metrics(
            analyzer,
            metric_names=["peak_to_valley"],
            metrics_kwargs={"peak_relative_threshold": 0, "peak_width_ms": 0},
        )

        # Filter by spike width range - convert microseconds to seconds for comparison
        min_width_s = spike_width_range_us[0] / 1e6
        max_width_s = spike_width_range_us[1] / 1e6
        units_to_keep = widths_df[
            (widths_df["peak_to_valley"] >= min_width_s)
            & (widths_df["peak_to_valley"] <= max_width_s)
        ].index
        analyzer.select_units(units_to_keep)

    # Apply burst filter if specified and if any units remain
    if burst_params is not None and len(units_to_keep) > 0:
        window_ms, bin_ms, threshold_ms = burst_params
        first_moment_passing = []

        # Calculate autocorrelograms
        correlograms, bin_edges = spost.compute_correlograms(
            analyzer, window_ms=window_ms, bin_ms=bin_ms
        )

        # Calculate bin centers
        bins = (bin_edges[:-1] + bin_edges[1:]) / 2
        middle_index = len(bins) // 2
        positive_bins = bins[middle_index:]

        for i, unit in enumerate(units_to_keep):
            # Get positive part of autocorrelogram (after 0 ms)
            positive_autocorrelogram = correlograms[i, i, middle_index:]

            # Skip if no spikes in positive bins
            if np.sum(positive_autocorrelogram) == 0:
                continue

            # Calculate the weighted sum of positive bin centers
            weighted_sum = np.sum(positive_bins * positive_autocorrelogram)

            # Calculate the total count in positive bins
            total_positive_count = np.sum(positive_autocorrelogram)

            # Calculate the first moment (mean) of the positive parts
            first_moment = weighted_sum / total_positive_count

            # Keep units with first moment below threshold
            if first_moment < threshold_ms:
                first_moment_passing.append(unit)

        # Update units to keep
        units_to_keep = first_moment_passing
        analyzer.select_units(units_to_keep)

    return analyzer, units_to_keep
