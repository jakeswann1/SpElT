from typing import Any

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.postprocessing as spost

from spelt.postprocessing.burst_index_and_autocorrelograms import compute_first_moment


def filter_cells_by_property(
    analyzer: si.SortingAnalyzer,
    depth_range: tuple[float, float] | None = None,
    fr_range: tuple[float, float] | None = None,
    spike_width_range_us: tuple[float, float] | None = None,
    burst_params: tuple[float, float, float] | None = None,
    return_dataframe: bool = True,
) -> tuple[Any, list] | tuple[Any, list, pd.DataFrame]:
    """
    Filter cells based on various properties and return the filtered analyzer object.

    Parameters:
    -----------
    analyzer : si.SortingAnalyzer
        The analyzer object with sorting data
    depth_range : tuple[float, float] | None, optional
        Range of depth in micrometers as (min, max)
    fr_range : tuple[float, float] | None, optional
        Range of firing rate in Hz as (min, max)
    spike_width_range_us : tuple[float, float] | None, optional
        Range of spike width in microseconds as (min, max)
    burst_params : tuple[float, float, float] | None, optional
        Bursting parameters as (window_ms, bin_ms, threshold_ms)
    return_dataframe : bool, optional
        Whether to return a DataFrame with all metrics (default: True)

    Returns:
    --------
    analyzer : si.SortingAnalyzer
        The analyzer object with filtered units selected
    units_to_keep : list
        List of unit IDs that were kept after filtering
    metrics_df : pd.DataFrame (only if return_dataframe=True)
        DataFrame with columns:
        - unit_id: Unit identifier
        - depth: Recording depth (um)
        - firing_rate: Firing rate (Hz)
        - spike_width_us: Spike width in microseconds
        - burst_index: First moment of autocorrelogram (ms)
        - passed_filter: Boolean indicating if unit passed all filters
    """
    sorting: si.BaseSorting = analyzer.sorting

    # Get all unit IDs
    all_units = sorting.get_unit_ids()

    # Initialize metrics collection
    metrics_data = []

    # Get depth property
    try:
        depths = sorting.get_property("depth")
    except Exception:
        depths = np.full(len(all_units), np.nan)

    # Get firing rate property
    try:
        frs = sorting.get_property("fr")
    except Exception:
        frs = np.full(len(all_units), np.nan)

    # Calculate spike widths for ALL units
    try:
        widths_df: pd.DataFrame = spost.compute_template_metrics(
            analyzer,
            metric_names=["peak_to_valley"],
            metrics_kwargs={"peak_relative_threshold": 0, "peak_width_ms": 0},
        )
        spike_widths_s = widths_df["peak_to_valley"]
    except Exception:
        spike_widths_s = pd.Series(np.nan, index=all_units)

    # Calculate burst indices for ALL units if burst_params specified
    burst_indices = {}
    if burst_params is not None:
        window_ms, bin_ms, threshold_ms = burst_params

        # Calculate autocorrelograms
        correlograms, bin_edges = spost.compute_correlograms(
            analyzer, window_ms=window_ms, bin_ms=bin_ms
        )

        # Calculate bin centers (in seconds)
        # Will be converted to ms in compute_first_moment
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        for i, unit in enumerate(all_units):
            # Get autocorrelogram counts for this unit
            counts = correlograms[i, i, :]

            # Compute first moment using shared function
            first_moment = compute_first_moment(bin_centers, counts)
            burst_indices[unit] = first_moment

    # Build metrics data for each unit
    for i, unit in enumerate(all_units):
        depth = depths[i] if i < len(depths) else np.nan
        fr = frs[i] if i < len(frs) else np.nan
        spike_width_us = (
            spike_widths_s.loc[unit] * 1e6 if unit in spike_widths_s.index else np.nan
        )
        burst_index = (
            burst_indices.get(unit, np.nan) if burst_params is not None else np.nan
        )

        # Check if unit passes all filters
        passed = True

        # Depth filter
        if depth_range is not None and not np.isnan(depth):
            if not (depth_range[0] <= depth <= depth_range[1]):
                passed = False

        # Firing rate filter
        if fr_range is not None and not np.isnan(fr):
            if not (fr_range[0] <= fr <= fr_range[1]):
                passed = False

        # Spike width filter
        if spike_width_range_us is not None and not np.isnan(spike_width_us):
            if not (
                spike_width_range_us[0] <= spike_width_us <= spike_width_range_us[1]
            ):
                passed = False

        # Burst index filter
        if burst_params is not None and not np.isnan(burst_index):
            if burst_index >= threshold_ms:
                passed = False

        metrics_data.append(
            {
                "unit_id": unit,
                "depth": depth,
                "firing_rate": fr,
                "spike_width_us": spike_width_us,
                "burst_index": burst_index,
                "passed_filter": passed,
            }
        )

    # Create DataFrame
    metrics_df = pd.DataFrame(metrics_data)

    # Get units that passed filter
    units_to_keep = metrics_df[metrics_df["passed_filter"]]["unit_id"].tolist()

    # Update analyzer with filtered units
    analyzer.select_units(units_to_keep)

    if return_dataframe:
        return analyzer, units_to_keep, metrics_df
    else:
        return analyzer, units_to_keep
