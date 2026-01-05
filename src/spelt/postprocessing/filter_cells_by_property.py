from typing import Any

import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.postprocessing as spost

from spelt.postprocessing.burst_index_and_autocorrelograms import compute_first_moment


def _passes_range_filter(value: float, value_range: tuple[float, float] | None) -> bool:
    """
    Check if a value passes a range filter.

    Parameters
    ----------
    value : float
        The value to check
    value_range : tuple[float, float] | None
        Range as (min, max), or None to skip filtering

    Returns
    -------
    bool
        True if value passes filter or filter is None, False otherwise
    """
    if value_range is None or np.isnan(value):
        return True
    return value_range[0] <= value <= value_range[1]


def _passes_threshold_filter(
    value: float, threshold: float | None, comparison: str = "<="
) -> bool:
    """
    Check if a value passes a threshold filter.

    Parameters
    ----------
    value : float
        The value to check
    threshold : float | None
        Threshold value, or None to skip filtering
    comparison : str
        Comparison operator: "<=", ">=", "<", ">", "==", "!="

    Returns
    -------
    bool
        True if value passes filter or filter is None, False otherwise
    """
    if threshold is None or np.isnan(value):
        return True

    if comparison == "<=":
        return value <= threshold
    elif comparison == ">=":
        return value >= threshold
    elif comparison == "<":
        return value < threshold
    elif comparison == ">":
        return value > threshold
    elif comparison == "==":
        return value == threshold
    elif comparison == "!=":
        return value != threshold
    else:
        raise ValueError(f"Unknown comparison operator: {comparison}")


def filter_cells_by_property(
    analyzer: si.SortingAnalyzer,
    depth_range: tuple[float, float] | None = None,
    fr_range: tuple[float, float] | None = None,
    spike_width_range_us: tuple[float, float] | None = None,
    halfwidth_range_us: tuple[float, float] | None = None,
    repolarization_slope_range: tuple[float, float] | None = None,
    peak_trough_ratio_range: tuple[float, float] | None = None,
    burst_params: tuple[float, float, float] | None = None,
    return_dataframe: bool = True,
) -> tuple[Any, list] | tuple[Any, list, pd.DataFrame]:
    """
    Filter cells based on various properties and return the filtered analyzer object.

    Parameters
    ----------
    analyzer : si.SortingAnalyzer
        The analyzer object with sorting data
    depth_range : tuple[float, float] | None, optional
        Range of depth in micrometers as (min, max)
    fr_range : tuple[float, float] | None, optional
        Range of firing rate in Hz as (min, max)
    spike_width_range_us : tuple[float, float] | None, optional
        Range of spike width in microseconds as (min, max)
    halfwidth_range_us : tuple[float, float] | None, optional
        Range of halfwidth in microseconds as (min, max)
    repolarization_slope_range : tuple[float, float] | None, optional
        Range of repolarization slope in mV/s as (min, max)
    peak_trough_ratio_range : tuple[float, float] | None, optional
        Range of peak-to-trough ratio as (min, max)
    burst_params : tuple[float, float, float] | None, optional
        Bursting parameters as (window_ms, bin_ms, threshold_ms)
    return_dataframe : bool, optional
        Whether to return a DataFrame with all metrics (default: True)

    Returns
    -------
    analyzer : si.SortingAnalyzer
        The analyzer object with filtered units selected
    units_to_keep : list
        List of unit IDs that were kept after filtering
    metrics_df : pd.DataFrame (only if return_dataframe=True)
        DataFrame with columns:
        - unit_id: Unit identifier
        - depth: Recording depth (um)
        - firing_rate: Firing rate (Hz)
        - spike_width_us: Spike width in microseconds (peak_to_valley)
        - halfwidth_us: Halfwidth at 50% amplitude in microseconds
        - repolarization_slope: Repolarization slope (mV/s)
        - peak_trough_ratio: Ratio between negative and positive peaks
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

    # Calculate template metrics for ALL units
    try:
        template_metrics_df: pd.DataFrame = spost.compute_template_metrics(
            analyzer,
            metric_names=[
                "peak_to_valley",
                "half_width",
                "repolarization_slope",
                "peak_trough_ratio",
            ],
            metrics_kwargs={"peak_relative_threshold": 0, "peak_width_ms": 0},
        )
        spike_widths_s = template_metrics_df["peak_to_valley"]
        halfwidths_s = template_metrics_df["half_width"]
        repolarization_slopes_s = template_metrics_df["repolarization_slope"]
        peak_trough_ratios_s = template_metrics_df["peak_trough_ratio"]
    except Exception:
        spike_widths_s = pd.Series(np.nan, index=all_units)
        halfwidths_s = pd.Series(np.nan, index=all_units)
        repolarization_slopes_s = pd.Series(np.nan, index=all_units)
        peak_trough_ratios_s = pd.Series(np.nan, index=all_units)

    # Calculate burst indices for ALL units if burst_params specified
    burst_indices = {}
    burst_threshold_ms = None
    if burst_params is not None:
        window_ms, bin_ms, burst_threshold_ms = burst_params

        # Calculate autocorrelograms
        correlograms, bin_edges = spost.compute_correlograms(
            analyzer, window_ms=window_ms, bin_ms=bin_ms
        )

        # Calculate bin centers (spost.compute_correlograms returns ms)
        # Convert to seconds for compute_first_moment
        bin_centers_ms = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_centers = bin_centers_ms / 1000  # Convert to seconds

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
        halfwidth_us = (
            halfwidths_s.loc[unit] * 1e6 if unit in halfwidths_s.index else np.nan
        )
        repolarization_slope = (
            repolarization_slopes_s.loc[unit]
            if unit in repolarization_slopes_s.index
            else np.nan
        )
        peak_trough_ratio = (
            peak_trough_ratios_s.loc[unit]
            if unit in peak_trough_ratios_s.index
            else np.nan
        )
        burst_index = (
            burst_indices.get(unit, np.nan) if burst_params is not None else np.nan
        )

        # Check if unit passes all filters using helper functions
        passed = True
        passed &= _passes_range_filter(depth, depth_range)
        passed &= _passes_range_filter(fr, fr_range)
        passed &= _passes_range_filter(spike_width_us, spike_width_range_us)
        passed &= _passes_range_filter(halfwidth_us, halfwidth_range_us)
        passed &= _passes_range_filter(repolarization_slope, repolarization_slope_range)
        passed &= _passes_range_filter(peak_trough_ratio, peak_trough_ratio_range)
        passed &= _passes_threshold_filter(burst_index, burst_threshold_ms, "<=")

        metrics_data.append(
            {
                "unit_id": unit,
                "depth": depth,
                "firing_rate": fr,
                "spike_width_us": spike_width_us,
                "halfwidth_us": halfwidth_us,
                "repolarization_slope": repolarization_slope,
                "peak_trough_ratio": peak_trough_ratio,
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
