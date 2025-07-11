import matplotlib.pyplot as plt
import numpy as np


def visualize_place_fields(
    rate_map: np.ndarray,
    fields: list[tuple[int, int]],
    smoothed_map: np.ndarray,
    spatial_bin_size_cm: float = 1.0,
    metrics: dict | None = None,
    title: str = "Place Field Detection",
    spike_positions: np.ndarray | None = None,
    spike_phases: np.ndarray | None = None,
    field_phase_correlations: list | None = None,
    show_binned_analysis: bool = True,
) -> None:
    """
    Visualize place fields with phase precession.
    Will plot spike position vs phase even if no fields are detected.

    Args:
        rate_map: 1D firing rate map
        fields: List of (start_bin, end_bin) tuples for detected fields
        smoothed_map: Smoothed version of rate map
        spatial_bin_size_cm: Size of spatial bins in cm
        metrics: Dictionary containing field detection metrics
        title: Plot title
        spike_positions: Array of spike positions
        spike_phases: Array of spike phases
        field_phase_correlations: List of correlation results for each field
        show_binned_analysis: Whether to show binned circular means and bin edges
    """
    positions = np.arange(len(rate_map)) * spatial_bin_size_cm

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot rate maps
    ax1.plot(positions, rate_map, "k-", alpha=0.5, label="Original Rate")
    ax1.plot(positions, smoothed_map, "b-", linewidth=2, label="Smoothed Rate")

    # Highlight fields if they exist
    if fields:
        for i, (start, end) in enumerate(fields):
            field_start_idx = max(0, start)
            field_end_idx = min(len(positions) - 1, end)

            if field_start_idx <= field_end_idx:
                field_positions_plot = positions[field_start_idx : field_end_idx + 1]
                field_rates_plot = smoothed_map[field_start_idx : field_end_idx + 1]
                ax1.fill_between(
                    field_positions_plot,
                    0,
                    field_rates_plot,
                    alpha=0.3,
                    color=f"C{i}",
                    label=f"Field {i+1}" if i == 0 else f"_Field {i+1}",
                )

            if metrics and i < len(metrics.get("metrics", [])):
                metric_data = metrics["metrics"][i]
                if "peak_idx" in metric_data and metric_data["peak_idx"] < len(
                    positions
                ):
                    peak_idx = metric_data["peak_idx"]
                    peak_rate = metric_data["peak_rate"]
                    ax1.plot(positions[peak_idx], peak_rate, "ro", markersize=6)

    ax1.set_xlabel("Position (cm)")
    ax1.set_ylabel("Firing Rate (Hz)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_title(title)

    # Create phase axis if spike data is available, regardless of whether fields exist
    if (
        spike_positions is not None
        and spike_phases is not None
        and len(spike_positions) > 0
    ):
        ax2 = ax1.twinx()

        # Plot all spike phases (showing 0-4pi range like MATLAB)
        # This will happen regardless of whether fields exist
        ax2.scatter(
            spike_positions,
            spike_phases % (2 * np.pi),
            s=20,
            c="k",
            alpha=0.6,
            marker=".",
            label="Individual Spikes",
        )
        ax2.scatter(
            spike_positions,
            (spike_phases % (2 * np.pi)) + 2 * np.pi,
            s=20,
            c="k",
            alpha=0.6,
            marker=".",
        )

        # Normalized positions and phases processing - only if fields exist
        if fields:
            normalized_positions = {}
            normalized_phases = {}

            for i, (start, end) in enumerate(fields):
                # Get field boundaries in position units
                field_min_cm = positions[start] if start < len(positions) else 0
                field_max_cm = positions[end] if end < len(positions) else positions[-1]

                # Find spikes within this field
                field_mask = (spike_positions >= field_min_cm) & (
                    spike_positions <= field_max_cm
                )
                if np.sum(field_mask) > 0:
                    field_spike_pos = spike_positions[field_mask]
                    field_spike_phases = spike_phases[field_mask]

                    # Normalize positions to 0-1 range as in MATLAB code
                    norm_pos = (field_spike_pos - field_min_cm) / (
                        field_max_cm - field_min_cm
                    )

                    normalized_positions[i] = norm_pos
                    normalized_phases[i] = field_spike_phases

                    # Show binned analysis if requested
                    if (
                        show_binned_analysis
                        and field_phase_correlations
                        and i < len(field_phase_correlations)
                    ):
                        if (
                            field_phase_correlations[i] is not None
                            and "bin_data" in field_phase_correlations[i]
                        ):
                            _add_regression_bin_visualization(
                                ax2,
                                field_phase_correlations[i]["bin_data"],
                                field_min_cm,
                                field_max_cm,
                                i,
                            )

            # Mark field boundaries with colored dashed lines (like MATLAB)
            for start_bin, end_bin in fields:
                if start_bin < len(positions):
                    ax1.axvline(
                        x=positions[start_bin], color="c", linestyle="--", alpha=0.5
                    )
                if end_bin < len(positions):
                    ax1.axvline(
                        x=positions[end_bin], color="m", linestyle="--", alpha=0.5
                    )

            # Plot phase precession regression lines with proper wrapping handling
            if field_phase_correlations:
                _plot_phase_regression_lines_fixed(
                    ax2, field_phase_correlations, fields, positions
                )

        # Set y-axis limits and ticks - do this regardless of fields
        ax2.set_ylim(0, 4 * np.pi)  # MATLAB uses 0 to 4π
        ax2.set_yticks(
            [
                0,
                np.pi / 2,
                np.pi,
                3 * np.pi / 2,
                2 * np.pi,
                5 * np.pi / 2,
                3 * np.pi,
                7 * np.pi / 2,
                4 * np.pi,
            ]
        )
        ax2.set_yticklabels(
            [
                "0",
                "$\\pi/2$",
                "$\\pi$",
                "$3\\pi/2$",
                "$2\\pi$",
                "$5\\pi/2$",
                "$3\\pi$",
                "$7\\pi/2$",
                "$4\\pi$",
            ]
        )
        ax2.set_ylabel("Spike Phase (radians)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

    # Create legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    visible_handles1 = [
        handle for label, handle in zip(labels1, handles1) if not label.startswith("_")
    ]
    visible_labels1 = [label for label in labels1 if not label.startswith("_")]

    # Add phase axis legend if it exists
    legend_handles = visible_handles1
    legend_labels = visible_labels1

    if "ax2" in locals():
        handles2, labels2 = ax2.get_legend_handles_labels()
        visible_handles2 = [
            handle
            for label, handle in zip(labels2, handles2)
            if not label.startswith("_")
        ]
        visible_labels2 = [label for label in labels2 if not label.startswith("_")]
        legend_handles.extend(visible_handles2)
        legend_labels.extend(visible_labels2)

    if legend_handles:
        ax1.legend(legend_handles, legend_labels, loc="upper right")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show()

    return fig, ax1, ax2 if "ax2" in locals() else None


def _plot_phase_regression_lines_fixed(
    ax2, field_phase_correlations, fields, positions
):
    """
    FIXED: Properly plot phase precession regression lines handling wrapping
    """
    for i, (start_bin, end_bin) in enumerate(fields):
        if (
            start_bin >= len(positions)
            or end_bin >= len(positions)
            or start_bin > end_bin
            or i >= len(field_phase_correlations)
            or field_phase_correlations[i] is None
        ):
            continue

        corr_data = field_phase_correlations[i]
        slope = corr_data["slope"]
        phi0 = corr_data["phi0"]
        p_val = corr_data.get("p_val", 1.0)

        # Get bin data for proper unwrapping
        bin_data = corr_data.get("bin_data", {})
        bin_centers = bin_data.get("bin_centers", [])
        bin_means = bin_data.get("bin_means", [])

        field_min_cm = positions[start_bin]
        field_max_cm = positions[end_bin]

        # Determine line style based on significance
        if p_val < 0.05:
            line_style = "-"  # Green solid line for significant
            line_color = "green"
            line_label = f"Regression Field {i+1} (p<0.05)" if i == 0 else None
        else:
            line_style = "--"  # Red dashed line for non-significant
            line_color = "red"
            line_label = f"Regression Field {i+1} (n.s.)" if i == 0 else None

        # FIXED METHOD: Use the actual bin data to determine proper unwrapping
        if len(bin_centers) >= 3 and len(bin_means) >= 3:
            # Method 1: Use the bin means to guide the unwrapping
            try:
                # Unwrap the actual bin means to get the true continuous progression
                unwrapped_bin_means = np.unwrap(np.array(bin_means))

                # Create high-resolution line for smooth plotting
                x_fine = np.linspace(field_min_cm, field_max_cm, 200)
                x_fine_norm = (x_fine - field_min_cm) / (field_max_cm - field_min_cm)

                # Generate line in unwrapped space using the regression parameters
                # The slope and phi0 from the regression already account for unwrapping
                line_phases_unwrapped = slope * x_fine_norm + phi0

                # Key insight: The regression was fit to properly unwrapped data,
                # so we need to maintain that continuity when plotting

                # Check if we have a wrapping case by looking at the range
                bin_phase_range = np.ptp(bin_means)
                unwrapped_range = np.ptp(unwrapped_bin_means)

                if bin_phase_range > 1.5 * np.pi and unwrapped_range > bin_phase_range:
                    # This is a wrapping case - the regression was fit to unwrapped data
                    # We need to use the unwrapped bin means to determine the offset

                    # Find the offset between our regression line and the unwrapped bins
                    bin_x_norm = (np.array(bin_centers) - field_min_cm) / (
                        field_max_cm - field_min_cm
                    )
                    regression_at_bins = slope * bin_x_norm + phi0

                    # Calculate the offset needed to align with unwrapped data
                    # Use the first bin as reference point
                    offset = unwrapped_bin_means[0] - regression_at_bins[0]

                    # Apply this offset to the display line
                    display_line = line_phases_unwrapped + offset

                else:
                    # Normal case - use the original regression line
                    display_line = line_phases_unwrapped

                # Plot in both phase ranges (0-2π and 2π-4π)
                for phase_offset in [0, 2 * np.pi]:
                    # Map to this display range
                    display_phases = (display_line % (2 * np.pi)) + phase_offset

                    # Always split into continuous segments to avoid vertical lines
                    segments = _split_wrapped_line(
                        x_fine, display_phases, max_jump=np.pi
                    )

                    for j, (seg_x, seg_y) in enumerate(segments):
                        if len(seg_x) >= 2:  # Only plot segments with multiple points
                            label = (
                                line_label if (phase_offset == 0 and j == 0) else None
                            )
                            ax2.plot(
                                seg_x,
                                seg_y,
                                line_style,
                                linewidth=2,
                                color=line_color,
                                alpha=0.8,
                                label=label,
                            )

            except Exception as e:
                print(f"Warning: Could not plot regression line for field {i+1}: {e}")
                # Fallback to simple method
                _plot_simple_regression_line(
                    ax2,
                    slope,
                    phi0,
                    field_min_cm,
                    field_max_cm,
                    line_style,
                    line_color,
                    line_label,
                )
        else:
            # Fallback for cases without sufficient bin data
            _plot_simple_regression_line(
                ax2,
                slope,
                phi0,
                field_min_cm,
                field_max_cm,
                line_style,
                line_color,
                line_label,
            )


def _split_wrapped_line(x, y, max_jump=np.pi):
    """
    Split x,y data into continuous segments where y doesn't jump by more than max_jump
    """
    if len(x) < 2:
        return [(x, y)]

    segments = []
    current_x = [x[0]]
    current_y = [y[0]]

    for i in range(1, len(x)):
        # Check if there's a large jump in y (indicating wrapping)
        y_diff = abs(y[i] - y[i - 1])

        # Detect both upward and downward wrapping
        is_wrap_jump = (y_diff > max_jump) or (y_diff > 1.5 * np.pi)

        if is_wrap_jump:
            # End current segment if it has enough points
            if len(current_x) >= 2:  # Reduced minimum for better segment capture
                segments.append((np.array(current_x), np.array(current_y)))
            # Start new segment
            current_x = [x[i]]
            current_y = [y[i]]
        else:
            # Continue current segment
            current_x.append(x[i])
            current_y.append(y[i])

    # Add final segment if it has enough points
    if len(current_x) >= 2:
        segments.append((np.array(current_x), np.array(current_y)))

    return segments


def _plot_simple_regression_line(
    ax2, slope, phi0, field_min_cm, field_max_cm, line_style, line_color, line_label
):
    """
    Fallback simple regression line plotting
    """
    x_for_line = np.linspace(field_min_cm, field_max_cm, 100)
    x_norm = (x_for_line - field_min_cm) / (field_max_cm - field_min_cm)
    line_phases = slope * x_norm + phi0

    # Plot in both ranges
    for phase_offset in [0, 2 * np.pi]:
        wrapped_phases = (line_phases % (2 * np.pi)) + phase_offset
        label = line_label if phase_offset == 0 else None
        ax2.plot(
            x_for_line,
            wrapped_phases,
            line_style,
            linewidth=2,
            color=line_color,
            alpha=0.8,
            label=label,
        )


def _add_regression_bin_visualization(
    ax2: plt.Axes,
    bin_data: dict,
    field_min_cm: float,
    field_max_cm: float,
    field_idx: int,
) -> None:
    """
    Add visualization of the actual binned data used for regression.

    Args:
        ax2: Phase axis to plot on
        bin_data: Dictionary containing bin data from binned_cl_corr
        field_min_cm: Field minimum position in cm (for context)
        field_max_cm: Field maximum position in cm (for context)
        field_idx: Index of the field (for coloring)
    """
    # Extract the actual data used for regression
    bin_centers = bin_data.get("bin_centers", [])
    bin_means = bin_data.get("bin_means", [])
    bin_counts = bin_data.get("bin_counts", [])
    bin_edges = bin_data.get("bin_edges", [])

    if len(bin_centers) < 3:
        return  # Not enough valid bins

    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)
    bin_counts = np.array(bin_counts)

    # Choose colors that contrast with existing elements
    bin_colors = ["orange", "purple", "brown", "pink", "olive"]
    color = bin_colors[field_idx % len(bin_colors)]

    # Plot bin edges as vertical lines (only the ones that were actually used)
    for edge in bin_edges:
        if field_min_cm <= edge <= field_max_cm:  # Only show edges within the field
            ax2.axvline(x=edge, color=color, linestyle=":", alpha=0.4, linewidth=1)

    # Plot circular means for each valid bin (the actual regression data points)
    # Scale marker size based on spike count in bin
    min_count = np.min(bin_counts)
    marker_sizes = 50 + (bin_counts - min_count) * 10  # Base size + scaling

    # Plot bin means in both phase ranges (0-2π and 2π-4π)
    for offset in [0, 2 * np.pi]:
        # Adjust phases to be in the correct range
        plot_phases = bin_means % (2 * np.pi) + offset

        ax2.scatter(
            bin_centers,
            plot_phases,
            s=marker_sizes,
            c=color,
            marker="o",
            alpha=0.8,
            edgecolors="black",
            linewidth=1,
            label=f"Regression Data Field {field_idx+1}" if offset == 0 else None,
            zorder=5,  # Plot on top of other elements
        )

    # Add text annotation for this field's binning info
    text_y = 3.5 * np.pi - field_idx * 0.3 * np.pi  # Stagger text for multiple fields
    text_x = field_min_cm + 0.05 * (field_max_cm - field_min_cm)

    n_bins_used = bin_data.get("n_bins_used", len(bin_centers))
    n_bins_requested = bin_data.get("n_bins_requested", 10)

    ax2.text(
        text_x,
        text_y,
        f"Field {field_idx+1}: {n_bins_used}/{n_bins_requested} bins",
        fontsize=9,
        color=color,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.7},
    )
