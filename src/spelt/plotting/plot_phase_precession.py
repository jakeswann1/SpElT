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
) -> None:
    """
    Visualize place fields with phase precession.
    Will plot spike position vs phase even if no fields are detected.
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

            # Plot phase precession regression lines
            if field_phase_correlations:
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
                    p_val = corr_data.get(
                        "p_val", 1.0
                    )  # Default to 1.0 if not provided

                    field_min_cm = positions[start_bin]
                    field_max_cm = positions[end_bin]

                    # Use normalized positions (0-1) for the regression line
                    x_for_line = np.array([field_min_cm, field_max_cm])

                    # Calculate regression phase values
                    line_phases = (
                        slope
                        * (x_for_line - field_min_cm)
                        / (field_max_cm - field_min_cm)
                        + phi0
                    )

                    # Plot with MATLAB-style color and line style based on significance
                    if p_val < 0.05:
                        line_style = "g-"  # Green solid line for significant
                    else:
                        line_style = "r--"  # Red dashed line for non-significant

                    # Plot multiple copies offset by 2π, similar to MATLAB approach
                    for offset in [0, 2 * np.pi, 4 * np.pi]:
                        ax2.plot(
                            x_for_line, line_phases + offset, line_style, linewidth=2
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

    if visible_handles1:
        ax1.legend(visible_handles1, visible_labels1, loc="upper right")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show()

    return fig, ax1, ax2 if "ax2" in locals() else None
