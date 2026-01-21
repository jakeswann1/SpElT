"""Plotting functions for splitter cell analysis."""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def _get_sector_boundaries_in_bins(
    sectors: list[int],
    rate_map_shape: tuple[int, int],
    pos_header: dict,
    bin_size: float,
) -> tuple[float, float, float, float]:
    """
    Calculate the external boundaries of T-maze sectors in bin coordinates.

    Parameters
    ----------
    sectors : list of int
        Sector numbers (1-12) to get boundaries for
    rate_map_shape : tuple
        Shape of the rate map (y_bins, x_bins)
    pos_header : dict
        Position header with spatial boundaries
    bin_size : float
        Spatial bin size in cm

    Returns
    -------
    x_min, x_max, y_min, y_max : float
        Boundaries in bin coordinates (0-indexed)
    """
    # Get field of view dimensions
    scaled_ppm = pos_header.get("scaled_ppm", 400)
    min_x = pos_header.get("min_x", 0)
    max_x = pos_header.get("max_x")
    min_y = pos_header.get("min_y", 0)
    max_y = pos_header.get("max_y")

    # Convert bin size to pixels
    bin_length = bin_size * scaled_ppm / 100  # cm -> m -> pixels

    # T-maze layout: 4 columns x 3 rows
    num_cols = 4
    num_rows = 3

    sector_width = (max_x - min_x) / num_cols
    sector_height = (max_y - min_y) / num_rows

    # Convert sector numbers to row/col positions
    # Sectors: Row 1 (1-4), Row 2 (5-8), Row 3 (9-12)
    sector_rows = [(s - 1) // num_cols + 1 for s in sectors]
    sector_cols = [(s - 1) % num_cols + 1 for s in sectors]

    # Find min/max row and column
    min_row = min(sector_rows)
    max_row = max(sector_rows)
    min_col = min(sector_cols)
    max_col = max(sector_cols)

    # Convert to spatial coordinates (pixels)
    spatial_x_min = min_x + (min_col - 1) * sector_width
    spatial_x_max = min_x + max_col * sector_width
    spatial_y_min = min_y + (min_row - 1) * sector_height
    spatial_y_max = min_y + max_row * sector_height

    # Convert to bin coordinates
    bin_x_min = (spatial_x_min - min_x) / bin_length
    bin_x_max = (spatial_x_max - min_x) / bin_length
    bin_y_min = (spatial_y_min - min_y) / bin_length
    bin_y_max = (spatial_y_max - min_y) / bin_length

    return bin_x_min, bin_x_max, bin_y_min, bin_y_max


def plot_splitter_maps(
    unit_id: int,
    left_rate_map: np.ndarray,
    right_rate_map: np.ndarray,
    correlation: float | None = None,
    left_max_rate: float | None = None,
    right_max_rate: float | None = None,
    session_name: str | None = None,
    animal: str | None = None,
    age: int | None = None,
    n_left_trajectories: int | None = None,
    n_right_trajectories: int | None = None,
    is_significant_splitter: bool | None = None,
    n_significant_bins: int | None = None,
    significant_bins_mask: np.ndarray | None = None,
    correlation_sectors: list[int] | None = None,
    pos_header: dict | None = None,
    bin_size: float = 2.5,
    left_rate_profile_1d: np.ndarray | None = None,
    right_rate_profile_1d: np.ndarray | None = None,
    p_values_per_bin: np.ndarray | None = None,
    figsize: tuple = (15, 8),
    cmap: str = "jet",
    save_path: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot side-by-side rate maps for left and right choice trajectories.

    Creates a figure with four panels: left-choice rate map, right-choice
    rate map, their difference map, and 1D rate profiles with significance.

    Parameters
    ----------
    unit_id : int
        Unit/cluster ID being plotted
    left_rate_map : np.ndarray
        2D rate map for left-choice trajectories
    right_rate_map : np.ndarray
        2D rate map for right-choice trajectories
    correlation : float, optional
        Spatial correlation between left and right maps
    left_max_rate : float, optional
        Maximum firing rate in left map
    right_max_rate : float, optional
        Maximum firing rate in right map
    session_name : str, optional
        Session identifier for title
    animal : str, optional
        Animal ID for title
    age : int, optional
        Animal age for title
    n_left_trajectories : int, optional
        Number of left trajectories used
    n_right_trajectories : int, optional
        Number of right trajectories used
    is_significant_splitter : bool, optional
        Whether this unit is a significant splitter cell
    n_significant_bins : int, optional
        Number of significant X positions
    significant_bins_mask : np.ndarray, optional
        1D boolean mask of significant X bins (from correlation sectors).
        Used to highlight significant X positions on 2D difference map
        and 1D profile plot.
    correlation_sectors : list of int, optional
        Sector numbers used for correlation analysis (e.g., [6, 7])
    pos_header : dict, optional
        Position header with spatial boundaries
        (required if correlation_sectors provided)
    bin_size : float, optional
        Spatial bin size in cm (default: 2.5)
    left_rate_profile_1d : np.ndarray, optional
        1D firing rate profile for left trajectories (Hz vs X position)
    right_rate_profile_1d : np.ndarray, optional
        1D firing rate profile for right trajectories (Hz vs X position)
    p_values_per_bin : np.ndarray, optional
        P-values for each X bin from shuffle test (same length as 1D profiles)
    figsize : tuple, optional
        Figure size (width, height) in inches
    cmap : str, optional
        Colormap name (default: 'jet')
    save_path : str, optional
        If provided, save figure to this path

    Returns
    -------
    fig : plt.Figure
        Figure object
    axes : np.ndarray
        Array of axis objects

    Examples
    --------
    >>> fig, axes = plot_splitter_maps(
    ...     unit_id=42,
    ...     left_rate_map=left_map,
    ...     right_rate_map=right_map,
    ...     correlation=0.35,
    ...     session_name='r1364_230613',
    ...     left_rate_profile_1d=left_profile,
    ...     right_rate_profile_1d=right_profile
    ... )
    """
    fig = plt.figure(figsize=figsize)

    # Determine if we need the 1D panel
    show_1d_panel = (
        left_rate_profile_1d is not None and right_rate_profile_1d is not None
    )

    if show_1d_panel:
        # Two rows: top row for 2D maps, bottom row for 1D profile
        gs = GridSpec(
            2,
            3,
            figure=fig,
            height_ratios=[2, 1],
            hspace=0.25,
            wspace=0.3,
            left=0.04,
            right=0.96,
            top=0.90,
            bottom=0.06,
        )
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        ax_1d = fig.add_subplot(gs[1, :])  # 1D panel spans all columns
        axes.append(ax_1d)
    else:
        # Original layout: single row with 3 panels
        gs = GridSpec(
            1, 3, figure=fig, wspace=0.3, left=0.04, right=0.96, top=0.88, bottom=0.04
        )
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # Determine common scale for left and right maps (use max across both)
    vmax = max(np.nanmax(left_rate_map), np.nanmax(right_rate_map))

    # Plot left-choice map
    _ = axes[0].imshow(
        left_rate_map, cmap=cmap, origin="lower", vmin=0, vmax=vmax, aspect="equal"
    )
    axes[0].set_title("Left-choice trajectories", fontsize=11, fontweight="bold", pad=5)
    axes[0].axis("off")

    # Plot right-choice map
    im2 = axes[1].imshow(
        right_rate_map, cmap=cmap, origin="lower", vmin=0, vmax=vmax, aspect="equal"
    )
    axes[1].set_title(
        "Right-choice trajectories", fontsize=11, fontweight="bold", pad=5
    )
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.031, pad=0.04, label="Hz")

    # Plot difference map (left - right)
    diff_map = left_rate_map - right_rate_map
    vmax_diff = max(abs(np.nanmin(diff_map)), abs(np.nanmax(diff_map)))
    im3 = axes[2].imshow(
        diff_map,
        cmap="RdBu_r",
        origin="lower",
        vmin=-vmax_diff,
        vmax=vmax_diff,
        aspect="equal",
    )
    axes[2].set_title("Difference (L - R)", fontsize=11, fontweight="bold", pad=5)
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2], fraction=0.031, pad=0.04, label="Hz")

    # Overlay sector boundaries on all three plots (if provided)
    if correlation_sectors is not None and pos_header is not None:
        try:
            x_min, x_max, y_min, y_max = _get_sector_boundaries_in_bins(
                correlation_sectors, left_rate_map.shape, pos_header, bin_size
            )

            # Draw grey rectangle on all three plots showing correlation sectors
            for ax in axes[:3]:  # Only draw on the three 2D maps
                rect = patches.Rectangle(
                    (x_min - 0.5, y_min - 0.5),  # -0.5 for bin edge alignment
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=2,
                    edgecolor="grey",
                    facecolor="none",
                    linestyle="--",
                    alpha=0.8,
                    zorder=10,
                )
                ax.add_patch(rect)
        except Exception as e:
            import warnings

            warnings.warn(f"Failed to draw sector boundaries: {e}", stacklevel=2)

    # Highlight significant X columns on difference map
    if significant_bins_mask is not None and np.any(significant_bins_mask):
        # Handle both 1D and 2D masks
        if significant_bins_mask.ndim == 1:
            # 1D mask: significant X bins from correlation sectors
            # Get sector boundaries to determine which X bins to highlight
            if correlation_sectors is not None and pos_header is not None:
                try:
                    x_min_sector, x_max_sector, y_min, y_max = (
                        _get_sector_boundaries_in_bins(
                            correlation_sectors,
                            left_rate_map.shape,
                            pos_header,
                            bin_size,
                        )
                    )

                    # Draw vertical bars for significant X positions within sector range
                    for i, is_sig in enumerate(significant_bins_mask):
                        if is_sig:
                            # Map from sector-relative index to absolute X position
                            x_idx = int(x_min_sector) + i
                            rect = patches.Rectangle(
                                (x_idx - 0.5, y_min - 0.5),
                                1.0,  # Width of one bin
                                y_max - y_min,  # Height of sector region
                                linewidth=0,
                                edgecolor="none",
                                facecolor="orange",
                                alpha=0.5,
                                zorder=9,
                            )
                            axes[2].add_patch(rect)
                except Exception as e:
                    import warnings

                    warnings.warn(
                        f"Failed to highlight significant bins: {e}", stacklevel=2
                    )

        elif significant_bins_mask.ndim == 2:
            # 2D mask: direct bin-wise significance
            height, width = significant_bins_mask.shape
            significant_x_columns = np.any(significant_bins_mask, axis=0)

            # Get sector boundaries if available
            try:
                if correlation_sectors is not None and pos_header is not None:
                    x_min, x_max, y_min, y_max = _get_sector_boundaries_in_bins(
                        correlation_sectors,
                        significant_bins_mask.shape,
                        pos_header,
                        bin_size,
                    )
                else:
                    x_min, x_max, y_min, y_max = -0.5, width - 0.5, -0.5, height - 0.5
            except Exception:
                x_min, x_max, y_min, y_max = -0.5, width - 0.5, -0.5, height - 0.5

            # Draw vertical bars for each significant X position
            for x_idx in np.where(significant_x_columns)[0]:
                rect = patches.Rectangle(
                    (x_idx - 0.5, y_min - 0.5),
                    1.0,
                    y_max - y_min,
                    linewidth=0,
                    edgecolor="none",
                    facecolor="orange",
                    alpha=0.5,
                    zorder=9,
                )
                axes[2].add_patch(rect)
    title_parts = [f"Unit {unit_id}"]
    if session_name:
        title_parts.append(f"Session {session_name}")
    if animal:
        title_parts.append(f"Animal {animal}")
    if age is not None:
        title_parts.append(f"P{age}")

    title = " | ".join(title_parts)

    # Build subtitle with stats
    subtitle_parts = []
    if left_max_rate is not None and right_max_rate is not None:
        subtitle_parts.append(
            f"Max FR: {left_max_rate:.2f} Hz (L), {right_max_rate:.2f} Hz (R)"
        )
    if correlation is not None:
        subtitle_parts.append(f"Correlation: r = {correlation:.3f}")
    if n_left_trajectories is not None and n_right_trajectories is not None:
        subtitle_parts.append(
            f"N = {n_left_trajectories} (L), {n_right_trajectories} (R)"
        )
    if is_significant_splitter is not None:
        if is_significant_splitter:
            sig_text = (
                f"SIGNIFICANT SPLITTER "
                f"({n_significant_bins} consecutive bins, p < 0.05)"
            )
            subtitle_parts.append(sig_text)
        else:
            subtitle_parts.append("Non-significant")

    subtitle = " | ".join(subtitle_parts)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.97)
    if subtitle:
        if show_1d_panel:
            fig.text(0.5, 0.93, subtitle, ha="center", fontsize=10)
        else:
            fig.text(0.5, 0.91, subtitle, ha="center", fontsize=10)

    # Plot 1D profiles if available (sector-specific only)
    if show_1d_panel:
        ax_1d = axes[3]

        # Create x-axis (bin positions within sector)
        n_bins = len(left_rate_profile_1d)
        x_bins = np.arange(n_bins)

        # Plot left and right profiles (pre-smoothed from significance testing)
        ax_1d.plot(
            x_bins,
            left_rate_profile_1d,
            "b-",
            linewidth=2,
            label="Left trajectories",
            alpha=0.8,
        )
        ax_1d.plot(
            x_bins,
            right_rate_profile_1d,
            "r-",
            linewidth=2,
            label="Right trajectories",
            alpha=0.8,
        )

        # Highlight significant bins with background shading
        if significant_bins_mask is not None:
            for i, is_sig in enumerate(significant_bins_mask):
                if is_sig and i < n_bins:
                    ax_1d.axvspan(i - 0.5, i + 0.5, color="orange", alpha=0.3, zorder=0)

        # Set labels and formatting
        sector_label = (
            f" (sectors {correlation_sectors})" if correlation_sectors else ""
        )
        ax_1d.set_xlabel(f"Position along center stem{sector_label} (bin)", fontsize=11)
        ax_1d.set_ylabel("Firing rate (Hz)", fontsize=11)
        ax_1d.set_xlim(-0.5, n_bins - 0.5)
        ax_1d.legend(loc="upper left", fontsize=9)
        ax_1d.grid(True, alpha=0.3)
        ax_1d.spines["top"].set_visible(False)
        ax_1d.spines["right"].set_visible(False)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, axes


def plot_splitter_population(
    unit_ids: list[int],
    left_rate_maps: dict[int, np.ndarray],
    right_rate_maps: dict[int, np.ndarray],
    correlations: dict[int, float] | None = None,
    max_units_per_figure: int = 12,
    ncols: int = 4,
    figsize_per_unit: tuple = (3, 3),
    cmap: str = "jet",
    session_name: str | None = None,
    save_prefix: str | None = None,
) -> list[tuple[plt.Figure, np.ndarray]]:
    """
    Plot rate maps for multiple units in a grid layout.

    Creates figure(s) with units arranged in a grid, showing left and right
    rate maps side-by-side for each unit.

    Parameters
    ----------
    unit_ids : list of int
        List of unit IDs to plot
    left_rate_maps : dict
        {unit_id: left_rate_map}
    right_rate_maps : dict
        {unit_id: right_rate_map}
    correlations : dict, optional
        {unit_id: correlation_value}
    max_units_per_figure : int, optional
        Maximum units per figure (default: 12)
    ncols : int, optional
        Number of columns in grid (default: 4)
    figsize_per_unit : tuple, optional
        Figure size per unit (width, height) in inches
    cmap : str, optional
        Colormap name
    session_name : str, optional
        Session name for title
    save_prefix : str, optional
        If provided, save figures with this prefix + figure number

    Returns
    -------
    figures : list of (fig, axes) tuples
        List of created figures and their axes

    Examples
    --------
    >>> figs = plot_splitter_population(
    ...     unit_ids=[1, 2, 3],
    ...     left_rate_maps=left_maps,
    ...     right_rate_maps=right_maps,
    ...     correlations=corrs,
    ...     session_name='r1364_230613'
    ... )
    """
    figures = []
    n_units = len(unit_ids)
    n_figures = int(np.ceil(n_units / max_units_per_figure))

    for fig_idx in range(n_figures):
        start_idx = fig_idx * max_units_per_figure
        end_idx = min((fig_idx + 1) * max_units_per_figure, n_units)
        units_in_fig = unit_ids[start_idx:end_idx]

        nrows = int(np.ceil(len(units_in_fig) / ncols))
        figsize = (ncols * figsize_per_unit[0] * 2, nrows * figsize_per_unit[1])

        fig, axes = plt.subplots(
            nrows, ncols * 2, figsize=figsize, squeeze=False
        )  # *2 for L+R

        for idx, unit_id in enumerate(units_in_fig):
            row = idx // ncols
            col_pair = (idx % ncols) * 2  # Each unit gets 2 columns

            if unit_id not in left_rate_maps or unit_id not in right_rate_maps:
                continue

            left_map = left_rate_maps[unit_id]
            right_map = right_rate_maps[unit_id]

            # Common scale
            vmax = max(np.nanmax(left_map), np.nanmax(right_map))

            # Plot left
            axes[row, col_pair].imshow(
                left_map, cmap=cmap, origin="lower", vmin=0, vmax=vmax, aspect="auto"
            )
            axes[row, col_pair].axis("off")

            # Plot right
            axes[row, col_pair + 1].imshow(
                right_map, cmap=cmap, origin="lower", vmin=0, vmax=vmax, aspect="auto"
            )
            axes[row, col_pair + 1].axis("off")

            # Title with unit ID and correlation
            title = f"Unit {unit_id}"
            if correlations and unit_id in correlations:
                title += f"\nr = {correlations[unit_id]:.3f}"
            axes[row, col_pair].set_title(title, fontsize=9)
            axes[row, col_pair + 1].set_title("", fontsize=9)

        # Hide unused subplots
        for idx in range(len(units_in_fig), nrows * ncols):
            row = idx // ncols
            col_pair = (idx % ncols) * 2
            axes[row, col_pair].axis("off")
            axes[row, col_pair + 1].axis("off")

        # Overall title
        title = "Splitter Cell Analysis"
        if session_name:
            title += f" - {session_name}"
        if n_figures > 1:
            title += f" (Page {fig_idx + 1}/{n_figures})"
        fig.suptitle(title, fontsize=14, fontweight="bold")

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if save_prefix:
            save_path = f"{save_prefix}_page{fig_idx + 1}.svg"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        figures.append((fig, axes))

    return figures
