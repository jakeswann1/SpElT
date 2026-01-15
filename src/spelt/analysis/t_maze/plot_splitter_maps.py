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
    figsize: tuple = (12, 5),
    cmap: str = "jet",
    save_path: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot side-by-side rate maps for left and right choice trajectories.

    Creates a figure with three panels: left-choice rate map, right-choice
    rate map, and their difference map.

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
        Number of significant X positions (collapsed across Y, for display)
    significant_bins_mask : np.ndarray, optional
        Boolean mask of significant bins (same shape as rate maps).
        Marks all bins in significant X columns within dual occupancy.
        Significant X columns are visualized as semi-transparent yellow
        vertical bars on the difference map.
    correlation_sectors : list of int, optional
        Sector numbers used for correlation analysis (e.g., [6, 7])
    pos_header : dict, optional
        Position header with spatial boundaries
        (required if correlation_sectors provided)
    bin_size : float, optional
        Spatial bin size in cm (default: 2.5)
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
    ...     session_name='r1364_230613'
    ... )
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        1, 3, figure=fig, wspace=0.4, left=0.05, right=0.95, top=0.85, bottom=0.05
    )

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # Determine common scale for left and right maps (use max across both)
    vmax = max(np.nanmax(left_rate_map), np.nanmax(right_rate_map))

    # Plot left-choice map
    im1 = axes[0].imshow(
        left_rate_map, cmap=cmap, origin="lower", vmin=0, vmax=vmax, aspect="equal"
    )
    axes[0].set_title("Left-choice trajectories", fontsize=11, fontweight="bold", pad=5)
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label="Hz")

    # Plot right-choice map
    im2 = axes[1].imshow(
        right_rate_map, cmap=cmap, origin="lower", vmin=0, vmax=vmax, aspect="equal"
    )
    axes[1].set_title(
        "Right-choice trajectories", fontsize=11, fontweight="bold", pad=5
    )
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label="Hz")

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
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label="Hz")

    # Overlay sector boundaries on all three plots (if provided)
    if correlation_sectors is not None and pos_header is not None:
        try:
            x_min, x_max, y_min, y_max = _get_sector_boundaries_in_bins(
                correlation_sectors, left_rate_map.shape, pos_header, bin_size
            )

            # Draw grey rectangle on all three plots showing correlation sectors
            for ax in axes:
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
        # Get the shape of the rate map
        height, width = significant_bins_mask.shape

        # Find which X columns are significant (have any True values)
        significant_x_columns = np.any(significant_bins_mask, axis=0)

        # Get sector boundaries if available (for constraining highlight)
        try:
            if correlation_sectors is not None and pos_header is not None:
                x_min, x_max, y_min, y_max = _get_sector_boundaries_in_bins(
                    correlation_sectors,
                    significant_bins_mask.shape,
                    pos_header,
                    bin_size,
                )
            else:
                # Use full extent
                x_min, x_max, y_min, y_max = -0.5, width - 0.5, -0.5, height - 0.5
        except Exception:
            # Fallback to full extent
            x_min, x_max, y_min, y_max = -0.5, width - 0.5, -0.5, height - 0.5

        # Draw vertical highlight bars for each significant X position
        for x_idx in np.where(significant_x_columns)[0]:
            # Draw a semi-transparent yellow/gold rectangle spanning the Y extent
            # Constrained by sector boundaries
            rect = patches.Rectangle(
                (x_idx - 0.5, y_min - 0.5),  # -0.5 for bin edge alignment
                1.0,  # Width of one bin
                y_max - y_min,  # Height of sector region
                linewidth=0,
                edgecolor="none",
                facecolor="orange",
                alpha=0.5,
                zorder=9,  # Below sector boundary box (zorder=10)
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
                f"SIGNIFICANT SPLITTER ({n_significant_bins} X positions, p < 0.05)"
            )
            subtitle_parts.append(sig_text)
        else:
            subtitle_parts.append("Non-significant")

    subtitle = " | ".join(subtitle_parts)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.96)
    if subtitle:
        fig.text(0.5, 0.90, subtitle, ha="center", fontsize=10)

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
            save_path = f"{save_prefix}_page{fig_idx + 1}.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        figures.append((fig, axes))

    return figures
