import glob
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import spikeinterface as si
import spikeinterface.widgets as sw
from matplotlib.backends.backend_pdf import PdfPages

from spelt.utils import compute_extensions_lazy


def plot_unit_summaries(
    analyzer: si.SortingAnalyzer,
    units_to_plot=None,
    output_folder: Path | None = None,
    show: bool = True,
):

    required_extensions = [
        "random_spikes",
        "waveforms",
        "templates",
        "correlograms",
        "spike_amplitudes",
        "unit_locations",
        "template_similarity",
    ]

    analyzer = compute_extensions_lazy(analyzer, required_extensions)

    if units_to_plot is None:
        units_to_plot = analyzer.sorting.get_unit_ids()

    for unit_id in units_to_plot:
        segments = list(range(analyzer.get_num_segments()))
        if si.__version__ >= "0.102.2":
            sw.plot_unit_summary(
                analyzer,
                unit_id,
            )
        else:
            sw.plot_unit_summary(
                analyzer,
                unit_id,
                subwidget_kwargs={
                    "amplitudes": {"segment_indices": segments},
                },
            )

        if output_folder:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                output_folder / f"unit{unit_id}.png", dpi=300, bbox_inches="tight"
            )

        if show:
            plt.show()
        plt.close()


def combine_rate_maps_and_unit_summaries(
    figure_dir: str, session_name: str, output_filename: str | None = None
) -> None:
    """
    Combine rate maps and unit summaries for each unit into a single PDF per session.

    Args:
        figure_dir: Base directory containing the figures
        session_name: Name of the session (e.g., 'P21_session1')
        output_filename: Optional custom output filename
    """
    session_dir = Path(figure_dir) / session_name
    rate_maps_dir = session_dir / "rate_maps"
    unit_summary_dir = session_dir / "unit_summaries"

    # Check if directories exist
    if not rate_maps_dir.exists():
        print(f"No rate_maps directory found for session {session_name}")
        return

    if not unit_summary_dir.exists():
        print(f"No unit_summaries directory found for session {session_name}")
        return

    # Get rate map files
    rate_map_files = glob.glob(str(rate_maps_dir / "*_rate_maps.png"))

    if not rate_map_files:
        print(f"No rate map files found for session {session_name}")
        return

    # Set output filename
    if output_filename is None:
        output_filename = f"{session_name}_combined_summary.pdf"

    output_path = session_dir / output_filename

    # Get all unique unit IDs from rate map files
    unit_ids = []
    for rate_map_file in rate_map_files:
        filename = Path(rate_map_file).stem
        unit_id = filename.replace("_rate_maps", "")
        unit_ids.append(unit_id)

    # Create PDF
    with PdfPages(output_path) as pdf:
        for unit_id in sorted(unit_ids):
            # Paths to the images
            rate_map_path = rate_maps_dir / f"{unit_id}_rate_maps.png"
            unit_summary_path = unit_summary_dir / f"unit{unit_id}.png"

            # Check if both files exist
            if not rate_map_path.exists():
                print(f"Rate map not found for unit {unit_id}")
                continue
            if not unit_summary_path.exists():
                print(f"Unit summary not found for unit {unit_id}")
                continue

            # Create figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

            # Load and display rate map
            try:
                rate_map_img = mpimg.imread(rate_map_path)
                ax1.imshow(rate_map_img)
                ax1.set_title(
                    f"Rate Maps - Unit {unit_id}", fontsize=14, fontweight="bold"
                )
                ax1.axis("off")
            except Exception as e:
                print(f"Error loading rate map for unit {unit_id}: {e}")
                ax1.text(
                    0.5,
                    0.5,
                    "Error loading\nrate map",
                    ha="center",
                    va="center",
                    transform=ax1.transAxes,
                )

            # Load and display unit summary
            try:
                unit_summary_img = mpimg.imread(unit_summary_path)
                ax2.imshow(unit_summary_img)
                ax2.set_title(
                    f"Unit Summary - Unit {unit_id}", fontsize=14, fontweight="bold"
                )
                ax2.axis("off")
            except Exception as e:
                print(f"Error loading unit summary for unit {unit_id}: {e}")
                ax2.text(
                    0.5,
                    0.5,
                    "Error loading\nunit summary",
                    ha="center",
                    va="center",
                    transform=ax2.transAxes,
                )

            # Add overall title
            fig.suptitle(
                f"Session {session_name} - Unit {unit_id}",
                fontsize=16,
                fontweight="bold",
            )

            plt.tight_layout()

            # Save to PDF
            pdf.savefig(fig, bbox_inches="tight", dpi=150)
            plt.close(fig)

    print(f"Combined PDF saved: {output_path}")
