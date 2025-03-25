from pathlib import Path

import matplotlib.pyplot as plt
import spikeinterface as si
import spikeinterface.widgets as sw

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
        sw.plot_unit_summary(analyzer, unit_id)

        if output_folder:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                output_folder / f"unit{unit_id}.png", dpi=300, bbox_inches="tight"
            )

        if show:
            plt.show()
        plt.close()
