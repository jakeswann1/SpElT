"""Ephys data loading and SortingAnalyzer creation functions."""

from pathlib import Path

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre


def load_trial_recordings(
    recording_path: Path,
    trial_list: list[str],
    trial_iterators: list[int],
    recording_type: str,
    probe_type: str,
    area: str | None = None,
) -> list[si.BaseRecording]:
    """
    Load recording objects for all trials in a session.

    Args:
        recording_path: Path to the recording directory
        trial_list: List of trial names
        trial_iterators: List of trial indices
        recording_type: Recording system type ('nexus', 'NP2_openephys', 'NP2_onebox')
        probe_type: Probe type identifier
        area: Brain area (required for NP2 recordings)

    Returns:
        List of recording objects for each trial

    Raises:
        ValueError: If recording type or probe type is not supported
    """
    from spelt.axona_utils.load_ephys import load_axona_ephys
    from spelt.np2_utils.load_ephys import load_np2_onebox, load_np2_pcie

    recording_list = []

    for trial_iterator in trial_iterators:
        if recording_type == "nexus":
            path = recording_path / f"{trial_list[trial_iterator]}.set"
            recording = load_axona_ephys(path, probe_type)

        elif probe_type == "NP2_openephys":
            path = recording_path / trial_list[trial_iterator] / area
            if recording_type == "NP2_openephys":
                recording = load_np2_pcie(path)
            elif recording_type == "NP2_onebox":
                recording = load_np2_onebox(path)
            else:
                raise ValueError(
                    f"Recording type {recording_type} not implemented "
                    f"for probe type {probe_type}"
                )

        else:
            raise ValueError(
                f"Recording type {recording_type} or "
                f"probe type {probe_type} not implemented"
            )

        recording_list.append(recording)

    return recording_list


def validate_sorting_curation(sorting_path: Path) -> None:
    """
    Validate that manual curation has been completed in Phy.

    Args:
        sorting_path: Path to the sorting directory

    Raises:
        FileNotFoundError: If cluster_info.tsv not found (incomplete Phy curation)
    """
    cluster_info_path = sorting_path / "cluster_info.tsv"

    if not cluster_info_path.exists():
        raise FileNotFoundError(
            f"Manual curation required: cluster_info.tsv not found at "
            f"{sorting_path}.\n\n"
            f"Please complete these steps:\n"
            f"  1. Open the sorting in Phy GUI:\n"
            f"     phy template-gui {sorting_path}\n"
            f"  2. Manually curate units (label as 'good', 'mua', or 'noise')\n"
            f"  3. Save your curation (Ctrl+S or File → Save)\n"
            f"  4. Close Phy\n"
            f"  5. Re-run this preprocessing step\n\n"
            f"Phy will create cluster_info.tsv when you save your curation."
        )


def load_sorting_data(
    sorting_path: Path, keep_good_only: bool = False
) -> si.BaseSorting:
    """
    Load sorting data from Phy output.

    Args:
        sorting_path: Path to the sorting directory containing Phy output
        keep_good_only: If True, excludes units labeled as 'noise' or 'mua'

    Returns:
        Sorting object with unit spike times

    Raises:
        ValueError: If no units remain after filtering
    """
    # Load sorting with optional filtering
    if keep_good_only:
        sorting = se.read_phy(
            str(sorting_path), exclude_cluster_groups=["noise", "mua", "unclassified"]
        )
    else:
        sorting = se.read_phy(str(sorting_path))

    # Validate units exist
    if sorting.get_num_units() == 0:
        filter_msg = " after filtering for good units" if keep_good_only else ""
        raise ValueError(
            f"No units found in sorting data{filter_msg}. "
            f"Sorting path: {sorting_path}. "
            "Please check that spike sorting has been performed and "
            "units have been curated. "
            "If using keep_good_only=True, ensure at least one unit is "
            "labeled as 'good' in Phy."
        )

    return sorting


def create_sorting_analyzer(
    sorting: si.BaseSorting,
    recording_list: list[si.BaseRecording],
    recording_path: Path,
    sparse: bool = True,
) -> tuple[si.SortingAnalyzer, si.BaseRecording]:
    """
    Create a multi-segment SortingAnalyzer for the session.

    Args:
        sorting: Sorting object with spike times
        recording_list: List of recording objects for each trial
        recording_path: Path to save the analyzer
        sparse: If True, uses sparse representation for efficiency

    Returns:
        Tuple of (analyzer, raw_recording):
            - analyzer: SortingAnalyzer object for the session
            - raw_recording: Unfiltered multi-segment recording for LFP extraction
    """
    # Create multi-segment objects
    multi_segment_sorting = si.split_sorting(sorting, recording_list)
    multi_segment_recording = si.append_recordings(recording_list)

    # Save raw recording for LFP extraction (before filtering)
    raw_recording = multi_segment_recording

    # Apply highpass filter for spike analysis
    filtered_recording = spre.highpass_filter(multi_segment_recording, 300)

    # Create SortingAnalyzer
    analyzer = si.create_sorting_analyzer(
        multi_segment_sorting,
        filtered_recording,
        sparse=sparse,
        format="zarr",
        folder=recording_path / "sorting_analyzer",
        return_scaled=True,
        overwrite=True,
    )

    return analyzer, raw_recording
