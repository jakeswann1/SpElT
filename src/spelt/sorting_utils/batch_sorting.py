"""Batch sorting utilities for processing multiple sessions using SessionManager."""

from pathlib import Path

import pandas as pd
import spikeinterface as si

from spelt.axona_utils.axona_preprocessing import sort_axona
from spelt.np2_utils.np2_preprocessing import sort_np2


def sort_single_session(
    session_name, obj, sorting_suffix, recording_type="NP2_openephys", overwrite=False
):
    """
    Sort a single session using SessionManager.

    This function is designed to work with SessionManager.map().

    Parameters
    ----------
    session_name : str
        Name of the session (e.g., 'r1354_230503')
    obj : ephys
        ephys object for this session (from SessionManager)
    sorting_suffix : str
        Suffix to append to sorting folder name
    recording_type : str
        Recording type (e.g., 'NP2_openephys', 'NP2_onebox')
    overwrite : bool
        Whether to overwrite existing sorting

    Returns
    -------
    dict
        Results dictionary with status and info
    """
    probe_type = obj.probe_type
    area = obj.area if hasattr(obj, "area") else None
    base_folder = obj.recording_path

    # Check if already sorted
    if probe_type == "NP2_openephys":
        sorting_path = base_folder / f"{session_name[:6]}_{area}_{sorting_suffix}"
    else:
        sorting_path = base_folder / f"{session_name[:6]}_{sorting_suffix}"

    if sorting_path.exists() and not overwrite:
        print(f"Skipping {session_name} - already sorted at {sorting_path}")
        return {
            "session": session_name,
            "status": "skipped",
            "sorting_path": str(sorting_path),
        }

    # Load recording (this assumes recordings are concatenated per session)
    # For spike sorting, we need the raw recording, not the ephys object
    # This would need to be adapted based on your specific data structure
    print(f"Sorting {session_name} ({probe_type}, area={area})")

    try:
        if probe_type == "NP2_openephys":
            # Sort NP2 recording
            # Note: You'll need to load the actual recording here
            # This is a placeholder that needs to be adapted to your data structure
            raise NotImplementedError(
                "Direct sorting from ephys objects not yet implemented. "
                "Use sort_from_sheet() instead for now."
            )

        elif probe_type == "5x12_buz":
            # Sort Axona recording
            raise NotImplementedError(
                "Direct sorting from ephys objects not yet implemented. "
                "Use sort_from_sheet() instead for now."
            )

        else:
            raise ValueError(f"Probe type {probe_type} not recognized")

    except Exception as e:
        return {"session": session_name, "status": "error", "error": str(e)}


def sort_from_sheet(sheet, sorting_suffix, data_path):
    """
    Sort all sessions marked with Sort='Y' in Google Sheets.

    Automatically processes each session using its probe_type and recording_type
    from the sheet data.

    Parameters
    ----------
    sheet : pandas.DataFrame
        DataFrame from Google Sheets with columns:
        Session, trial_name, path, probe_type, recording_type, Sort, Areas (optional)
    sorting_suffix : str
        Suffix to append to sorting folder names (e.g., 'sorting_ks4')
    data_path : str
        Base path to data directory (prepended to relative paths from sheet)

    Returns
    -------
    list
        List of sorting objects for each successfully sorted session

    Examples
    --------
    >>> from spelt.utils import gs_to_df, load_config
    >>> config = load_config('../session_selection.yaml', 'spike_sorting')
    >>> sheet = gs_to_df(config['sheet_path'])
    >>> sortings = sort_from_sheet(sheet, 'sorting_ks4', config['data_path'])
    """
    from spelt.sorting_utils.collect_sessions import collect_sessions

    # Filter sheet for recordings to sort
    sheet_inc = sheet[sheet["Sort"] == "Y"].copy()

    if len(sheet_inc) == 0:
        print("No sessions found with Sort='Y'")
        return []

    # Convert relative paths to absolute paths
    sheet_inc["path"] = sheet_inc["path"].apply(
        lambda p: str(Path(data_path) / p.lstrip("/"))
    )

    # Group by Session and Areas (if present)
    group_cols = ["Session"]
    if "Areas" in sheet_inc.columns:
        # Replace NaN areas with a placeholder for grouping
        sheet_inc["Areas"] = sheet_inc["Areas"].fillna("_no_area_")
        group_cols.append("Areas")

    session_groups = sheet_inc.groupby(group_cols)

    print(
        f"Found {len(session_groups)} sessions to sort " f"({len(sheet_inc)} trials)\n"
    )

    # Sort each session
    sortings = []
    for group_key, group_df in session_groups:
        # Extract session name and area
        if isinstance(group_key, tuple):
            session_name, area = group_key
            area = None if area == "_no_area_" else area
        else:
            session_name = group_key
            area = None

        # Get session metadata from first trial (all trials share these values)
        first_trial = group_df.iloc[0]
        probe_type = first_trial["probe_type"]
        recording_type = first_trial["recording_type"]
        base_folder = Path(first_trial["path"])  # Already absolute path

        # Display processing message
        session_display = f"{session_name}_{area}" if area else session_name
        print(f"Processing: {session_display} ({probe_type}/{recording_type})")

        # Prepare trial lists for collect_sessions
        trial_list = group_df["trial_name"].to_list()

        # For collect_sessions, we need to match the expected format
        # It expects session names like "260211_r0000_CA1" (matching trial format)
        # Extract date and animal from the Session column format (animal_date)
        animal, date = session_name.split("_")
        # Construct session name in trial format (date_animal)
        session_for_collect = f"{date}_{animal}"
        if area:
            session_for_collect = f"{session_for_collect}_{area}"
            area_list = [area] * len(trial_list)
        else:
            area_list = None

        # Load and concatenate recordings
        recording_list = collect_sessions(
            [session_for_collect],
            trial_list,
            sheet_inc,  # Pass full filtered sheet, not just group
            probe_type,
            recording_type,
            area_list,
        )

        if not recording_list:
            print("  → No recordings found, skipping\n")
            continue

        # Process the session data
        session_data = recording_list[0]
        session_df = pd.DataFrame(session_data)

        # Concatenate recordings
        recordings_concat = si.concatenate_recordings(session_df.iloc[:, 0].to_list())

        # Sort based on probe type
        if probe_type == "NP2_openephys":
            # Get trial name for output folder
            trial_name = first_trial["trial_name"]

            # Sort NP2 recording
            sorting = sort_np2(
                recording=recordings_concat,
                recording_name=trial_name,
                base_folder=str(base_folder),
                sorting_suffix=f"{area}_{sorting_suffix}",
                area=area,
            )

            # Save session info
            output_path = base_folder / f"{trial_name[:6]}_{area}_{sorting_suffix}"
            session_df.to_csv(output_path / "session.csv", index=False)

        elif probe_type == "5x12_buz":
            # Get trial name for output folder
            trial_name = first_trial["trial_name"]

            # Save concatenated recording to .dat if needed
            concat_path = base_folder / "concat.dat"
            if concat_path.exists():
                print(f"  → {concat_path} already exists, " "skipping concatenation")
            else:
                si.write_binary_recording(recordings_concat, str(concat_path))
                print(f"  → Concatenated recording saved to {concat_path}")

            # Sort Axona recording
            sorting = sort_axona(
                recording=recordings_concat,
                recording_name=trial_name,
                base_folder=str(base_folder),
                electrode_type=probe_type,
                sorting_suffix=sorting_suffix,
            )

            # Save session info
            output_path = base_folder / f"{trial_name[:6]}_{sorting_suffix}"
            session_df.to_csv(output_path / "session.csv", index=False)

        else:
            print(f"  → Probe type {probe_type} not recognized, skipping\n")
            continue

        sortings.append(sorting)
        print("  → Successfully sorted\n")

    print(f"{'='*80}")
    print(
        f"Completed! Successfully sorted "
        f"{len(sortings)}/{len(session_groups)} sessions."
    )
    print(f"{'='*80}")

    return sortings
