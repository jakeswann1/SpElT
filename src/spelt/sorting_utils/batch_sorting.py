"""Batch sorting utilities for processing multiple sessions using SessionManager."""

from pathlib import Path

import pandas as pd
import spikeinterface as si

from spelt.axona_utils.axona_preprocessing import sort_axona
from spelt.np2_utils.np2_preprocessing import sort_np2
from spelt.sorting_utils.collect_sessions import collect_sessions


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


def sort_from_sheet(
    sheet, probe_to_sort, recording_type, sorting_suffix, config_path=None
):
    """
    Sort sessions directly from Google Sheets (legacy interface).

    This maintains compatibility with the original notebook workflow while
    providing a cleaner interface.

    Parameters
    ----------
    sheet : pandas.DataFrame
        DataFrame from Google Sheets with columns:
        trial_name, path, probe_type, recording_type, Sort, Areas (optional)
    probe_to_sort : str
        Probe type to sort (e.g., 'NP2_openephys', '5x12_buz')
    recording_type : str
        Recording type (e.g., 'NP2_openephys', 'NP2_onebox')
    sorting_suffix : str
        Suffix to append to sorting folder names
    config_path : str or None
        Optional path to session_selection.yaml for filtering

    Returns
    -------
    list
        List of sorting objects for each successfully sorted session
    """
    # Filter sheet for recordings to sort
    sheet_inc = sheet[
        (sheet["Sort"] == "Y")
        & (sheet["probe_type"] == probe_to_sort)
        & (sheet["recording_type"] == recording_type)
    ]

    # Get trial and area lists
    trial_list = sheet_inc["trial_name"].to_list()
    area_list = sheet_inc["Areas"].to_list() if "Areas" in sheet_inc.columns else None

    # Check if area_list is all NaN
    if area_list is not None and all(pd.isna(area_list)):
        area_list = None

    # Create session list
    import numpy as np

    if area_list:
        trial_list_areas = [
            f"{trial_list[i]}_{area_list[i]}" for i in range(len(trial_list))
        ]
        session_list = np.unique(
            [
                f"{t.split('_')[0]}_{t.split('_')[1]}_{t.split('_')[-1]}"
                for t in trial_list_areas
            ]
        )
    else:
        session_list = np.unique(
            [f"{t.split('_')[0]}_{t.split('_')[1]}" for t in trial_list]
        )

    # Collect recordings by session
    recording_list = collect_sessions(
        session_list, trial_list, sheet_inc, probe_to_sort, recording_type, area_list
    )

    # Sort each session
    sortings = []
    for session_data in recording_list:
        session = pd.DataFrame(session_data)
        trial_name = session.iloc[0, 1]
        base_folder = Path(session.iloc[0, 2])
        probe_type = session.iloc[0, 3]
        area = session.iloc[0, -1] if area_list is not None else None

        try:
            # Concatenate recordings
            recordings_concat = si.concatenate_recordings(session.iloc[:, 0].to_list())
            print(f"Sorting {recordings_concat}")

            if probe_type == "NP2_openephys":
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
                session.to_csv(output_path / "session.csv", index=False)

            elif probe_type == "5x12_buz":
                # Save concatenated recording to .dat if needed
                concat_path = base_folder / "concat.dat"
                if concat_path.exists():
                    print(f"{concat_path} already exists, skipping concatenation")
                else:
                    si.write_binary_recording(recordings_concat, str(concat_path))
                    print(f"Concatenated recording saved to {concat_path}")

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
                session.to_csv(output_path / "session.csv", index=False)

            else:
                print(f"Probe type {probe_type} not recognized, skipping")
                continue

            sortings.append(sorting)

        except Exception as e:
            print(f"Error sorting {trial_name}: {e}")
            continue

    return sortings
