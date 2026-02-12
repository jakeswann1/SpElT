from pathlib import Path

from spelt.axona_utils.axona_preprocessing import pos_from_bin, preprocess_axona
from spelt.axona_utils.load_ephys import load_axona_ephys
from spelt.np2_utils.load_ephys import load_np2_recording


def parse_session_info(session):
    parts = session.split("_")
    base_session = f"{parts[0]}_{parts[1]}"
    area = parts[2] if len(parts) > 2 else None
    return base_session, area


def load_and_process_recording(
    trial_info, trial, probe_to_sort, base_folder, recording_type, area=None
):
    """Load and process a recording based on probe and recording type."""
    base_path = Path(base_folder)

    if probe_to_sort == "NP2_openephys":
        # Construct path with optional area
        path = base_path / trial / area if area else base_path / trial

        # Select appropriate loader based on recording type
        if recording_type == "NP2_openephys":
            return load_np2_recording(path, method="pcie")
        elif recording_type == "NP2_onebox":
            return load_np2_recording(path, method="onebox")
        else:
            raise ValueError(f"Unknown recording type: {recording_type}")

    elif probe_to_sort == "5x12_buz":
        # Generate .pos file if it doesn't exist
        trial_path = base_path / trial
        pos_file = trial_path.with_suffix(".pos")
        if not pos_file.exists():
            pos_from_bin(str(trial_path))

        # Load and preprocess Axona recording
        recording = load_axona_ephys(
            str(trial_path.with_suffix(".set")), trial_info["probe_type"]
        )
        return preprocess_axona(
            recording=recording,
            recording_name=trial,
            base_folder=str(base_path),
            electrode_type=trial_info["probe_type"],
        )

    else:
        raise ValueError(
            f"Probe type '{probe_to_sort}' not recognized. "
            'Currently only "NP2_openephys" and "5x12_buz" are supported.'
        )


def collect_trial_info(sheet, trial):
    trial_data = sheet[sheet["trial_name"] == trial].iloc[0]
    trial_info = {"path": trial_data["path"], "probe_type": trial_data["probe_type"]}
    # Check for 'Areas' column and include it if present
    trial_info["area"] = trial_data["Areas"] if "Areas" in sheet.columns else None
    return trial_info


def collect_sessions(
    session_list, trial_list, sheet, probe_to_sort, recording_type, area_list
):
    """
    Collect recordings from a list of sessions and trials.

    Returns a list of lists, where each sublist corresponds to a session
    and contains that session's recordings.

    Parameters
    ----------
    session_list : list
        List of session names.
    trial_list : list
        List of trial names.
    sheet : pandas.DataFrame
        DataFrame containing the trial information.
    probe_to_sort : str
        Probe type to sort. Currently only 'NP2_openephys' and '5x12_buz' are supported.
    recording_type : str
        Recording type (e.g., 'NP2_openephys', 'NP2_onebox').
    area_list : list or None
        List of areas to include. If None, all areas will be included.

    Returns
    -------
    recording_list : list
        List of lists containing the recordings for each session.

    """
    recording_list = [[] for _ in session_list]

    for i, session in enumerate(session_list):
        base_session, area = parse_session_info(session)
        for j, trial in enumerate(trial_list):
            if area_list is None or (area in area_list[j] and base_session in trial):
                trial_info = collect_trial_info(sheet, trial)
                base_folder = trial_info["path"]
                print(f"Loading {base_folder}/{trial}")

                recording = load_and_process_recording(
                    trial_info, trial, probe_to_sort, base_folder, recording_type, area
                )
                trial_duration = recording.get_num_samples()

                recording_data = [
                    recording,
                    trial,
                    base_folder,
                    trial_info["probe_type"],
                    trial_duration,
                ]
                if area_list is not None:
                    recording_data.append(area)
                recording_list[i].append(recording_data)

    return recording_list
