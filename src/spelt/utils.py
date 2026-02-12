# Various functions useful for analysis
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import spikeinterface as si
from scipy.sparse import coo_matrix

if TYPE_CHECKING:
    from spelt.ephys import ephys


def load_session(obj, lfp_sampling_rate):
    """
    Loads all data for all trials of a session into an existing ephys class object
    Params:
     - obj: instance of an ephys class object
    """
    for i in range(len(obj.trial_list)):
        obj.load_metadata(i)
        obj.load_pos(i)
        (obj.load_lfp(i, lfp_sampling_rate),)
    obj.load_spikes("good")


def gs_to_df(url: str) -> pd.DataFrame:
    # Replace /edit#gid= with /export?format=csv&gid= for any gid number
    url = re.sub(r"/edit#gid=(\d+)", r"/export?format=csv&gid=\1", url)

    # Replace /edit?gid=X#gid=X with /export?format=csv&gid=X for any gid number
    csv_export_url = re.sub(
        r"/edit\?gid=(\d+)#gid=\d+", r"/export?format=csv&gid=\1", url
    )

    df = pd.read_csv(csv_export_url, on_bad_lines="skip")
    return df


def find_all_sessions(
    sheet_path: str,
    data_path: str,
    raw_only: bool = False,
    probe: str | list[str] | None = None,
    animal: str | list[str] | None = None,
    area: str | list[str] | None = None,
    date: str | list[str] | None = None,
    age: int | list[int] | None = None,
) -> dict[str, str]:
    """
    Function to find all sessions and session paths from Recording Master Spreadsheet

    Params:
     - sheet_path: Path to the Google Sheets spreadsheet
     - data_path: Base path to data directory
     - raw_only: If True, exclude thresholded format sessions
     - probe: Single probe type or list of probe types to filter by
     - animal: Single animal or list of animals to filter by
     - area: Single area or list of areas to filter by
     - date: Single date (YYYY-MM-DD) or list of dates to filter by
     - age: Single age or list of ages to filter by

    Returns:
     - Dictionary mapping session names to their full paths
    """
    sheet = gs_to_df(sheet_path)
    sheet_inc = sheet[sheet["Include"] == "Y"]

    if raw_only:
        sheet_inc = sheet_inc[sheet_inc["Format"] != "thresholded"]

    if animal is not None:
        if isinstance(animal, str):
            sheet_inc = sheet_inc[sheet_inc["Animal"] == animal]
        else:
            sheet_inc = sheet_inc[sheet_inc["Animal"].isin(animal)]

    if probe is not None:
        if isinstance(probe, str):
            sheet_inc = sheet_inc[sheet_inc["probe_type"] == probe]
        else:
            sheet_inc = sheet_inc[sheet_inc["probe_type"].isin(probe)]

    if area is not None:
        if isinstance(area, str):
            sheet_inc = sheet_inc[sheet_inc["Areas"] == area]
        else:
            sheet_inc = sheet_inc[sheet_inc["Areas"].isin(area)]

    if date is not None:
        if isinstance(date, str):
            sheet_inc = sheet_inc[sheet_inc["Date"] == date]
        else:
            sheet_inc = sheet_inc[sheet_inc["Date"].isin(date)]

    if age is not None:
        if isinstance(age, int):
            sheet_inc = sheet_inc[sheet_inc["Age"] == age]
        else:
            sheet_inc = sheet_inc[sheet_inc["Age"].isin(age)]

    session_list = np.unique(sheet_inc["Session"].to_list())
    session_dict = {}

    for i in session_list:
        session_df = sheet_inc[sheet_inc["Session"] == i]
        sheet_path = session_df["path"].values[0]
        # Use sheet's path column as source of truth, prepend data_path
        path_to_session = f"{data_path}/{sheet_path.lstrip('/')}"
        session_dict[i] = path_to_session

    return session_dict


def load_config(config_path: str, config_name: str) -> dict[str, str]:
    import yaml

    with open(config_path) as file:
        config = yaml.safe_load(file)

    # Navigate to the specified configuration section
    config_parts = config_name.split(".")
    session_config = config
    for part in config_parts:
        session_config = session_config[part]

    return session_config


def load_sessions_from_config(config_path: str, config_name: str) -> dict[str, str]:
    """
    Load session dictionary using YAML configuration file.

    Parameters:
    -----------
    config_path : str
        Path to the YAML configuration file
    config_name : str, optional
        Name of the configuration section to use (default: "session_finder")
        Can also use example configurations like "examples.ca1_neuropixels"

    Returns:
    --------
    dict[str, str]
        Dictionary mapping session names to their full paths

    Examples:
    ---------
    # Use main configuration
    sessions = load_sessions_from_config('session_selection.yaml')

    # Use example configuration
    sessions = load_sessions_from_config('session_selection.yaml', 'spike_sorting')
    """
    session_config = load_config(config_path, config_name)

    # Extract parameters
    sheet_path = session_config["sheet_path"]
    data_path = session_config["data_path"]
    filters = session_config["filters"]

    # Call find_all_sessions with the configuration
    return find_all_sessions(
        sheet_path=sheet_path,
        data_path=data_path,
        raw_only=filters["raw_only"],
        probe=filters["probe"],
        animal=filters["animal"],
        area=filters["area"],
        date=filters.get("date", None),
        age=filters.get("age", None),
    )


def make_df_all_sessions(session_dict, sheet_url):
    """
    Function to make a dataframe of all sessions and their paths
    """

    from spelt.ephys import ephys

    # Initialise DataFrame for ephys objects
    df_all_sessions = pd.DataFrame(
        data=None, index=session_dict.keys(), columns=["ephys_object"], dtype="object"
    )

    for i, session_path in enumerate(session_dict.values()):
        # Create ephys object for session and add to dataframe
        obj = ephys(sheet_url=sheet_url, path=session_path)
        df_all_sessions.at[list(session_dict.keys())[i], "ephys_object"] = obj

    return df_all_sessions


def select_spikes_by_trial(
    spike_data: dict, trials: int | list[int], trial_offsets: list[float]
):
    """
    Select spikes from specific trials.
    Returns spikes time-indexed from 0 at the start of each trial

    Parameters:
    - spike_data: Dictionary containing spike data (including 'spike_trial').
    - trials: trial number(s) to filter by.
    - trial_offsets: trial offset start times from 0 (FOR ALL TRIALS IN SESSION)

    Returns:
    - Dictionary containing filtered spike times and clusters.
    """
    if isinstance(trials, int):
        trials = [trials]  # Convert single trial number to list

    result = {}
    # Select spikes by trial and reset spike times to start at 0 for each trial
    for trial in trials:
        mask = np.isin(spike_data["spike_trial"], trial)
        result[trial] = {
            "spike_times": spike_data["spike_times"][mask] - trial_offsets[trial],
            "spike_clusters": spike_data["spike_clusters"][mask],
        }

    return result


def find_template_for_clusters(clu, spike_templates):
    """
    Determine the most represented template for each cluster and return as a dictionary.

    Parameters:
    -----------
    clu : np.ndarray
        Array of cluster IDs corresponding to each spike.
    spike_templates : np.ndarray
        Array of template IDs corresponding to each spike.

    Returns:
    --------
    temp_per_clu_dict : dict
        Dictionary where keys are cluster IDs and values are the template most
        represented for that cluster.
    """
    # Ensure the input arrays are 1D
    clu = clu.reshape(-1)
    spike_templates = spike_templates.reshape(-1)

    # Create a sparse matrix to count occurrences
    temp_counts_by_clu = coo_matrix(
        (np.ones(clu.shape[0]), (clu, spike_templates))
    ).toarray()

    # Find the column index with the maximum count for each row (cluster)
    temp_per_clu = np.argmax(temp_counts_by_clu, axis=1) - 1

    # Convert to float array for inserting NaN values
    temp_per_clu = temp_per_clu.astype(float)

    # Identify and set NaN for non-existent clusters
    existent_clusters = np.unique(clu)
    non_existent_clusters = np.setdiff1d(
        np.arange(len(temp_per_clu)), existent_clusters
    )
    temp_per_clu[non_existent_clusters] = np.nan

    # Create dictionary mapping cluster ID to the most represented template ID
    temp_per_clu_dict = {
        cluster: template
        for cluster, template in enumerate(temp_per_clu)
        if not np.isnan(template)
    }

    return temp_per_clu_dict


def compute_extensions_lazy(analyzer: si.SortingAnalyzer, extension_list: list[str]):
    """
    Compute extensions for a given sorting extractor using a list of extensions.
    This function checks if each extension is already computed before computing it.

    Parameters:
    -----------
    analyzer : si.SortingAnalyzer
        The sorting analyzer object.
    extension_list : list[str]
        List of extensions to compute.

    Returns:
    --------
    analyzer : si.SortingAnalyzer
        The updated sorting analyzer object with computed extensions.
    """
    for ext in extension_list:
        if analyzer.has_extension(ext):
            print(f"Extension {ext} already computed, skipping computation.")
        else:
            analyzer.compute(ext)
    return analyzer


def get_subject_data(sheet_url: str, subject_id: str | None = None) -> pd.DataFrame:
    """
    Retrieve subject data from a Google Sheets URL.

    Parameters:
    sheet_url (str): The URL of the Google Sheets document.

    Returns:
    pd.DataFrame: A DataFrame containing subject data.
    """
    df = gs_to_df(sheet_url)
    if subject_id is not None:
        subject_data = df[df["ID"] == subject_id]
    else:
        subject_data = df

    return subject_data


def load_session_spatial_data(session_name: str, obj: "ephys") -> dict:
    """
    Load preprocessed spatial information data for a single session.

    This function loads spatial information, significance results, and place cell
    classifications from files generated by the preprocessing pipeline
    (specifically the Place Cell Plotting notebook).

    Parameters:
    -----------
    session_name : str
        Name of the session
    obj : ephys
        ephys object for the session

    Returns:
    --------
    dict
        Dictionary containing:
        - 'session': session name
        - 'status': 'success', 'skipped', or 'error'
        - 'animal': animal ID (if successful)
        - 'age': age in days (if successful)
        - 'date': recording date (if successful)
        - 'n_cells': total number of cells (if successful)
        - 'n_place_cells': number of place cells (if successful)
        - 'cell_data': list of dicts, one per cell (averaged across trials) with keys:
            - 'cluster_id': cluster/unit ID
            - 'mean_spatial_info': mean spatial information across trials
            - 'std_spatial_info': std of spatial information across trials
            - 'n_trials': number of trials
            - 'mean_p_value': mean p-value across trials
            - 'is_place_cell': boolean indicating if classified as place cell
            - 'total_n_spikes': total number of spikes across all trials
            - 'mean_n_spikes': mean number of spikes per trial
        - 'trial_data': list of dicts, one per trial per cell with keys:
            - 'cluster_id': cluster/unit ID
            - 'trial_name': trial identifier
            - 'trial_type': trial type (e.g., 'open-field', 't-maze')
            - 'trial_id': trial index/iterator
            - 'spatial_info': spatial information for this specific trial
            - 'p_value': p-value for this specific trial
            - 'is_place_cell': boolean indicating if classified as place cell
            - 'n_spikes': number of spikes in this trial
        - 'message': error or skip message (if not successful)
        - 'error', 'traceback': error details (if status is 'error')

    Example:
    --------
    >>> from spelt.ephys import ephys
    >>> from spelt.utils import load_session_spatial_data
    >>> obj = ephys(sheet_url=url, path='/path/to/session')
    >>> result = load_session_spatial_data('session1', obj)
    >>> if result['status'] == 'success':
    ...     for cell in result['cell_data']:
    ...         print(f"Cell {cell['cluster_id']}: SI={cell['mean_spatial_info']:.3f}")
    """
    from pathlib import Path

    try:
        # Check for required files
        spatial_info_path = Path(obj.recording_path) / "spatial_info_dict.npy"
        spatial_sig_path = Path(obj.recording_path) / "spatial_significance_dict.npy"
        place_cells_path = Path(obj.recording_path) / "place_cells.npy"

        if not all(
            [
                spatial_info_path.exists(),
                spatial_sig_path.exists(),
                place_cells_path.exists(),
            ]
        ):
            return {
                "session": session_name,
                "status": "skipped",
                "message": (
                    "Preprocessing data not found "
                    "(run Place Cell Plotting notebook first)"
                ),
            }

        # Load data
        spatial_info_dict = np.load(spatial_info_path, allow_pickle=True).item()
        spatial_sig_dict = np.load(spatial_sig_path, allow_pickle=True).item()
        place_cells = np.load(place_cells_path, allow_pickle=True)

        # Load analyzer for spike count data
        if obj.analyzer is None:
            obj._load_ephys(from_disk=True)

        # Create trial name to trial type mapping and trial index mapping
        trial_type_map = {}
        trial_name_to_idx = {}
        for i, trial_name in enumerate(obj.trial_list):
            trial_type = obj.trial_types[i]
            trial_type_map[trial_name] = trial_type
            trial_name_to_idx[trial_name] = i

        # Extract data for each cell (both trial-level and averaged)
        cell_data = []
        trial_data = []

        for cluster_id in spatial_info_dict.keys():
            # Get spatial info across all trials
            si_dict = spatial_info_dict[cluster_id]
            si_values = list(si_dict.values())
            trial_names = list(si_dict.keys())

            mean_si = np.mean(si_values)
            std_si = np.std(si_values)

            # Get significance data
            p_values_dict = spatial_sig_dict["p_values"].get(cluster_id, {})
            p_values = list(p_values_dict.values())
            mean_p = np.mean(p_values) if p_values else np.nan

            # Check if place cell
            is_place_cell = cluster_id in place_cells

            # Get spike counts for each trial
            spike_counts = []
            for trial_name in trial_names:
                trial_idx = trial_name_to_idx.get(trial_name)
                if trial_idx is not None:
                    try:
                        # Get spike train for this unit and trial
                        spike_train = obj.analyzer.sorting.get_unit_spike_train(
                            cluster_id, segment_index=trial_idx, return_times=False
                        )
                        n_spikes = len(spike_train)
                    except Exception:
                        # If spike data not available, use NaN
                        n_spikes = np.nan
                else:
                    n_spikes = np.nan
                spike_counts.append(n_spikes)

            # Calculate total and mean spike counts
            valid_spike_counts = [sc for sc in spike_counts if not np.isnan(sc)]
            total_n_spikes = (
                int(np.sum(valid_spike_counts)) if valid_spike_counts else 0
            )
            mean_n_spikes = (
                np.mean(valid_spike_counts) if valid_spike_counts else np.nan
            )

            # Store cell-level data (averaged across trials)
            cell_data.append(
                {
                    "cluster_id": cluster_id,
                    "mean_spatial_info": mean_si,
                    "std_spatial_info": std_si,
                    "n_trials": len(si_values),
                    "mean_p_value": mean_p,
                    "is_place_cell": is_place_cell,
                    "total_n_spikes": total_n_spikes,
                    "mean_n_spikes": mean_n_spikes,
                }
            )

            # Store trial-level data (individual trials)
            for trial_name, si_value, n_spikes in zip(
                trial_names, si_values, spike_counts
            ):
                p_value = p_values_dict.get(trial_name, np.nan)
                trial_type = trial_type_map.get(trial_name, "unknown")
                trial_idx = trial_name_to_idx.get(trial_name)
                trial_data.append(
                    {
                        "cluster_id": cluster_id,
                        "trial_name": trial_name,
                        "trial_type": trial_type,
                        "trial_id": trial_idx,
                        "spatial_info": si_value,
                        "p_value": p_value,
                        "is_place_cell": is_place_cell,
                        "n_spikes": n_spikes,
                    }
                )

        return {
            "session": session_name,
            "status": "success",
            "animal": obj.animal,
            "age": obj.age,
            "date": obj.date,
            "n_cells": len(cell_data),
            "n_place_cells": int(np.sum([c["is_place_cell"] for c in cell_data])),
            "cell_data": cell_data,
            "trial_data": trial_data,
        }

    except Exception as e:
        import traceback

        return {
            "session": session_name,
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
