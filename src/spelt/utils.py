# Various functions useful for analysis
import re

import numpy as np
import pandas as pd
import spikeinterface as si
from scipy.sparse import coo_matrix


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

    session_list = np.unique(sheet_inc["Session"].to_list())
    session_dict = {}

    for i in session_list:
        session_df = sheet_inc[sheet_inc["Session"] == i]
        animal = session_df["Animal"].values[0]
        date_long = session_df["Date"].values[0]
        path_to_session = f"{data_path}/{animal}/{date_long}"
        session_dict[i] = path_to_session

    return session_dict


def load_sessions_from_config(
    config_path: str, config_name: str = "session_finder"
) -> dict[str, str]:
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
    import yaml

    with open(config_path) as file:
        config = yaml.safe_load(file)

    # Navigate to the specified configuration section
    config_parts = config_name.split(".")
    session_config = config
    for part in config_parts:
        session_config = session_config[part]

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
