"""Cache management utilities for ephys data."""

import pickle as pkl
from pathlib import Path


def load_pickle(path: Path) -> dict:
    """
    Load pickled data with error handling.

    Parameters
    ----------
    path : Path
        Path to pickle file

    Returns
    -------
    dict
        Loaded data

    Raises
    ------
    FileNotFoundError
        If pickle file doesn't exist
    """
    with path.open("rb") as f:
        return pkl.load(f)  # noqa: S301


def save_pickle(data: dict, path: Path) -> None:
    """
    Save data to pickle file.

    Parameters
    ----------
    data : dict
        Data to save
    path : Path
        Path to save pickle file
    """
    with path.open("wb") as f:
        pkl.dump(data, f)


def get_cache_path(recording_path: Path, data_type: str, trial_index: int) -> Path:
    """
    Get standardized cache file path.

    Parameters
    ----------
    recording_path : Path
        Path to recording directory
    data_type : str
        Type of data ('lfp', 'ttl', 'theta_phase')
    trial_index : int
        Trial index

    Returns
    -------
    Path
        Cache file path
    """
    return recording_path / f"{data_type}_data_trial{trial_index}.pkl"
