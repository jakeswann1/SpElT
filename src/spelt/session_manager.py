"""
SessionManager for parallel and efficient processing of ephys sessions.

This module provides a high-level interface for working with multiple ephys sessions,
enabling parallel processing and eliminating repetitive for-loop code.
"""

from functools import partial
from typing import Any, Callable

import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .ephys import ephys
from .utils import load_sessions_from_config


class SessionManager:
    """
    Manage ephys sessions with support for lazy loading and parallel processing.

    This class provides a clean interface to:
    - Load sessions from YAML config files
    - Apply custom analysis functions across all sessions in parallel
    - Cache loaded ephys objects to avoid redundant loading
    - Aggregate results into DataFrames

    Example:
        >>> manager = SessionManager(
        ...     config_path='../session_selection.yaml',
        ...     config_key='all_neuropixels',
        ...     sheet_path='https://...'
        ... )
        >>>
        >>> # Define a custom analysis function
        >>> def count_units(session_name, obj):
        ...     obj._load_ephys(keep_good_only=True)
        ...     return {"session": session_name,"n_units": obj.analyzer.get_num_units()}
        >>>
        >>> # Run in parallel across all sessions
        >>> results = manager.map(count_units, n_jobs=8, show_progress=True)
        >>> df = pd.DataFrame(results)
    """

    def __init__(
        self,
        config_path: str,
        config_key: str,
        sheet_path: str,
        subject_sheet_path: str | None = None,
        cache_objects: bool = False,
    ):
        """
        Initialize the SessionManager.

        Parameters:
            config_path: Path to YAML config file containing session information
            config_key: Key in the YAML file to use (e.g., 'all_neuropixels')
            sheet_path: Google Sheets URL for session metadata
            subject_sheet_path: Optional Google Sheets URL for subject metadata
            cache_objects: If True, cache loaded ephys objects in memory (uses more RAM)
        """
        self.config_path = config_path
        self.config_key = config_key
        self.sheet_path = sheet_path
        self.subject_sheet_path = subject_sheet_path
        self.cache_objects = cache_objects

        # Load session dictionary from config
        self.session_dict = load_sessions_from_config(config_path, config_key)
        self.session_names = list(self.session_dict.keys())
        self.session_paths = list(self.session_dict.values())

        # Cache for ephys objects (if enabled)
        self._object_cache: dict[str, ephys] = {}

        print(f"SessionManager initialized with {len(self.session_dict)} sessions")

    def get_ephys_object(self, session_name: str, **kwargs) -> ephys:
        """
        Get an ephys object for a given session.

        Parameters:
            session_name: Name of the session
            **kwargs: Additional arguments to pass to ephys constructor

        Returns:
            ephys object for the session
        """
        if self.cache_objects and session_name in self._object_cache:
            return self._object_cache[session_name]

        session_path = self.session_dict[session_name]
        obj = ephys(path=session_path, sheet_url=self.sheet_path, **kwargs)

        if self.cache_objects:
            self._object_cache[session_name] = obj

        return obj

    def map(
        self,
        func: Callable[[str, ephys], Any],
        n_jobs: int = 1,
        show_progress: bool = True,
        filter_sessions: Callable[[str], bool] | None = None,
        return_exceptions: bool = False,
        **func_kwargs,
    ) -> list[Any]:
        """
        Apply a function to all sessions, optionally in parallel.

        Parameters:
            func: Function to apply. Should take (session_name, ephys_obj) as arguments
                  and return a result (typically a dict for easy DataFrame conversion)
            n_jobs: Number of parallel jobs. Use -1 for all CPUs
            show_progress: Whether to show a progress bar
            filter_sessions: Optional function to filter which sessions to process.
                           Takes session_name and returns True to include it.
            return_exceptions: If True, return exceptions instead of raising them
            **func_kwargs: Additional keyword arguments to pass to func

        Returns:
            List of results from applying func to each session

        Example:
            >>> def analyze_session(session_name, obj, min_units=5):
            ...     obj._load_ephys(keep_good_only=True)
            ...     n_units = obj.analyzer.get_num_units()
            ...     if n_units < min_units:
            ...         return None
            ...     return {'session': session_name, 'units': n_units}
            >>>
            >>> results = manager.map(analyze_session, n_jobs=8, min_units=10)
            >>> results = [r for r in results if r is not None]  # Filter out None
        """
        # Filter sessions if needed
        if filter_sessions:
            sessions_to_process = [s for s in self.session_names if filter_sessions(s)]
        else:
            sessions_to_process = self.session_names

        # Wrap function to handle ephys object creation
        if func_kwargs:
            func = partial(func, **func_kwargs)

        def _process_session(session_name: str):
            try:
                obj = self.get_ephys_object(session_name)
                result = func(session_name, obj)
                return result
            except Exception as e:
                if return_exceptions:
                    return {"session": session_name, "error": str(e)}
                else:
                    raise

        # Process sessions
        if n_jobs == 1:
            # Serial processing
            if show_progress:
                results = [
                    _process_session(s)
                    for s in tqdm(sessions_to_process, desc="Processing sessions")
                ]
            else:
                results = [_process_session(s) for s in sessions_to_process]
        else:
            # Parallel processing
            results = Parallel(n_jobs=n_jobs)(
                delayed(_process_session)(s)
                for s in tqdm(
                    sessions_to_process,
                    desc="Processing sessions",
                    disable=not show_progress,
                )
            )

        return results

    def map_to_dataframe(
        self,
        func: Callable[[str, ephys], dict[str, Any]],
        n_jobs: int = 1,
        show_progress: bool = True,
        filter_sessions: Callable[[str], bool] | None = None,
        **func_kwargs,
    ) -> pd.DataFrame:
        """
        Apply a function to all sessions and return results as a DataFrame.

        This is a convenience wrapper around map() that automatically converts
        the results to a pandas DataFrame.

        Parameters:
            func: Function to apply. Should return a dict for each session.
            n_jobs: Number of parallel jobs
            show_progress: Whether to show progress bar
            filter_sessions: Optional function to filter sessions
            **func_kwargs: Additional keyword arguments to pass to func

        Returns:
            DataFrame with one row per session

        Example:
            >>> def get_unit_count(session_name, obj):
            ...     obj._load_ephys(keep_good_only=True)
            ...     return {
            ...         'session': session_name,
            ...         'animal': obj.animal,
            ...         'date': obj.date,
            ...         'n_units': obj.analyzer.get_num_units()
            ...     }
            >>>
            >>> df = manager.map_to_dataframe(get_unit_count, n_jobs=8)
        """
        results = self.map(
            func=func,
            n_jobs=n_jobs,
            show_progress=show_progress,
            filter_sessions=filter_sessions,
            return_exceptions=False,
            **func_kwargs,
        )

        # Filter out None results
        results = [r for r in results if r is not None]

        return pd.DataFrame(results)

    def get_session_subset(
        self, filter_func: Callable[[str], bool]
    ) -> "SessionManager":
        """
        Create a new SessionManager with a subset of sessions.

        Parameters:
            filter_func: Function that takes session_name and returns True to include

        Returns:
            New SessionManager with filtered sessions

        Example:
            >>> # Only include sessions from animal r1572
            >>> subset = manager.get_session_subset(lambda s: 'r1572' in s)
        """
        filtered_dict = {k: v for k, v in self.session_dict.items() if filter_func(k)}

        new_manager = SessionManager.__new__(SessionManager)
        new_manager.config_path = self.config_path
        new_manager.config_key = f"{self.config_key}_filtered"
        new_manager.sheet_path = self.sheet_path
        new_manager.subject_sheet_path = self.subject_sheet_path
        new_manager.cache_objects = self.cache_objects
        new_manager.session_dict = filtered_dict
        new_manager.session_names = list(filtered_dict.keys())
        new_manager.session_paths = list(filtered_dict.values())
        new_manager._object_cache = {}

        print(f"Created subset with {len(filtered_dict)} sessions")
        return new_manager

    def __len__(self) -> int:
        """Return number of sessions."""
        return len(self.session_dict)

    def __repr__(self) -> str:
        """String representation."""
        return f"SessionManager({len(self)} sessions, config_key='{self.config_key}')"


# Convenience functions for common analysis patterns
def extract_unit_metrics(session_name: str, obj: ephys) -> dict[str, Any]:
    """
    Extract basic unit count metrics for a session.

    This is an example analysis function that can be used with SessionManager.map()
    """
    obj._load_ephys(keep_good_only=True)
    return {
        "session": session_name,
        "animal": obj.animal,
        "date": obj.date,
        "age": obj.age,
        "n_units": obj.analyzer.get_num_units(),
    }
