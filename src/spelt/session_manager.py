"""
SessionManager for parallel and efficient processing of ephys sessions.

This module provides a high-level interface for working with multiple ephys sessions,
enabling parallel processing and eliminating repetitive for-loop code.
"""

import threading
import time
import traceback
import warnings
from functools import partial
from multiprocessing import Manager
from typing import Any, Callable

import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .ephys import ephys
from .utils import load_sessions_from_config

# Try to import ipywidgets for enhanced Jupyter progress display
try:
    from IPython.display import display
    from ipywidgets import HTML, VBox

    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False


def _is_jupyter() -> bool:
    """Check if running in Jupyter environment."""
    try:
        shell = get_ipython().__class__.__name__  # type: ignore # noqa: F821
        return shell is not None
    except NameError:
        return False


class JupyterProgressDisplay:
    """Custom Jupyter progress display with colored segments."""

    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.desc = desc
        self.start_time = time.time()
        self.widget = HTML()
        self.container = VBox([self.widget])
        display(self.container)

    def update(self, completed: int, running: int, pending: int):
        """Update the progress display."""
        # Calculate percentages
        completed_pct = (completed / self.total) * 100 if self.total > 0 else 0
        running_pct = (running / self.total) * 100 if self.total > 0 else 0
        pending_pct = (pending / self.total) * 100 if self.total > 0 else 0

        # Calculate timing information
        elapsed_time = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed_time)

        if completed > 0:
            mean_time = elapsed_time / completed
            mean_str = self._format_time(mean_time)
            # Estimate remaining time
            remaining_time = mean_time * (running + pending)
            remaining_str = self._format_time(remaining_time)
            timing_info = (
                f"Elapsed: {elapsed_str} | "
                f"Mean: {mean_str}/session | "
                f"Est. remaining: {remaining_str}"
            )
        else:
            timing_info = f"Elapsed: {elapsed_str}"

        # Create HTML for multi-colored progress bar with theme-aware styling
        html = f"""
        <style>
            .progress-container {{
                margin: 10px 0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                             Helvetica, Arial, sans-serif;
            }}
            .progress-status {{
                margin-bottom: 5px;
                color: var(--jp-ui-font-color1, inherit);
            }}
            .progress-bar {{
                width: 100%;
                height: 25px;
                background-color: var(--jp-layout-color2,
                                        rgba(128, 128, 128, 0.15));
                border: 1px solid var(--jp-border-color1,
                                       rgba(128, 128, 128, 0.3));
                border-radius: 4px;
                overflow: hidden;
                display: flex;
            }}
            .progress-segment-completed {{
                background-color: var(--jp-success-color1, #28a745);
                height: 100%;
            }}
            .progress-segment-running {{
                background-color: var(--jp-warn-color1, #ffc107);
                height: 100%;
            }}
            .progress-segment-pending {{
                background-color: var(--jp-layout-color3,
                                        rgba(128, 128, 128, 0.3));
                height: 100%;
            }}
            .progress-text {{
                margin-top: 5px;
                font-size: 0.9em;
                color: var(--jp-ui-font-color2, #666);
            }}
            .progress-timing {{
                margin-top: 3px;
                font-size: 0.85em;
                color: var(--jp-ui-font-color3, #888);
            }}
            .status-completed {{ color: var(--jp-success-color1, #28a745); }}
            .status-running {{ color: var(--jp-warn-color1, #ffc107); }}
            .status-pending {{ color: var(--jp-ui-font-color2, #6c757d); }}
        </style>
        <div class="progress-container">
            <div class="progress-status">
                <strong>{self.desc}:</strong> {completed}/{self.total}
                <span class="status-completed">● Completed: {completed}</span>
                <span class="status-running">● Running: {running}</span>
                <span class="status-pending">● Pending: {pending}</span>
            </div>
            <div class="progress-bar">
                <div class="progress-segment-completed" style="width: {completed_pct}%;"></div>
                <div class="progress-segment-running" style="width: {running_pct}%;"></div>
                <div class="progress-segment-pending" style="width: {pending_pct}%;"></div>
            </div>
            <div class="progress-text">
                {completed_pct:.1f}% complete
            </div>
            <div class="progress-timing">
                {timing_info}
            </div>
        </div>
        """  # noqa: E501
        self.widget.value = html

    def _format_time(self, seconds: float) -> str:
        """Format time duration in a human-readable way."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def close(self):
        """Finalize the progress display."""
        self.update(self.total, 0, 0)


class ProgressMonitor:
    """Manages progress tracking and display for parallel processing."""

    def __init__(
        self, total: int, completed_counter, running_counter, use_jupyter: bool = False
    ):
        self.total = total
        self.completed_counter = completed_counter
        self.running_counter = running_counter
        self.use_jupyter = use_jupyter
        self.stop_event = threading.Event()
        self.monitor_thread = None

        # Create appropriate display
        if use_jupyter:
            self.display = JupyterProgressDisplay(
                total=total, desc="Processing sessions"
            )
        else:
            self.display = tqdm(total=total, desc="Processing sessions")

    def start(self):
        """Start the monitoring thread."""
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _monitor_loop(self):
        """Background thread that updates progress display."""
        while not self.stop_event.is_set():
            completed = self.completed_counter.value
            running = self.running_counter.value
            pending = self.total - completed - running

            if self.use_jupyter:
                self.display.update(completed, running, pending)
            else:
                self.display.n = completed
                self.display.set_postfix_str(
                    f"Running: {running} | Pending: {pending}", refresh=False
                )
                self.display.refresh()

            time.sleep(0.1)  # Poll every 100ms

    def stop(self):
        """Stop monitoring and finalize display."""
        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

        if self.use_jupyter:
            self.display.close()
        else:
            self.display.n = self.total
            self.display.set_postfix_str("All complete!", refresh=False)
            self.display.refresh()
            self.display.close()


warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore", message="Versions are not the same.*", category=UserWarning
)


class SessionProcessingError(Exception):
    """Exception raised when a session fails to process."""

    def __init__(
        self, session_name: str, original_error: Exception, traceback_str: str
    ):
        self.session_name = session_name
        self.original_error = original_error
        self.traceback_str = traceback_str
        super().__init__(
            f"\nError processing session '{session_name}':\n"
            f"  Error type: {type(original_error).__name__}\n"
            f"  Error message: {str(original_error)}\n"
            f"\nFull traceback:\n{traceback_str}"
        )

    def __reduce__(self):
        """Make exception picklable for parallel processing."""
        return (
            SessionProcessingError,
            (self.session_name, self.original_error, self.traceback_str),
        )


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
        ephys_kwargs: dict | None = None,
    ):
        """
        Initialize the SessionManager.

        Parameters:
            config_path: Path to YAML config file containing session information
            config_key: Key in the YAML file to use (e.g., 'all_neuropixels')
            sheet_path: Google Sheets URL for session metadata
            subject_sheet_path: Optional Google Sheets URL for subject metadata
            cache_objects: If True, cache loaded ephys objects in memory (uses more RAM)
            ephys_kwargs: Dict of kwargs to pass to ephys constructor
                (e.g., {'pos_only': True})
        """
        self.config_path = config_path
        self.config_key = config_key
        self.sheet_path = sheet_path
        self.subject_sheet_path = subject_sheet_path
        self.cache_objects = cache_objects
        self.ephys_kwargs = ephys_kwargs or {}

        # Load session dictionary from config
        self.session_dict = load_sessions_from_config(config_path, config_key)
        self.session_names = list(self.session_dict.keys())
        self.session_paths = list(self.session_dict.values())

        # Cache for ephys objects (if enabled)
        self._object_cache: dict[str, ephys] = {}

        print(f"SessionManager initialized with {len(self.session_dict)} sessions")

    def _create_session_processor(
        self,
        func: Callable,
        return_exceptions: bool,
        completed_counter=None,
        running_counter=None,
        lock=None,
    ):
        """Create a session processing function with progress tracking."""

        def _process_session(session_name: str):
            # Mark as running
            if running_counter is not None and lock is not None:
                with lock:
                    running_counter.value += 1

            try:
                obj = self.get_ephys_object(session_name)
                result = func(session_name, obj)

                # Mark as completed
                if completed_counter is not None and lock is not None:
                    with lock:
                        running_counter.value -= 1
                        completed_counter.value += 1

                return result
            except Exception as e:
                # Get full traceback
                tb_str = traceback.format_exc()

                # Mark as completed even on error
                if completed_counter is not None and lock is not None:
                    with lock:
                        running_counter.value -= 1
                        completed_counter.value += 1

                if return_exceptions:
                    return {
                        "session": session_name,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "traceback": tb_str,
                    }
                else:
                    raise SessionProcessingError(session_name, e, tb_str) from e

        return _process_session

    def get_ephys_object(self, session_name: str, **kwargs) -> ephys:
        """
        Get an ephys object for a given session.

        Parameters:
            session_name: Name of the session
            **kwargs: Additional arguments to pass to ephys constructor.
                These override any ephys_kwargs set in __init__.

        Returns:
            ephys object for the session
        """
        if self.cache_objects and session_name in self._object_cache:
            return self._object_cache[session_name]

        session_path = self.session_dict[session_name]
        # Merge default ephys_kwargs with any overrides from kwargs
        merged_kwargs = {**self.ephys_kwargs, **kwargs}
        obj = ephys(path=session_path, sheet_url=self.sheet_path, **merged_kwargs)

        if self.cache_objects:
            self._object_cache[session_name] = obj

        return obj

    def _map_serial(
        self, sessions: list[str], process_fn: Callable, show_progress: bool
    ) -> list[Any]:
        """Execute serial processing with optional progress display."""
        results = []
        total = len(sessions)

        if show_progress:
            iterator = tqdm(sessions, desc="Processing sessions")
        else:
            iterator = sessions

        for i, session_name in enumerate(iterator, 1):
            result = process_fn(session_name)
            results.append(result)
            if show_progress:
                print(f"Processed {i}/{total} sessions")

        return results

    def _map_parallel(
        self,
        sessions: list[str],
        process_fn: Callable,
        n_jobs: int,
        show_progress: bool,
    ) -> list[Any]:
        """Execute parallel processing with optional progress display."""
        if not show_progress:
            # No progress tracking - use function as-is
            return Parallel(n_jobs=n_jobs)(delayed(process_fn)(s) for s in sessions)

        # Setup shared counters for progress tracking
        manager = Manager()
        try:
            completed_counter = manager.Value("i", 0)
            running_counter = manager.Value("i", 0)
            lock = manager.Lock()

            # Wrap process function with counter tracking
            def tracked_process_fn(session_name: str):
                # Mark as running
                with lock:
                    running_counter.value += 1

                try:
                    result = process_fn(session_name)
                    # Mark as completed
                    with lock:
                        running_counter.value -= 1
                        completed_counter.value += 1
                    return result
                except Exception:
                    # Mark as completed even on error
                    with lock:
                        running_counter.value -= 1
                        completed_counter.value += 1
                    raise

            # Start progress monitoring
            use_jupyter = JUPYTER_AVAILABLE and _is_jupyter()
            monitor = ProgressMonitor(
                total=len(sessions),
                completed_counter=completed_counter,
                running_counter=running_counter,
                use_jupyter=use_jupyter,
            )
            monitor.start()

            try:
                results = Parallel(n_jobs=n_jobs)(
                    delayed(tracked_process_fn)(s) for s in sessions
                )
            finally:
                monitor.stop()

            return results
        finally:
            # Explicitly shutdown the manager to avoid deprecation warnings
            manager.shutdown()

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
            return_exceptions: If True, return error info as dict instead of raising.
                             Error dicts include: session, error, error_type, traceback
            **func_kwargs: Additional keyword arguments to pass to func

        Returns:
            List of results from applying func to each session

        Raises:
            SessionProcessingError: If a session fails and return_exceptions=False.
                Contains session name, error type, and full traceback.

        Example:
            >>> # Normal usage - raises on error
            >>> def analyze_session(session_name, obj, min_units=5):
            ...     obj._load_ephys(keep_good_only=True)
            ...     n_units = obj.analyzer.get_num_units()
            ...     if n_units < min_units:
            ...         return None
            ...     return {'session': session_name, 'units': n_units}
            >>>
            >>> results = manager.map(analyze_session, n_jobs=8, min_units=10)

            >>> # Graceful error handling
            >>> results = manager.map(analyze_session, n_jobs=8, return_exceptions=True)
            >>> errors = [r for r in results if 'error' in r]
            >>> successful = [r for r in results if 'error' not in r]
        """
        # Filter sessions if needed
        if filter_sessions:
            sessions_to_process = [s for s in self.session_names if filter_sessions(s)]
        else:
            sessions_to_process = self.session_names

        # Wrap function with additional kwargs if provided
        if func_kwargs:
            func = partial(func, **func_kwargs)

        # Create session processor with error handling
        process_fn = self._create_session_processor(
            func=func, return_exceptions=return_exceptions
        )

        # Dispatch to serial or parallel processing
        if n_jobs == 1:
            results = self._map_serial(sessions_to_process, process_fn, show_progress)
        else:
            results = self._map_parallel(
                sessions_to_process, process_fn, n_jobs, show_progress
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
