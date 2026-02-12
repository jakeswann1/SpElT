from pathlib import Path

import numpy as np
import spikeinterface as si

from .loaders.session_loader import extract_session_metadata, load_session_from_sheet

si.set_global_job_kwargs(n_jobs=-2)


class ephys:  # noqa: N801
    """
    Manage ephys data including metadata, position, LFP, and spike data.

    Supports:
        - Axona/DACQ recordings sorted with Kilosort 2, curated with Phy
        - Neuropixels 2 recordings (OpenEphys), sorted with Kilosort 4, curated with Phy
        - Video tracking from Bonsai and DeepLabCut

    Assumes file structure: path_to_data/animal/YYYY-MM-DD/

    Example:
        obj = ephys(path='path/to/animal/date', sheet_url='https://...')
        obj.load_pos([0, 1, 2])
        obj.load_lfp([0, 1, 2], sampling_rate=1000, channels=[0, 1, 2])
        obj.load_spikes()

    Attributes:
        recording_type: Recording system ('nexus', 'NP2_openephys', 'NP2_onebox')
        recording_path: Path to recording data folder
        sorting_path: Path to sorting data folder
        date: Recording date (YYYY-MM-DD format)
        date_short: Recording date (YYMMDD format)
        animal: Animal ID
        age: Animal age at recording time
        probe_type: Probe type identifier
        area: Brain area (e.g., 'CA1')
        session: DataFrame with session info from Google Sheets
        trial_list: List of trial names
        trial_iterators: List of trial indices
        analyzer: SortingAnalyzer object for the session
        lfp_data: List storing LFP data per trial
        pos_data: List storing position data per trial
        sync_data: List storing TTL/sync data per trial
        spike_data: Dictionary storing spike data for session
        max_speed: Maximum speed for position filtering (default: 5 m/s)
        smoothing_window_size: Window size for position smoothing (default: 3)
    """

    def __init__(self, path, sheet_url, analyzer=None, area=None, pos_only=False):
        """
        Initialize the ephys object.

        Args:
            path: Path to the recording directory
            sheet_url: URL of the Google Sheet with session metadata
            analyzer: Pre-computed SortingAnalyzer object (created if None)
            area: Brain area targeted for recording (e.g., 'CA1')
            pos_only: If True, only loads position data (skips spike loading checks)
        """
        self.recording_path = Path(path)

        # Parse date and animal ID from path (for session lookup)
        path_date = self.recording_path.parts[-1]
        path_animal = self.recording_path.parts[-2]

        # Handle both date formats: YYYY-MM-DD (10 chars) and YYMMDD (6 chars)
        if len(path_date) == 10 and "-" in path_date:
            # Date is in YYYY-MM-DD format, convert to YYMMDD for lookup
            date_short_for_lookup = f"{path_date[2:4]}{path_date[5:7]}{path_date[8:10]}"
        elif len(path_date) == 6 and path_date.isdigit():
            # Date is already in YYMMDD format
            date_short_for_lookup = path_date
        else:
            raise ValueError(
                f"Unrecognized date format: {path_date}. "
                "Expected YYYY-MM-DD (e.g., '2026-02-11') or YYMMDD (e.g., '260211')"
            )

        # Load session metadata from Google Sheet
        try:
            session = load_session_from_sheet(
                sheet_url, path_animal, date_short_for_lookup, area, pos_only
            )
        except Exception as e:
            print(f"Failed to load session from Google Sheet: {e}")
            raise

        # Store session DataFrame
        self.session = session

        # Extract trial information
        self.trial_list = session["trial_name"].to_list()
        self.trial_types = session["Trial Type"].to_list()
        self.tracking_types = session["tracking_type"].to_list()
        self.trial_iterators = list(range(len(self.trial_list)))

        # Extract session metadata from sheet (authoritative source)
        (
            self.animal,
            self.date,
            self.age,
            self.probe_type,
            self.area,
            self.recording_type,
        ) = extract_session_metadata(session)

        # Compute date_short from authoritative date
        if len(self.date) == 10 and "-" in self.date:
            # Date is in YYYY-MM-DD format, convert to YYMMDD
            self.date_short = f"{self.date[2:4]}{self.date[5:7]}{self.date[8:10]}"
        elif len(self.date) == 6 and self.date.isdigit():
            # Date is already in YYMMDD format
            self.date_short = self.date
        else:
            raise ValueError(
                f"Unrecognized date format from sheet: {self.date}. "
                "Expected YYYY-MM-DD (e.g., '2026-02-11') or YYMMDD (e.g., '260211')"
            )

        # Configure sorting path based on recording type
        if self.recording_type == "nexus":
            self.sorting_path = (
                self.recording_path / f"{self.date_short}_sorting_ks2_custom"
            )
        elif self.recording_type in ("NP2_openephys", "NP2_onebox"):
            self.sorting_path = (
                self.recording_path
                / f"{self.date_short}_{self.area}_sorting_ks4/sorter_output"
            )
        else:
            raise ValueError(f"Unsupported recording type: {self.recording_type}")

        # Set analyzer
        self.analyzer = analyzer

        # Initialize data storage
        self.spike_data = {}
        self.unit_spikes = None
        self.lfp_data = [None] * len(self.trial_list)
        self.sync_data = [None] * len(self.trial_list)
        self.pos_data = [None] * len(self.trial_list)

        # Set position processing parameters
        self.max_speed = 5
        self.smoothing_window_size = 3

        # Map data types to attribute names
        self._data_type_map = {
            "position": "pos_data",
            "lfp": "lfp_data",
            "ttl": "sync_data",
            "spike": "spike_data",
        }

    def load_spikes(
        self,
        unit_ids: list | None = None,
        quality: list[str] | None = None,
        from_disk=True,
    ):
        """
        Load spike data for the session from Kilosort output.

        Args:
            unit_ids: Unit IDs to load (default: all units)
            quality: Quality labels to filter by (e.g., ['good'], default: all)
            from_disk: If True, loads pre-computed analyzer from disk

        Populates:
            self.spike_data (dict): Spike data with keys:
                - 'spike_times': Spike times in seconds
                - 'spike_clusters': Cluster ID for each spike
                - 'spike_trial': Trial index for each spike
                - 'sampling_rate': Sampling rate in Hz
        """

        if self.analyzer is None:
            self.load_ephys(from_disk=from_disk)

        if unit_ids is not None:
            sorting = self.analyzer.sorting.select_units(unit_ids)
        else:
            sorting = self.analyzer.sorting

        if quality is not None:
            unit_quality = self.analyzer.sorting.get_property("quality")
            quality_mask = np.isin(unit_quality, quality)
            units_to_keep = self.analyzer.sorting.get_unit_ids()[quality_mask]
            sorting = sorting.select_units(units_to_keep)
            unit_ids = units_to_keep

        spike_vector = sorting.to_spike_vector()
        sampling_rate = (
            self.analyzer.get_total_samples() / self.analyzer.get_total_duration()
        )

        self.analyzer.sorting = sorting

        # Populate spike_data
        self.spike_data["spike_times"] = spike_vector["sample_index"] / sampling_rate
        self.spike_data["spike_clusters"] = spike_vector["unit_index"]
        self.spike_data["spike_trial"] = spike_vector["segment_index"]
        self.spike_data["sampling_rate"] = sampling_rate

    def load_single_unit_spike_trains(self, unit_ids=None, sparse=True):
        """
        Load spike trains for specified units across all trials.

        Args:
            unit_ids: Unit IDs to load (default: all good units)
            sparse: If True, uses sparse representation for efficiency

        Returns:
            Dictionary with format {trial: {unit: spike_train}}
            where spike_train is an array of spike times in seconds
        """
        if self.analyzer is None:
            self.load_ephys(keep_good_only=True, sparse=sparse)

        unit_spikes = {}

        if unit_ids is None:
            unit_ids = self.analyzer.sorting.get_unit_ids()

        for trial in self.trial_iterators:
            unit_spikes[trial] = {}
            for unit in unit_ids:
                unit_spikes[trial][unit] = self.analyzer.sorting.get_unit_spike_train(
                    unit, segment_index=trial, return_times=True
                )

        self.unit_spikes = unit_spikes

        return unit_spikes

    def load_pyramidal_cell_ids(self):
        """
        Load pyramidal cell IDs from clusters_inc.npy file.

        Returns cluster IDs that passed pyramidal cell filtering criteria:
        - Mean FR < 10 Hz
        - Mean spike width > 500us
        - Burst index < 25ms

        Returns:
            Array of cluster IDs identified as pyramidal cells.
            Returns empty array if file not found.
        """
        clusters_inc_file = self.recording_path / "clusters_inc.npy"

        if not clusters_inc_file.exists():
            print(f"Warning: clusters_inc.npy not found at {clusters_inc_file}")
            print("Run preprocessing/3. Pyramidal Cell Selection.ipynb first")
            return np.array([])

        return np.load(clusters_inc_file, allow_pickle=True)

    def load_place_cell_ids(self):
        """
        Load place cell IDs from place_cells.npy file.

        Returns cluster IDs with significant spatial selectivity based on:
        - Spatial information (bits/spike)
        - Statistical testing with population shuffles
        - Multiple comparison correction

        Returns:
            Array of cluster IDs identified as place cells.
            Returns empty array if file not found.
        """
        place_cells_file = self.recording_path / "place_cells.npy"

        if not place_cells_file.exists():
            print(f"Warning: place_cells.npy not found at {place_cells_file}")
            print("Run preprocessing/4. Place Cell Plotting.ipynb first")
            return np.array([])

        return np.load(place_cells_file, allow_pickle=True)

    def normalize_trial_list(
        self, trial_list: int | list[int] | None, data_type: str = "position"
    ) -> list[int]:
        """
        Normalize trial list input to a list of integers.

        Args:
            trial_list: The trial list to normalize. Can be an int, list[int], or None.
            data_type: The type of data being loaded (e.g., "position", "lfp", "ttl").

        Returns:
            A list of trial indices.
        """
        if isinstance(trial_list, int):
            return [trial_list]
        elif trial_list is None:
            print(f"No trial list specified, loading {data_type} data for all trials")
            return self.trial_iterators
        return trial_list

    def get_trial_type_indices(self, trial_type: str) -> list[int]:
        """
        Get the indices of trials matching the specified trial type.

        Args:
            trial_type: The trial type to filter by.

        Returns:
            A list of trial indices that match the specified trial type.
        """
        return [
            idx for idx, t_type in enumerate(self.trial_types) if t_type == trial_type
        ]

    def should_load_data(
        self, trial_iterator: int, data_type: str, reload_flag: bool
    ) -> bool:
        """
        Check if data should be loaded for the given trial.

        Args:
            trial_iterator: The trial index to check.
            data_type: The type of data being loaded (e.g., "position", "lfp", "ttl").
            reload_flag: If True, forces reloading of data.

        Returns:
            True if data should be loaded, False otherwise.
        """
        if data_type not in self._data_type_map:
            raise ValueError(f"Unknown data type: {data_type}")

        data_attr = self._data_type_map[data_type]
        if not reload_flag and getattr(self, data_attr)[trial_iterator] is not None:
            print(f"{data_type.capitalize()} already loaded for trial {trial_iterator}")
            return False
        return True

    def log_loading_info(self, path: Path, data_type: str, output_flag: bool):
        """
        Log information about loading data.

        Args:
            path: The path to the data file.
            data_type: The type of data being loaded (e.g., "position", "lfp", "ttl").
            output_flag: If True, print loading information.
        """
        if output_flag:
            print(f"Loading {data_type} data from path: {path}")

    def load_pos(
        self,
        trial_list: int | list[int] | None = None,
        output_flag=True,
        reload_flag=False,
    ):
        """
        Loads and postprocesses the position data for specified trials.
        Can load from Axona .pos files or Bonsai/DeepLabCut .csv files

        Args:
            trial_list: The trial indices to be loaded.
            output_flag: if True, print a statement when loading the pos file
            reload_flag: if True, forces reloading of data. If false, only loads data
                for trials with no position data loaded

        Populates:
            self.pos_data (list): A list that stores position data for each trial.
                The position data for the specified trial is added at the given index.
        """
        from .loaders.position_loader import load_position_data

        trial_list = self.normalize_trial_list(trial_list, "position")

        for trial_idx, trial_iterator in enumerate(trial_list):
            if not self.should_load_data(trial_iterator, "position", reload_flag):
                continue

            path = self.recording_path / self.trial_list[trial_iterator]
            self.log_loading_info(path, "position", output_flag)

            try:
                tracking_type = self.tracking_types[trial_idx]
                trial_name = self.trial_list[trial_iterator]
                trial_type = self.session["Trial Type"].iloc[trial_iterator]
                sync_data = self.sync_data[trial_iterator]

                # Load TTL data if needed for bonsai_roi
                if tracking_type == "bonsai_roi" and sync_data is None:
                    self.load_ttl([trial_iterator], output_flag=False)
                    sync_data = self.sync_data[trial_iterator]

                pos_data = load_position_data(
                    tracking_type=tracking_type,
                    path=path,
                    trial_name=trial_name,
                    trial_type=trial_type,
                    sync_data=sync_data,
                    max_speed=self.max_speed,
                    smoothing_window=self.smoothing_window_size,
                    output_flag=output_flag,
                )
                self.pos_data[trial_iterator] = pos_data
            except Exception as e:
                print(
                    f"Error loading position data for trial {trial_iterator}: {str(e)}"
                )
                self.pos_data[trial_iterator] = None

    def load_lfp(
        self,
        trial_list: int | list[int] = None,
        sampling_rate: int = 1000,
        channels: list | None = None,
        reload_flag=False,
        bandpass_filter: list[float, float] | None = None,
        from_disk=True,
    ):
        """
        Load LFP (Local Field Potential) data for specified trials.

        Tries to load from cached pickle files first, then falls back to computing
        from recordings. Automatically saves newly computed data to pickle cache.

        Args:
            trial_list: Trial indices to load (default: all)
            sampling_rate: Desired sampling rate in Hz (default: 1000)
            channels: Channel IDs to load (default: all channels)
            reload_flag: If True, forces recomputing from recordings
            bandpass_filter: Tuple of (min_freq, max_freq) for filtering
            from_disk: If True, uses sorted analyzer from disk (post-spike-sorting).
                      If False, uses raw recordings only (pre-spike-sorting).

        Populates:
            self.lfp_data (list): LFP data for each trial at the given index
        """
        from .loaders.cache import load_pickle, save_pickle
        from .loaders.lfp_loader import (
            has_requested_channels,
            load_lfp_data,
            subset_lfp_channels,
            validate_lfp_cache,
        )

        trial_list = self.normalize_trial_list(trial_list, "lfp")

        for trial_iterator in trial_list:
            # Check if we need to reload or apply channel selection
            should_reload = (
                reload_flag
                or self.lfp_data[trial_iterator] is None
                or (
                    channels is not None
                    and not has_requested_channels(
                        self.lfp_data[trial_iterator], channels
                    )
                )
            )

            if not should_reload:
                # Apply channel selection if needed
                if channels is not None:
                    self.lfp_data[trial_iterator] = subset_lfp_channels(
                        self.lfp_data[trial_iterator], channels
                    )
                if self.lfp_data[trial_iterator] is not None:
                    print(f"LFP already loaded for trial {trial_iterator}")
                    continue

            lfp_path = self.recording_path / f"lfp_data_trial{trial_iterator}.pkl"

            # Try to load from cached pickle first
            if lfp_path.exists():
                try:
                    lfp_data = load_pickle(lfp_path)

                    # Validate parameters against saved data
                    is_valid, reason = validate_lfp_cache(
                        lfp_data, sampling_rate, channels, bandpass_filter
                    )

                    if not is_valid:
                        raise ValueError(
                            f"Trial {trial_iterator}: {reason}. "
                            "Use reload_flag=True to recompute with new parameters."
                        )

                    self.lfp_data[trial_iterator] = lfp_data

                    # Backward compatibility: add timestamps_relative if not present
                    if "timestamps_relative" not in lfp_data:
                        timestamps = lfp_data["timestamps"]
                        lfp_data["timestamps_relative"] = timestamps - timestamps[0]

                    # Apply channel selection if needed
                    if channels is not None:
                        self.lfp_data[trial_iterator] = subset_lfp_channels(
                            lfp_data, channels
                        )

                    print(f"Loaded cached LFP data for trial {trial_iterator}")
                    continue
                except Exception as e:
                    print(f"Error loading cached LFP for trial {trial_iterator}: {e}")
                    print("Recomputing from recordings...")

            # Load from raw data
            if self.analyzer is None:
                if from_disk:
                    # Try to load sorted analyzer from disk
                    self.load_ephys(sparse=False, from_disk=True)
                else:
                    # Load raw recordings only (no sorting needed)
                    if not hasattr(self, "raw_recording") or self.raw_recording is None:
                        self._load_raw_recordings()

            path = self.recording_path / self.trial_list[trial_iterator]
            self.log_loading_info(path, "lfp", True)

            temp_folder = self.recording_path / "temp"

            try:
                # Use appropriate recording source (both are multi-segment)
                recording = (
                    self.analyzer.recording if self.analyzer else self.raw_recording
                )

                lfp_data = load_lfp_data(
                    recording=recording,
                    segment_index=trial_iterator,
                    sampling_rate=sampling_rate,
                    recording_type=self.recording_type,
                    temp_folder=temp_folder,
                    channels=channels,
                    bandpass_filter=bandpass_filter,
                )

                self.lfp_data[trial_iterator] = lfp_data

                # Save to cache for future loading
                try:
                    save_pickle(lfp_data, lfp_path)
                    file_size = lfp_path.stat().st_size
                    print(
                        f"Saved trial {trial_iterator} LFP data to "
                        f"{lfp_path} ({file_size / 1e6:.2f} MB)"
                    )
                except Exception as e:
                    print(f"Warning: Could not save LFP data to disk: {e}")

            except Exception as e:
                print(f"Error loading LFP data for trial {trial_iterator}: {str(e)}")
                self.lfp_data[trial_iterator] = None

    def _load_raw_recordings(self):
        """
        Load raw recordings without sorting data (for pre-spike-sorting operations).

        Creates a multi-segment recording where each segment corresponds to a trial,
        matching the structure used by the analyzer.

        Populates:
            self.raw_recording: Multi-segment recording with one segment per trial
        """
        from .loaders.ephys_loader import load_trial_recordings

        recording_list = load_trial_recordings(
            recording_path=self.recording_path,
            trial_list=self.trial_list,
            trial_iterators=self.trial_iterators,
            recording_type=self.recording_type,
            probe_type=self.probe_type,
            area=self.area,
        )

        # Create multi-segment recording (preserves segments, not concatenating time)
        if len(recording_list) > 1:
            import spikeinterface as si

            self.raw_recording = si.append_recordings(recording_list)
        else:
            self.raw_recording = recording_list[0]

    def load_ttl(
        self,
        trial_list: int | list[int] | None = None,
        output_flag=True,
        reload_flag=False,
        from_disk=True,
    ):
        """
        Load TTL/sync data for specified trials from OpenEphys recording.

        Tries to load from cached pickle files first, then falls back to computing
        from recordings. Automatically saves newly computed data to pickle cache.

        Args:
            trial_list: Trial indices to load (default: all)
            output_flag: If True, print loading messages
            reload_flag: If True, forces recomputing from recordings
            from_disk: If True, uses sorted analyzer from disk (post-spike-sorting).
                      If False, uses raw recordings only (pre-spike-sorting).

        Populates:
            self.sync_data (list): TTL data for each trial at the given index
        """
        from .loaders.cache import load_pickle, save_pickle
        from .loaders.ttl_loader import load_ttl_data

        if self.probe_type != "NP2_openephys":
            print("TTL data only available for NP2_openephys recordings")
            return

        trial_list = self.normalize_trial_list(trial_list, "ttl")

        for trial_iterator in trial_list:
            if not self.should_load_data(trial_iterator, "ttl", reload_flag):
                continue

            ttl_path = self.recording_path / f"ttl_data_trial{trial_iterator}.pkl"

            # Try to load from cached pickle first
            if ttl_path.exists():
                try:
                    ttl_data = load_pickle(ttl_path)
                    self.sync_data[trial_iterator] = ttl_data
                    if output_flag:
                        print(f"Loaded cached TTL data for trial {trial_iterator}")
                    continue
                except Exception as e:
                    if output_flag:
                        print(
                            f"Error loading cached TTL for trial {trial_iterator}: {e}"
                        )
                        print("Recomputing from recordings...")

            # Load from raw data
            path = self.recording_path / self.trial_list[trial_iterator]
            self.log_loading_info(path, "TTL", output_flag)

            # Get recording object to extract start time
            if not self.analyzer:
                if from_disk:
                    # Try to load sorted analyzer from disk
                    self.load_ephys(sparse=False, from_disk=True)
                else:
                    # Load raw recordings only (no sorting needed)
                    if not hasattr(self, "raw_recording") or self.raw_recording is None:
                        self._load_raw_recordings()

            # Get recording start time from analyzer or raw recording
            if self.analyzer:
                recording_start_time = self.analyzer.recording.get_start_time(
                    segment_index=trial_iterator
                )
            else:
                # Raw recording is multi-segment, use segment_index
                recording_start_time = self.raw_recording.get_start_time(
                    segment_index=trial_iterator
                )

            ttl_data = load_ttl_data(path, recording_start_time, self.recording_type)
            self.sync_data[trial_iterator] = ttl_data

            # Save to cache for future loading
            try:
                save_pickle(ttl_data, ttl_path)
                if output_flag:
                    print(f"Saved TTL data for trial {trial_iterator} to {ttl_path}")
            except Exception as e:
                print(f"Warning: Could not save TTL data to disk: {e}")

    def load_theta_phase(
        self,
        trial_list: int | list[int] | None = None,
        channels: list[int] | None = None,
        clip_value: int | None = 32000,
        output_flag: bool = True,
        reload_flag: bool = False,
        from_disk: bool = True,
    ):
        """
        Load theta phase data and append to existing LFP data.

        Tries to load from cached pickle files first, then falls back to computing
        from LFP data. Automatically saves newly computed data to pickle cache.

        Args:
            trial_list: Trial indices to load (default: all)
            channels: Channel IDs to use (default: all available channels)
            clip_value: Clipping value for LFP (32000 for Axona, None for others)
            output_flag: If True, print loading messages
            reload_flag: If True, forces recomputing from LFP data
            from_disk: If True, uses sorted analyzer from disk (post-spike-sorting).
                      If False, uses raw recordings only (pre-spike-sorting).

        Modifies:
            self.lfp_data[trial]: Adds 'theta_phase', 'cycle_numbers', 'theta_freqs'
        """
        from .loaders.cache import load_pickle, save_pickle
        from .loaders.lfp_loader import subset_lfp_channels
        from .loaders.theta_loader import compute_theta_phase

        # Use all channels if not specified
        if channels is None and output_flag:
            print("Using all available channels for theta phase analysis")

        trial_list = self.normalize_trial_list(trial_list, "theta_phase")

        for trial_iterator in trial_list:
            # Check if theta phase already exists and we're not reloading
            if (
                not reload_flag
                and self.lfp_data[trial_iterator] is not None
                and "theta_phase" in self.lfp_data[trial_iterator]
            ):
                # Apply channel selection if needed
                if channels is not None:
                    self.lfp_data[trial_iterator] = subset_lfp_channels(
                        self.lfp_data[trial_iterator], channels
                    )

                if output_flag:
                    channel_info = (
                        f" (channels {channels})" if channels else " (all channels)"
                    )
                    print(
                        f"Theta phase data already loaded for trial {trial_iterator}"
                        f"{channel_info}"
                    )
                continue

            theta_phase_path = (
                self.recording_path / f"theta_phase_trial_{trial_iterator}.pkl"
            )

            # Try to load from cached pickle first
            if theta_phase_path.exists():
                try:
                    saved_theta_data = load_pickle(theta_phase_path)

                    # Ensure LFP data is loaded
                    if self.lfp_data[trial_iterator] is None:
                        if output_flag:
                            print(f"Loading LFP data for trial {trial_iterator}")
                        self.load_lfp(trial_list=[trial_iterator], from_disk=from_disk)

                    # Add theta phase data to LFP data
                    self.lfp_data[trial_iterator].update(saved_theta_data)

                    # Apply channel selection if needed
                    if channels is not None:
                        self.lfp_data[trial_iterator] = subset_lfp_channels(
                            self.lfp_data[trial_iterator], channels
                        )

                    if output_flag:
                        channel_info = (
                            f" (channels {channels})" if channels else " (all channels)"
                        )
                        print(
                            f"Loaded cached theta phase data for trial "
                            f"{trial_iterator}{channel_info}"
                        )
                    continue

                except Exception as e:
                    if output_flag:
                        print(
                            f"Error loading cached theta phase for trial "
                            f"{trial_iterator}: {e}"
                        )
                        print("Recomputing from LFP data...")

            # Load/process from raw data
            if output_flag:
                print(f"Processing theta phase data for trial {trial_iterator}")

            # Check if LFP data is loaded for this trial
            if self.lfp_data[trial_iterator] is None:
                if output_flag:
                    print(f"Loading LFP data for trial {trial_iterator}")
                self.load_lfp(
                    trial_list=[trial_iterator], channels=channels, from_disk=from_disk
                )

            try:
                # Get LFP data for this trial
                lfp_array = self.lfp_data[trial_iterator]["data"]
                lfp_sampling_rate = self.lfp_data[trial_iterator]["sampling_rate"]
                available_channels = self.lfp_data[trial_iterator].get("channels")

                # Use all available channels if none specified
                if channels is None:
                    channels_to_use = (
                        [int(ch) for ch in available_channels]
                        if available_channels
                        else list(range(lfp_array.shape[1]))
                    )
                else:
                    channels_to_use = channels

                # Compute theta phase
                theta_data = compute_theta_phase(
                    lfp_array, lfp_sampling_rate, channels_to_use, clip_value
                )

                # Add theta phase data to existing LFP data
                self.lfp_data[trial_iterator].update(theta_data)

                # Apply channel selection if needed
                if channels is not None and len(channels) < lfp_array.shape[1]:
                    self.lfp_data[trial_iterator] = subset_lfp_channels(
                        self.lfp_data[trial_iterator], channels
                    )

                if output_flag:
                    print(f"Processed theta phase data for trial {trial_iterator}")

                # Save theta phase data to cache for future loading
                try:
                    save_pickle(theta_data, theta_phase_path)
                    file_size = theta_phase_path.stat().st_size
                    if output_flag:
                        print(
                            f"Saved trial {trial_iterator} theta phase data to "
                            f"{theta_phase_path} ({file_size / 1e6:.2f} MB)"
                        )
                except Exception as e:
                    print(f"Warning: Could not save theta phase data to disk: {e}")

            except Exception as e:
                print(
                    f"Error processing theta phase data for trial {trial_iterator}: {e}"
                )

        return self.lfp_data

    def load_ephys(self, keep_good_only=False, sparse=True, from_disk=True):
        """
        Create or load SortingAnalyzer for spike extraction and LFP loading.

        Args:
            keep_good_only: If True, only loads units labeled as 'good' in Phy
            sparse: If True, uses sparse representation for efficiency
            from_disk: If True, loads pre-computed analyzer from disk if available

        Raises:
            FileNotFoundError: If sorting_analyzer.zarr not found (when from_disk=True)
                or cluster_info.tsv not found (incomplete Phy curation)
            ValueError: If no units remain after filtering
        """
        from .loaders.ephys_loader import (
            create_sorting_analyzer,
            load_sorting_data,
            load_trial_recordings,
            validate_sorting_curation,
        )

        # Try loading from disk first
        if from_disk:
            analyzer_path = self.recording_path / "sorting_analyzer.zarr"
            if analyzer_path.exists():
                disk_analyzer = si.load_sorting_analyzer(analyzer_path)
                self.analyzer = disk_analyzer.copy()
                return
            else:
                raise FileNotFoundError(
                    "Sorting analyzer not found on disk. "
                    "Please run preprocessing script to create it."
                )

        # Load from raw data
        recording_list = load_trial_recordings(
            recording_path=self.recording_path,
            trial_list=self.trial_list,
            trial_iterators=self.trial_iterators,
            recording_type=self.recording_type,
            probe_type=self.probe_type,
            area=self.area,
        )

        validate_sorting_curation(Path(self.sorting_path))

        sorting = load_sorting_data(
            sorting_path=Path(self.sorting_path), keep_good_only=keep_good_only
        )

        self.analyzer, self.raw_recording = create_sorting_analyzer(
            sorting=sorting,
            recording_list=recording_list,
            recording_path=self.recording_path,
            sparse=sparse,
        )

    def _load_ephys(self, keep_good_only=False, sparse=True, from_disk=True):
        """
        Backward-compatible alias for load_ephys().
        New code should use load_ephys().

        .. deprecated::
            Use :meth:`load_ephys` instead.
            This method will be removed in a future version.
        """
        import warnings

        warnings.warn(
            "_load_ephys() is deprecated, use load_ephys() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.load_ephys(
            keep_good_only=keep_good_only, sparse=sparse, from_disk=from_disk
        )
