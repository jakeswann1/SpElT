from pathlib import Path

import numpy as np
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre

from spelt.axona_utils.load_ephys import load_axona_ephys
from spelt.np2_utils.load_ephys import load_np2_onebox, load_np2_pcie

from .utils import gs_to_df

si.set_global_job_kwargs(n_jobs=-2)


class ephys:  # noqa: N801
    """
    A class to manage ephys data, including metadata, position, LFP, and spike data
    recorded from (currently):
    - raw DACQ recordings sorted with Kilosort 2 and curated with phy
    - Neuropixels 2 recordings acquired with OpenEphys, sorted with Kilosort 4 and
      curated with phy, with video tracking data from Bonsai

    Assumes a basic file structure of path_to_data/animal/YYYY-MM-DD/ for each session

    Usage:
        # Initialize the class with recording type and optional path to session.
        obj = ephys('nexus', 'path/to/recording/data/animalID/date')
        obj = ephys('nexus') will prompt a box to select the recording folder


        # Load metadata for a list of trials.
        obj.load_metadata([0, 2, 3])

        # Load position data for a list of trials.
        obj.load_pos([0, 1, 2])

        # Load LFP data for a list of trials with a specific sampling rate and channels.
        obj.load_lfp([0, 1, 2], sampling_rate = 1000, channels = [0, 1, 2])

        # Load spike data for the session.
        obj.load_spikes()

    Attributes:
        recording_type (str): Type of the recording.
            Current options: 'nexus', 'np2_openephys'

        path (str): Path to the specific animal and date recording folder
        sheet_url (str): URL of the Google Sheet containing additional metadata
        area (str): Brain area targeted for recording (optional)

        recording_path (str): Path to the recording data folder
        sorting_path (str): Path to the sorting data folder

        date (str): Date of the recording in the format 'YYYY-MM-DD'
        date_short (str): Date of the recording in the format 'YYMMDD'
        animal (str): Animal ID

        age (int): Age of the animal at the time of recording
        probe_type (str): Type of probe used for recording
        area (str): Brain area targeted for recording (optional)

        session (pd.DataFrame): Dataframe with session information.
        trial_list (list): List containing the names of each trial.
        trial_iterators (list): List containing the indices of each trial.

        analyzer (SortingAnalyzer): SortingAnalyzer object for the whole session

        metadata (list): List to store metadata for each trial.
        lfp_data (list): List to store LFP data for each trial.
        pos_data (list): List to store position data for each trial.
        sync_data (list): List to store synchronization data for each trial.
        spike_data (dict): Dictionary to store spike data for each trial.

        max_speed (int): Maximum speed constant for position processing.
        smoothing_window_size (int): Smoothing window size for position processing.
    """

    def __init__(self, path, sheet_url, analyzer=None, area=None, pos_only=False):
        """
        Initialize the ephys object.

        Parameters:
        path (str): The path to the recording.
        sheet_url (str): URL of the Google Sheet containing additional metadata.
        analyzer (SortingAnalyzer, optional): A SortingAnalyzer object for the session.
            Default is None, a new one will be created on _load_ephys() call
        area (str, optional): The brain area targeted for recording. Default is None.
        pos_only (bool, optional): If True, only loads position data. Default is False.
        """

        self.recording_path = Path(path)

        # Get date and animal ID from path
        self.date = self.recording_path.parts[-1]
        self.date_short = f"{self.date[2:4]}{self.date[5:7]}{self.date[8:10]}"
        self.animal = self.recording_path.parts[-2]

        # Get session info from Google Sheet
        try:
            df = gs_to_df(sheet_url)
            df = df[df["Include"] == "Y"]
            df = df[df["Areas"] == area] if area else df
            session = df.loc[df["Session"] == f"{self.animal}_{self.date_short}"]
            if session.empty:
                raise ValueError(
                    f"Session {self.animal}_{self.date_short} not found in Google Sheet"
                )
            if "Format" in session.columns and not pos_only:
                session = session[
                    session["Format"] != "thresholded"
                ]  # Drops thresholded Axona recordings unless pos_only=True
                if session.empty:
                    raise ValueError(
                        f"Session {self.animal}_{self.date_short} has no "
                        f"non-thresholded recordings (all marked as thresholded)"
                    )
            self.session = session
        except Exception as e:
            print("Google Sheet not found, please specify a valid URL", e)
            raise e

        # Load some metadata from the Google Sheet
        self.trial_list = session.loc[:, "trial_name"].to_list()
        self.trial_types = session.loc[:, "Trial Type"].to_list()
        self.tracking_types = session.loc[:, "tracking_type"].to_list()

        self.age = (
            int(session.loc[:, "Age"].iloc[0]) if "Age" in session.columns else None
        )
        self.probe_type = (
            session.loc[:, "probe_type"].iloc[0]
            if "probe_type" in session.columns
            else None
        )
        self.area = (
            session.loc[:, "Areas"].iloc[0] if "Areas" in session.columns else None
        )
        self.recording_type = session.loc[:, "recording_type"].iloc[0]

        # Set sorting path based on recording type
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
            raise ValueError("Recording type not recognised from sheet")

        # Collect each trial number
        self.trial_iterators = [i for i, _ in enumerate(self.trial_list)]

        self.analyzer = analyzer

        # Initialise data variables
        self.spike_data = {}
        self.unit_spikes = None
        self.lfp_data = [None] * len(self.trial_list)
        self.sync_data = [None] * len(self.trial_list)
        self.pos_data = [None] * len(self.trial_list)

        # Set constants for position processing
        self.max_speed = 5
        self.smoothing_window_size = 3

        # Map data types to their attribute names
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
        load_templates=False,
        load_waveforms=False,
        load_channels=False,
        from_disk=True,
    ):
        """
        Loads the spike data for the session.
        Currently from Kilosort 2/4 output files using the spikeinterface package

        Args:
            unit_ids: A list of unit IDs to load. Default is all
            quality: A list of quality labels to load. Default is all
            load_templates (bool): If True, loads the templates for the specified units.
            load_waveforms (bool): If True, loads the waveforms for the specified units.
            load_channels (bool): If True, loads the peak channels for each unit.

        Populates:
            self.spike_data (dict): A dictionary that stores spike data for the session.
                The spike data is stored in the following keys:
                    - spike_times: A list of spike times in seconds
                    - spike_clusters: A list of cluster IDs for each spike
                    - spike_trial: A list of trial IDs for each spike
                    - sampling_rate: The sampling rate of the spike data
        """

        if self.analyzer is None:
            self._load_ephys(from_disk=from_disk)

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

        self.spike_data["templates"] = (
            self._load_templates(unit_ids) if load_templates else None
        )
        self.spike_data["waveforms"] = (
            self._load_waveforms(unit_ids) if load_waveforms else None
        )
        self.spike_data["channels"] = (
            self._load_extremum_channels(unit_ids) if load_channels else None
        )

    def load_single_unit_spike_trains(self, unit_ids=None, sparse=True):
        """
        Returns the spike trains from all trials for "good" units
        Format is {trial: {unit: spike_train}}
        """
        if self.analyzer is None:
            self._load_ephys(keep_good_only=True, sparse=sparse)

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

        Returns the cluster IDs that passed pyramidal cell filtering criteria:
        - Mean FR < 10 Hz
        - Mean spike width > 500us
        - Burst index < 25ms

        Returns:
        --------
        np.ndarray : Array of cluster IDs identified as pyramidal cells
                     Returns empty array if file not found
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

        Returns the cluster IDs that showed significant spatial selectivity based on:
        - Spatial information (bits/spike)
        - Statistical testing with population shuffles
        - Multiple comparison correction

        Returns:
        --------
        np.ndarray : Array of cluster IDs identified as place cells
                     Returns empty array if file not found
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
        from_preprocessed=True,
    ):
        """
        Loads the LFP (Local Field Potential) data for specified trials.
        Currently from raw Dacq .bin files using the spikeinterface package
        Masks clipped values and scales to microvolts based on the gain in the .set file

        Args:
            trial_list: The index of the trial(s) to load. Default is all
            sampling_rate: The desired sampling rate for the LFP data
            channels: A list of channel IDs to load. If None, loads all channels.
                    If specified, subsets data to only those channels.
            reload_flag (bool, optional): if true, forces reloading of data.
                If false, only loads data for trials with no LFP data loaded
            bandpass_filter: apply bandpass filter with min and max frequency.
            from_preprocessed (bool): If True, loads preprocessed LFP data from disk.

        Populates:
            self.lfp_data (list): A list that stores LFP data for each trial.
                The LFP data for the specified trial is added at the given index.
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

            # Try to load from preprocessed data first (if enabled and file exists)
            if from_preprocessed and lfp_path.exists():
                try:
                    lfp_data = load_pickle(lfp_path)

                    # Validate parameters against saved data
                    is_valid, reason = validate_lfp_cache(
                        lfp_data, sampling_rate, channels, bandpass_filter
                    )

                    if not is_valid:
                        raise ValueError(
                            f"Trial {trial_iterator}: {reason}. "
                            "Use from_preprocessed=False to reload with new parameters."
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

                    print(f"Loaded preprocessed LFP data for trial {trial_iterator}")
                    continue
                except Exception as e:
                    print(
                        f"Error loading preprocessed LFP for trial "
                        f"{trial_iterator}: {e}"
                    )
                    print("Falling back to raw data loading...")

            # Load from raw data
            if self.analyzer is None:
                self._load_ephys(sparse=False)

            path = self.recording_path / self.trial_list[trial_iterator]
            self.log_loading_info(path, "lfp", True)

            temp_folder = self.recording_path / "temp"

            try:
                lfp_data = load_lfp_data(
                    recording=self.raw_recording,
                    segment_index=trial_iterator,
                    sampling_rate=sampling_rate,
                    recording_type=self.recording_type,
                    temp_folder=temp_folder,
                    channels=channels,
                    bandpass_filter=bandpass_filter,
                )

                self.lfp_data[trial_iterator] = lfp_data

                # Save to disk for future loading (only if requested)
                if from_preprocessed:
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

    def load_ttl(
        self,
        trial_list: int | list[int] | None = None,
        output_flag=True,
        reload_flag=False,
        from_preprocessed=True,
    ):
        """
        Load TTL data for specified trials from OpenEphys recording

        Args:
            trial_list: The index of the trial(s) for which TTL data is to be loaded.
            output_flag: if True, print a statement when loading the TTL data
            reload_flag: if True, forces reloading of data. If false, only loads data
                for trials with no TTL data loaded
            from_preprocessed (bool): If True, loads preprocessed TTL data from disk.

        Populates:
            self.sync_data (list): A list that stores TTL data for each trial.
                The TTL data for the specified trial is added at the given index.
        """
        from .loaders.cache import load_pickle, save_pickle
        from .loaders.ttl_loader import load_ttl_data

        if self.recording_type != "NP2_openephys":
            print("TTL data only available for NP2_openephys recordings")
            return

        trial_list = self.normalize_trial_list(trial_list, "ttl")

        for trial_iterator in trial_list:
            if not self.should_load_data(trial_iterator, "ttl", reload_flag):
                continue

            ttl_path = self.recording_path / f"ttl_data_trial{trial_iterator}.pkl"

            # Try to load from preprocessed data first (if enabled and file exists)
            if from_preprocessed and ttl_path.exists():
                try:
                    ttl_data = load_pickle(ttl_path)
                    self.sync_data[trial_iterator] = ttl_data
                    if output_flag:
                        print(
                            f"Loaded preprocessed TTL data for trial {trial_iterator}"
                        )
                    continue
                except Exception as e:
                    if output_flag:
                        print(f"Error loading TTL for trial {trial_iterator}: {e}")
                        print("Falling back to raw data loading...")

            # Load from raw data
            path = self.recording_path / self.trial_list[trial_iterator]
            self.log_loading_info(path, "TTL", output_flag)

            # Ensure analyzer is loaded to get recording start time
            if not self.analyzer:
                self._load_ephys(sparse=False)

            recording_start_time = self.analyzer.recording.get_start_time(
                segment_index=trial_iterator
            )

            ttl_data = load_ttl_data(path, recording_start_time)
            self.sync_data[trial_iterator] = ttl_data

            # Save to cache if requested
            if from_preprocessed:
                try:
                    save_pickle(ttl_data, ttl_path)
                    if output_flag:
                        print(
                            f"Saved TTL data for trial {trial_iterator} to {ttl_path}"
                        )
                except Exception as e:
                    print(f"Warning: Could not save TTL data to disk: {e}")

    def load_theta_phase(
        self,
        trial_list: int | list[int] | None = None,
        channels: list[int] | None = None,
        clip_value: int | None = 32000,
        output_flag: bool = True,
        reload_flag: bool = False,
        from_preprocessed: bool = True,
    ):
        """
        Load theta phase data for specified trials and append to existing LFP data.

        Args:
            trial_list: The index of the trial(s) to load. Default is all
            channels: A list of channel IDs to use. If None, uses all available channels
                    If specified, subsets data to only those channels.
            clip_value: Clipping value for LFP data (32000 for Axona, None for others)
            output_flag: If True, print loading information
            reload_flag: If True, forces reloading of data
            from_preprocessed: If True, loads preprocessed theta phase data from disk

        Modifies:
            self.lfp_data[trial]['theta_phase']: Phase values for each channel
            self.lfp_data[trial]['cycle_numbers']: Cycle numbers for each channel
            self.lfp_data[trial]['theta_freqs']: Peak theta frequencies for each channel

        Also subsets all LFP data to requested channels if specified.
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

            # Try to load from preprocessed data first (if enabled and file exists)
            if from_preprocessed and theta_phase_path.exists():
                try:
                    saved_theta_data = load_pickle(theta_phase_path)

                    # Ensure LFP data is loaded
                    if self.lfp_data[trial_iterator] is None:
                        if output_flag:
                            print(f"Loading LFP data for trial {trial_iterator}")
                        self.load_lfp(
                            trial_list=[trial_iterator], from_preprocessed=True
                        )

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
                            f"Loaded preprocessed theta phase data for trial "
                            f"{trial_iterator}{channel_info}"
                        )
                    continue

                except Exception as e:
                    if output_flag:
                        print(
                            f"Error loading preprocessed theta phase for trial "
                            f"{trial_iterator}: {e}"
                        )
                        print("Falling back to raw data processing...")

            # Load/process from raw data
            if output_flag:
                print(f"Processing theta phase data for trial {trial_iterator}")

            # Check if LFP data is loaded for this trial
            if self.lfp_data[trial_iterator] is None:
                if output_flag:
                    print(f"Loading LFP data for trial {trial_iterator}")
                self.load_lfp(
                    trial_list=[trial_iterator],
                    channels=channels,
                    from_preprocessed=True,
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

                # Save theta phase data to disk for future loading if requested
                if from_preprocessed:
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

    def _load_ephys(self, keep_good_only=False, sparse=True, from_disk=True):
        """
        Make a SortingAnalyzer for extracting spikes and LFP
        """
        # Load sorting analyzer from disk if it exists. Raw recording will not be loaded
        # Makes a copy of the disk analyzer to avoid modifying the original
        if from_disk:
            if (self.recording_path / "sorting_analyzer.zarr").exists():
                disk_analyzer = si.load_sorting_analyzer(
                    self.recording_path / "sorting_analyzer.zarr"
                )
                self.analyzer = disk_analyzer.copy()
                # self.raw_recording = self.analyzer.recording
                return
            else:
                raise FileNotFoundError(
                    "Sorting analyzer not found on disk. "
                    "Please run preprocessing script to create it."
                )

        recording_list = []
        # Create list of recording objects
        for trial_iterator in self.trial_iterators:
            if self.recording_type == "nexus":
                path = self.recording_path / f"{self.trial_list[trial_iterator]}.set"
                recording = load_axona_ephys(path, self.probe_type)

            elif self.probe_type == "NP2_openephys":
                path = self.recording_path / self.trial_list[trial_iterator] / self.area
                if self.recording_type == "NP2_openephys":
                    recording = load_np2_pcie(path)
                elif self.recording_type == "NP2_onebox":
                    recording = load_np2_onebox(path)

            else:
                raise ValueError(
                    f"Recording type {self.recording_type}"
                    " or probe type {self.probe_type} not implementes"
                )

            recording_list.append(recording)

        # Check if sorting has been manually curated in Phy
        cluster_info_path = Path(self.sorting_path) / "cluster_info.tsv"
        if not cluster_info_path.exists():
            raise FileNotFoundError(
                f"Manual curation required: cluster_info.tsv not found at "
                f"{self.sorting_path}.\n\n"
                f"Please complete these steps:\n"
                f"  1. Open the sorting in Phy GUI:\n"
                f"     phy template-gui {self.sorting_path}\n"
                f"  2. Manually curate units (label as 'good', 'mua', or 'noise')\n"
                f"  3. Save your curation (Ctrl+S or File → Save)\n"
                f"  4. Close Phy\n"
                f"  5. Re-run this preprocessing step\n\n"
                f"Phy will create cluster_info.tsv when you save your curation."
            )

        # Load sorting
        if keep_good_only:
            sorting = se.read_phy(
                f"{self.sorting_path}", exclude_cluster_groups=["noise", "mua"]
            )
        else:
            sorting = se.read_phy(f"{self.sorting_path}")

        # Check if any units remain after filtering
        if sorting.get_num_units() == 0:
            filter_msg = " after filtering for good units" if keep_good_only else ""
            raise ValueError(
                f"No units found in sorting data{filter_msg}. "
                f"Sorting path: {self.sorting_path}. "
                "Please check that spike sorting has been performed and "
                "units have been curated. "
                "If using keep_good_only=True, ensure at least one unit is "
                "labeled as 'good' in Phy."
            )

        multi_segment_sorting = si.split_sorting(sorting, recording_list)

        multi_segment_recording = si.append_recordings(recording_list)

        # Save raw recording for LFP extraction
        self.raw_recording = multi_segment_recording

        # Highpass filter recording
        multi_segment_recording: si.BaseRecording = spre.highpass_filter(
            multi_segment_recording, 300
        )

        # Make a single multisegment SortingAnalyzer for the whole session
        self.analyzer = si.create_sorting_analyzer(
            multi_segment_sorting,
            multi_segment_recording,
            sparse=sparse,
            format="zarr",
            folder=self.recording_path / "sorting_analyzer",
            return_scaled=True,
            overwrite=True,
        )

    def _load_templates(self, clusters_to_load=None):
        if not self.analyzer.has_extension("waveforms"):
            self.analyzer.compute(["random_spikes", "waveforms"])

        if not self.analyzer.has_extension("templates"):
            self.analyzer.compute("templates")

        templates = self.analyzer.get_extension("templates").get_data()
        if clusters_to_load is not None:
            templates = templates[clusters_to_load]
        return templates

    def _load_waveforms(self, clusters_to_load=None):
        if not self.analyzer.has_extension("waveforms"):
            self.analyzer.compute(["random_spikes", "waveforms"])

        waveforms = self.analyzer.get_extension("waveforms").get_data(outputs="by_unit")
        if clusters_to_load is not None:
            waveforms = waveforms[clusters_to_load]
        return waveforms

    def _load_extremum_channels(self, clusters_to_load=None):
        # try loading from sorting properties
        try:
            extremum_channels = self.analyzer.sorting.get_property("ch")
            unit_ids = self.analyzer.sorting.get_unit_ids()
            # make extremum channels a dict
            extremum_channels = dict(zip(unit_ids, extremum_channels))
        except KeyError:
            if not self.analyzer.has_extension("waveforms"):
                self.analyzer.compute(["random_spikes", "waveforms"])
            if not self.analyzer.has_extension("templates"):
                self.analyzer.compute("templates")

            extremum_channels = si.get_template_extremum_channel(self.analyzer)
        # if clusters_to_load is specified, only return those
        if clusters_to_load is not None:
            extremum_channels = {k: extremum_channels[k] for k in clusters_to_load}
        return extremum_channels
