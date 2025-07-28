import pickle as pkl
from pathlib import Path

import numpy as np
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre

from .utils import gs_to_df

si.set_global_job_kwargs(n_jobs=-1)


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
            self.session = session
        except Exception as e:
            print("Google Sheet not found, please specify a valid URL")
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
        elif self.recording_type == "NP2_openephys":
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
            self._load_ephys(from_disk=True)

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
        trial_list = self.normalize_trial_list(trial_list, "position")

        for trial_idx, trial_iterator in enumerate(trial_list):
            if not self.should_load_data(trial_iterator, "position", reload_flag):
                continue

            path = self.recording_path / self.trial_list[trial_iterator]
            self.log_loading_info(path, "position", output_flag)

            try:
                tracking_type = self.tracking_types[trial_idx]
                pos_data = self._load_pos_data_by_type(
                    tracking_type, path, trial_iterator, output_flag
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
            channels: A list of channel IDs to load. Default is all
            reload_flag (bool, optional): if true, forces reloading of data.
                If false, only loads data for trials with no LFP data loaded
            bandpass_filter: apply bandpass filter with min and max frequency.
            from_preprocessed (bool): If True, loads preprocessed LFP data from disk.

        Populates:
            self.lfp_data (list): A list that stores LFP data for each trial.
                The LFP data for the specified trial is added at the given index.
        """
        trial_list = self.normalize_trial_list(trial_list, "lfp")

        for trial_iterator in trial_list:
            if not self.should_load_data(trial_iterator, "lfp", reload_flag):
                continue

            lfp_path = self.recording_path / f"lfp_data_trial{trial_iterator}.pkl"

            # Try to load from preprocessed data first (if enabled and file exists)
            if from_preprocessed and lfp_path.exists():
                try:
                    with lfp_path.open("rb") as f:
                        lfp_data = pkl.load(f)  # noqa: S301

                    # Validate parameters against saved data
                    self._validate_lfp_parameters(
                        lfp_data,
                        trial_iterator,
                        sampling_rate,
                        channels,
                        bandpass_filter,
                    )

                    self.lfp_data[trial_iterator] = lfp_data
                    print(f"Loaded preprocessed LFP data for trial {trial_iterator}")
                    continue
                except Exception as e:
                    print(
                        f"Error loading preprocessed LFP for trial {trial_iterator}: {e}"
                    )
                    print("Falling back to raw data loading...")

            # Load from raw data (either because from_preprocessed=False, file doesn't exist, or loading failed)
            self._load_and_save_lfp_data(
                trial_iterator,
                sampling_rate,
                channels,
                bandpass_filter,
                save_to_disk=from_preprocessed,
            )

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
                    with ttl_path.open("rb") as f:
                        ttl_data = pkl.load(f)  # noqa: S301
                    self.sync_data[trial_iterator] = ttl_data
                    if output_flag:
                        print(
                            f"Loaded preprocessed TTL data for trial {trial_iterator}"
                        )
                    continue
                except Exception as e:
                    if output_flag:
                        print(
                            f"Error loading preprocessed TTL for trial {trial_iterator}: {e}"
                        )
                        print("Falling back to raw data loading...")

            # Load from raw data (either because from_preprocessed=False, file doesn't exist, or loading failed)
            self._load_and_save_ttl_data(
                trial_iterator, output_flag, save_to_disk=from_preprocessed
            )

    def _validate_lfp_parameters(
        self,
        saved_data: dict,
        trial_iterator: int,
        requested_sampling_rate: int,
        requested_channels: list | None,
        requested_filter: list[float, float] | None,
    ):
        """
        Validate that the requested parameters match the saved LFP data.

        Args:
            saved_data: The loaded LFP data dictionary.
            trial_iterator: The trial index being loaded.
            requested_sampling_rate: The requested sampling rate.
            requested_channels: The requested channels list.
            requested_filter: The requested bandpass filter range.

        Raises:
            ValueError: If parameters don't match the saved data.
        """
        # Check sampling rate (only validate if both are not None)
        saved_sampling_rate = saved_data.get("sampling_rate")
        if (
            requested_sampling_rate is not None
            and saved_sampling_rate is not None
            and saved_sampling_rate != requested_sampling_rate
        ):
            raise ValueError(
                f"Trial {trial_iterator}: Requested sampling rate ({requested_sampling_rate} Hz) "
                f"doesn't match saved data ({saved_sampling_rate} Hz). "
                f"Use from_preprocessed=False to reload with new parameters."
            )

        # Check filter range (only validate if both are not None)
        saved_filter = saved_data.get("filter_range")
        if (
            requested_filter is not None
            and saved_filter is not None
            and saved_filter != requested_filter
        ):
            raise ValueError(
                f"Trial {trial_iterator}: Requested filter range ({requested_filter}) "
                f"doesn't match saved data ({saved_filter}). "
                f"Use from_preprocessed=False to reload with new parameters."
            )

        # Check channels if specified
        if requested_channels is not None:
            saved_channels = saved_data.get("channels")

            # Convert requested channels to strings to match saved format
            requested_channels_str = [str(ch) for ch in requested_channels]

            # Check if saved_channels is None (all channels were saved)
            if saved_channels is not None:
                # Check if all requested channels are available in saved data
                missing_channels = set(requested_channels_str) - set(saved_channels)
                if missing_channels:
                    raise ValueError(
                        f"Trial {trial_iterator}: Requested channels {list(missing_channels)} "
                        f"not found in saved data (available: {saved_channels}). "
                        f"Use from_preprocessed=False to reload with new channels."
                    )

    def _load_and_save_lfp_data(
        self,
        trial_iterator: int,
        sampling_rate: int,
        channels: list | None,
        bandpass_filter: list[float, float] | None,
        save_to_disk: bool = True,
    ):
        """
        Helper method to load LFP data from raw files and optionally save to disk.

        Args:
            trial_iterator: The trial index to load.
            sampling_rate: The desired sampling rate for the LFP data.
            channels: A list of channel IDs to load.
            bandpass_filter: Apply bandpass filter with min and max frequency.
            save_to_disk: If True, save the loaded data to disk for future use.
        """
        if self.analyzer is None:
            self._load_ephys(sparse=False)

        recording = self.raw_recording

        # Resample
        recording = spre.resample(recording, sampling_rate)
        print("Resampled to", sampling_rate, "Hz")

        # Bandpass filter
        if bandpass_filter is not None:
            recording = spre.bandpass_filter(
                recording,
                freq_min=bandpass_filter[0],
                freq_max=bandpass_filter[1],
            )

        # AXONA ONLY: clip values of +- 32000
        if self.recording_type == "nexus":
            recording = spre.clip(recording, a_min=-32000, a_max=32000)

        # Set channels to load to list of str to match recording object
        #  - not ideal but other fixes are harder
        if channels is not None:
            channels = list(map(str, channels))

        path = self.recording_path / self.trial_list[trial_iterator]
        self.log_loading_info(path, "lfp", True)

        try:
            lfp_data = recording.get_traces(
                segment_index=trial_iterator,
                channel_ids=channels,
                return_scaled=True,
            ).astype(float)

            lfp_timestamps = recording.get_times(segment_index=trial_iterator)

            trial_lfp_data = {
                "data": lfp_data,
                "timestamps": lfp_timestamps,
                "sampling_rate": sampling_rate,
                "channels": channels,
                "filter_range": bandpass_filter,
            }

            self.lfp_data[trial_iterator] = trial_lfp_data

            # Save to disk for future loading (only if requested)
            if save_to_disk:
                lfp_path = self.recording_path / f"lfp_data_trial{trial_iterator}.pkl"
                try:
                    with lfp_path.open("wb") as f:
                        pkl.dump(trial_lfp_data, f)

                    # Print saved file size
                    file_size = lfp_path.stat().st_size
                    print(
                        f"Saved trial {trial_iterator} LFP data to {lfp_path} ({file_size / 1e6:.2f} MB)"
                    )
                except Exception as e:
                    print(f"Warning: Could not save LFP data to disk: {e}")

        except Exception as e:
            print(f"Error loading LFP data for trial {trial_iterator}: {str(e)}")
            self.lfp_data[trial_iterator] = None

    def _load_and_save_ttl_data(
        self, trial_iterator: int, output_flag: bool, save_to_disk: bool = True
    ):
        """
        Helper method to load TTL data from raw files and optionally save to disk.

        Args:
            trial_iterator: The trial index to load.
            output_flag: If True, print loading information.
            save_to_disk: If True, save the loaded data to disk for future use.
        """
        path = self.recording_path / self.trial_list[trial_iterator]
        self.log_loading_info(path, "ttl", output_flag)

        try:
            ttl_times = {
                "ttl_timestamps": se.read_openephys_event(path).get_event_times(
                    channel_id="Neuropixels PXI Sync"
                )
            }

            # Get time when recording started and rescale timestamps
            if not self.analyzer:
                self._load_ephys(sparse=False)

            recording_start_time = self.analyzer.recording.get_start_time(
                segment_index=trial_iterator
            )

            if ttl_times["ttl_timestamps"][0] - recording_start_time < 0:
                Warning(
                    f"Recording start time {recording_start_time} is later than "
                    f"the first TTL pulse {ttl_times['ttl_timestamps'][0]} "
                    f"setting first TTL pulse to 0"
                )
                ttl_times["ttl_timestamps"] = (
                    ttl_times["ttl_timestamps"] - ttl_times["ttl_timestamps"][0]
                )
            else:
                ttl_times["ttl_timestamps"] -= recording_start_time

            self.sync_data[trial_iterator] = ttl_times

            # Save to disk for future loading (only if requested)
            if save_to_disk:
                ttl_path = self.recording_path / f"ttl_data_trial{trial_iterator}.pkl"
                try:
                    with ttl_path.open("wb") as f:
                        pkl.dump(ttl_times, f)
                    if output_flag:
                        print(
                            f"Saved TTL data for trial {trial_iterator} to {ttl_path}"
                        )
                except Exception as e:
                    print(f"Warning: Could not save TTL data to disk: {e}")

        except Exception as e:
            print(f"Error loading TTL data for trial {trial_iterator}: {str(e)}")
            self.sync_data[trial_iterator] = {"ttl_timestamps": None}

    def _load_pos_data_by_type(
        self, tracking_type: str, path: Path, trial_iterator: int, output_flag: bool
    ) -> dict:
        """Load position data based on tracking type."""
        if tracking_type == "axona":
            return self._load_axona_pos_data(path, trial_iterator, output_flag)
        elif tracking_type in ["bonsai_roi", "bonsai_leds"]:
            return self._load_bonsai_pos_data(
                path, trial_iterator, tracking_type, output_flag
            )
        elif (path / "dlc.csv").exists():
            return self._load_dlc_pos_data(path, output_flag)
        else:
            raise ValueError(f"Unsupported tracking type: {tracking_type}")

    def _load_axona_pos_data(
        self, path: Path, trial_iterator: int, output_flag: bool
    ) -> dict:
        """Load position data from Axona format."""
        from .axona_utils.postprocess_pos_data import postprocess_pos_data

        override_ppm = 615 if "t-maze" in self.trial_list[trial_iterator] else None
        if override_ppm and output_flag:
            print("Real PPM artificially set to 615 (t-maze default)")

        raw_pos_data, pos_sampling_rate = self._load_axona_raw_data(
            path, override_ppm, output_flag, trial_iterator
        )

        xy_pos, led_pos, led_pix, speed, direction, direction_disp = (
            postprocess_pos_data(
                raw_pos_data, self.max_speed, self.smoothing_window_size
            )
        )

        # Rescale timestamps to seconds
        xy_pos.columns /= pos_sampling_rate
        led_pos.columns /= pos_sampling_rate
        led_pix.columns /= pos_sampling_rate

        return {
            "header": raw_pos_data.get("header"),
            "xy_position": xy_pos,
            "led_positions": led_pos,
            "led_pixel_size": led_pix,
            "speed": speed,
            "direction": direction,
            "direction_from_displacement": direction_disp,
            "pos_sampling_rate": pos_sampling_rate,
            "scaled_ppm": 400,
        }

    def _load_axona_raw_data(
        self,
        path: Path,
        override_ppm: int | None,
        output_flag: bool,
        trial_iterator: int,
    ) -> tuple:
        """Load raw Axona position data from various file formats."""
        from .axona_utils.axona_preprocessing import pos_from_bin
        from .axona_utils.load_pos_axona import load_pos_axona
        from .axona_utils.postprocess_pos_data import write_csv_from_pos

        try:
            return load_pos_axona(path, override_ppm)
        except FileNotFoundError:
            if output_flag:
                print("No .csv file found, trying to load from .bin file")
            try:
                pos_from_bin(path)
                return load_pos_axona(
                    path / self.trial_list[trial_iterator], override_ppm
                )
            except FileNotFoundError:
                if output_flag:
                    print("No .csv or .bin file found, trying to load from .pos file")
                write_csv_from_pos(path.with_suffix(".pos"))
                return load_pos_axona(path, override_ppm)

    def _load_bonsai_pos_data(
        self, path: Path, trial_iterator: int, tracking_type: str, output_flag: bool
    ) -> dict:
        """Load position data from Bonsai format."""

        if self.sync_data[trial_iterator] is None:
            self.load_ttl(trial_iterator, output_flag=False)

        ttl_times = self.sync_data[trial_iterator].get("ttl_timestamps")
        ttl_freq = (
            1 / np.mean(np.diff(ttl_times[2:])) if ttl_times is not None else None
        )

        if tracking_type == "bonsai_roi" and path.with_suffix(".csv").exists():
            return self._load_bonsai_roi_data(
                path, trial_iterator, ttl_times, ttl_freq, output_flag
            )
        elif tracking_type == "bonsai_leds" and path.with_suffix(".csv").exists():
            return self._load_bonsai_leds_data(path, trial_iterator, output_flag)
        else:
            raise FileNotFoundError(
                f"No Bonsai position data found for trial {trial_iterator}"
            )

    def _load_bonsai_roi_data(
        self,
        path: Path,
        trial_iterator: int,
        ttl_times: np.ndarray,
        ttl_freq: float,
        output_flag: bool,
    ) -> dict:
        """Load position data from Bonsai ROI format."""
        from .np2_utils.load_pos_bonsai import load_pos_bonsai_jake
        from .np2_utils.postprocess_pos_data_np2 import (
            postprocess_bonsai_jake,
            sync_bonsai_jake,
        )

        if output_flag:
            print(f"Loading raw Bonsai position data from path {path}")

        trial_type = self.session["Trial Type"].iloc[trial_iterator]
        try:
            raw_pos_data = load_pos_bonsai_jake(
                path.with_suffix(".csv"), 400, trial_type
            )
        except FileNotFoundError:
            path = path.with_suffix(".csv").replace("t-maze", "T-maze")
            if output_flag:
                print(f"Looking for Bonsai file with name {path}")
            raw_pos_data = load_pos_bonsai_jake(path, 400, trial_type)

        xy_pos, speed, direction_disp = postprocess_bonsai_jake(
            raw_pos_data, self.max_speed, self.smoothing_window_size
        )

        pos_sampling_rate = 1 / np.mean(np.diff(ttl_times))
        xy_pos, speed, direction_disp = sync_bonsai_jake(
            xy_pos, ttl_times, pos_sampling_rate, speed, direction_disp
        )

        return {
            "xy_position": xy_pos,
            "speed": speed,
            "direction_from_displacement": direction_disp,
            "ttl_times": ttl_times,
            "ttl_freq": ttl_freq,
            "pos_sampling_rate": pos_sampling_rate,
            "scaled_ppm": 400,
        }

    def _load_bonsai_leds_data(
        self, path: Path, trial_iterator: int, output_flag: bool
    ) -> dict:
        """Load position data from Bonsai LEDs format."""
        from .np2_utils.load_pos_bonsai import load_pos_bonsai_isa
        from .np2_utils.postprocess_pos_data_np2 import postprocess_bonsai_jake

        if output_flag:
            print("Loading raw Bonsai position data (Isa format)")

        trial_type = self.session["Trial Type"].iloc[trial_iterator]
        raw_pos_data = load_pos_bonsai_isa(path.with_suffix(".csv"), 400, trial_type)

        xy_pos, speed, direction_disp = postprocess_bonsai_jake(
            raw_pos_data, self.max_speed, self.smoothing_window_size
        )

        return {
            "xy_position": xy_pos,
            "speed": speed,
            "direction_from_displacement": direction_disp,
            "pos_sampling_rate": raw_pos_data["sampling_rate"],
            "scaled_ppm": 400,
        }

    def _load_dlc_pos_data(self, path: Path, output_flag: bool) -> dict:
        """Load position data from DeepLabCut format."""
        from .np2_utils.load_pos_dlc import load_pos_dlc
        from .np2_utils.postprocess_pos_data_np2 import postprocess_dlc_data

        if output_flag:
            print("Loading DLC position data")

        raw_pos_data = load_pos_dlc(path, 400)
        raw_pos_data["header"]["tracked_point_angle_1"] = 0

        xy_pos, tracked_points, speed, direction, direction_disp = postprocess_dlc_data(
            raw_pos_data, self.max_speed, self.smoothing_window_size
        )

        return {
            "header": raw_pos_data["header"],
            "xy_position": xy_pos,
            "tracked_points": tracked_points,
            "speed": speed,
            "direction": direction,
            "direction_from_displacement": direction_disp,
            "bonsai_timestamps": raw_pos_data["bonsai_timestamps"],
            "camera_timestamps": raw_pos_data["camera_timestamps"],
            "scaled_ppm": 400,
        }

    def _load_ephys(self, keep_good_only=False, sparse=True, from_disk=True):
        """
        Make a SortingAnalyzer for extracting spikes and LFP
        """
        import probeinterface as pi

        # Load sorting analyzer from disk if it exists. Raw recording will not be loaded
        if from_disk:
            if (self.recording_path / "sorting_analyzer.zarr").exists():
                self.analyzer = si.load_sorting_analyzer(
                    self.recording_path / "sorting_analyzer.zarr"
                )
            else:
                raise FileNotFoundError(
                    "Sorting analyzer not found on disk. "
                    "Please run preprocessing script to create it."
                )

            return

        recording_list = []
        # Create list of recording objects
        for trial_iterator in self.trial_iterators:

            if self.recording_type == "nexus":
                path = self.recording_path / f"{self.trial_list[trial_iterator]}.set"
                probe_path = Path(__file__).parent / "axona_utils" / "probes"
                recording = se.read_axona(path, all_annotations=True)
                if self.probe_type == "5x12_buz":
                    probe = pi.read_prb(probe_path / "5x12-16_buz.prb").probes[0]
                else:
                    print("Axona probe type not implemented in _load_ephys")
                recording = recording.set_probe(probe)

            elif self.recording_type == "NP2_openephys":
                path = self.recording_path / self.trial_list[trial_iterator] / self.area
                if not path.exists():
                    path = self.recording_path / self.trial_list[trial_iterator]
                recording = se.read_openephys(path, stream_id="0", all_annotations=True)
            recording_list.append(recording)

        # Load sorting
        if keep_good_only:
            sorting = se.read_phy(
                f"{self.sorting_path}", exclude_cluster_groups=["noise", "mua"]
            )
        else:
            sorting = se.read_phy(f"{self.sorting_path}")

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
            overwrite=False,
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
