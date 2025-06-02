import numpy as np

from spelt.analysis.spatial_significance import spatial_significance


class TrialData:
    def __init__(
        self,
        rate_maps: dict,
        pos_map: np.ndarray,
        max_rates: dict,
        mean_rates: dict,
        spike_times: dict,
        age: int,
        trial_name: str,
    ):
        self.rate_maps = rate_maps
        self.pos_map = pos_map
        self.max_rates = max_rates
        self.mean_rates = mean_rates
        self.spike_times = spike_times
        self.age = age
        self.trial_name = trial_name
        self.bits_per_spike: dict | None = None
        self.bits_per_sec: dict | None = None
        self.p_values: dict = {}
        self.z_scores: dict = {}
        self.bps_shuffled: dict = {}
        self.include: list = []

    def calculate_significance(
        self,
        pos_sample_times: np.ndarray,
        pos_bin_idx: tuple[np.ndarray, np.ndarray],
        pos_sampling_rate: float,
        abs_threshold: float,
    ) -> None:
        """Calculates spatial significance."""
        for cluster in self.rate_maps.keys():
            (
                self.p_values[cluster],
                self.z_scores[cluster],
                self.bps_shuffled[cluster],
            ) = spatial_significance(
                pos_sample_times,
                pos_bin_idx,
                pos_sampling_rate,
                self.spike_times[cluster],
                n_shuffles=1000,
            )
            if (
                self.p_values[cluster] < 0.05
                and self.bits_per_spike[cluster] > abs_threshold
            ):
                self.include.append(cluster)
