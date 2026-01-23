from typing import Literal

import numpy as np

from spelt.analysis.spatial import spatial_significance


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
        alpha: float = 0.05,
    ) -> None:
        """Calculates spatial significance.

        Parameters
        ----------
        pos_sample_times : np.ndarray
            Array of time points at which position data was sampled
        pos_bin_idx : tuple[np.ndarray, np.ndarray]
            Tuple of arrays containing bin indices for position data
        pos_sampling_rate : float
            Sampling rate of position data in Hz
        abs_threshold : float
            Absolute threshold for spatial information (bits per spike)
            Must be > 0. Typically set to 95th percentile of population shuffles
        alpha : float, optional
            Significance threshold for p-value (default: 0.05)

        Raises
        ------
        ValueError
            If abs_threshold is <= 0 or None
        """
        if abs_threshold is None or abs_threshold <= 0:
            raise ValueError(
                f"abs_threshold must be a positive number, got {abs_threshold}. "
                "This threshold should be calculated from population shuffles. "
                "See documentation for generating population shuffle data."
            )

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
                self.p_values[cluster] < alpha
                and self.bits_per_spike[cluster] > abs_threshold
            ):
                self.include.append(cluster)

    @staticmethod
    def aggregate_significance_across_trials(
        trial_data_list: list["TrialData"],
        correction_method: Literal["bonferroni", "fdr_bh", "none"] = "fdr_bh",
        alpha: float = 0.05,
        min_significant_trials: int = 1,
    ) -> dict[int, dict]:
        """Aggregate significance results across multiple trials.

        Applies multiple comparison correction across trials and determines
        which cells are significant after correction.

        Parameters
        ----------
        trial_data_list : list[TrialData]
            List of TrialData objects from the same session
        correction_method : {'bonferroni', 'fdr_bh', 'none'}, optional
            Method for multiple comparison correction:
            - 'bonferroni': Conservative Bonferroni correction
            - 'fdr_bh': Benjamini-Hochberg FDR correction (default)
            - 'none': No correction (use with caution)
        alpha : float, optional
            Significance threshold (default: 0.05)
        min_significant_trials : int, optional
            Minimum number of trials that must be significant (default: 1)

        Returns
        -------
        dict[int, dict]
            Dictionary mapping cluster IDs to significance info:
            {
                cluster_id: {
                    'raw_p_values': list of p-values per trial,
                    'corrected_p_values': list of corrected p-values per trial,
                    'is_significant': bool (after correction and min_trials check),
                    'n_significant_trials': int,
                    'mean_p_value': float,
                    'correction_method': str
                }
            }
        """
        if not trial_data_list:
            return {}

        # Get all cluster IDs across trials
        all_clusters = set()
        for trial in trial_data_list:
            all_clusters.update(trial.p_values.keys())

        results = {}

        for cluster in all_clusters:
            # Collect p-values across trials for this cluster
            p_values = []
            for trial in trial_data_list:
                if cluster in trial.p_values:
                    p_val = trial.p_values[cluster]
                    # NaN values (insufficient spikes) are treated as not significant
                    if np.isnan(p_val):
                        p_values.append(1.0)
                    else:
                        p_values.append(p_val)
                else:
                    p_values.append(1.0)  # Not present = not significant

            p_values = np.array(p_values)
            n_trials = len(p_values)

            # Apply correction
            if correction_method == "bonferroni":
                corrected_p = p_values * n_trials
                corrected_p = np.minimum(corrected_p, 1.0)  # Cap at 1.0
            elif correction_method == "fdr_bh":
                # Benjamini-Hochberg FDR correction
                try:
                    from statsmodels.stats.multitest import multipletests

                    reject, corrected_p, _, _ = multipletests(
                        p_values, alpha=alpha, method="fdr_bh"
                    )
                except ImportError:
                    # Fallback if statsmodels not available
                    print(
                        "Warning: statsmodels not available, using Bonferroni instead"
                    )
                    corrected_p = p_values * n_trials
                    corrected_p = np.minimum(corrected_p, 1.0)
            elif correction_method == "none":
                corrected_p = p_values
            else:
                raise ValueError(
                    f"Unknown correction method: {correction_method}. "
                    "Use 'bonferroni', 'fdr_bh', or 'none'."
                )

            # Determine significance
            n_sig = np.sum(corrected_p < alpha)
            is_significant = n_sig >= min_significant_trials

            results[cluster] = {
                "raw_p_values": p_values.tolist(),
                "corrected_p_values": corrected_p.tolist(),
                "is_significant": is_significant,
                "n_significant_trials": int(n_sig),
                "mean_p_value": float(np.mean(p_values)),
                "correction_method": correction_method,
            }

        return results
