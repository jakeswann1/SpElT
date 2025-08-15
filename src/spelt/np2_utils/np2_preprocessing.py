from pathlib import Path

import spikeinterface as si
import spikeinterface.curation as scur
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss


def sort_np2(recording, recording_name, base_folder, sorting_suffix, area):
    sorting_path = Path(f"{base_folder}/{recording_name[:6]}_{sorting_suffix}")

    if (sorting_path).is_dir():
        try:
            sorting = si.load_extractor(sorting_path / "sort")
            print(f"Sorting loaded from file {sorting_path}\n")
        except ValueError as e:
            print(
                f"""Sorting at {sorting_path} failed to load -
                try deleting the folder and rerun"""
            )
            raise ValueError from e
    else:
        # Account for phase shift in NP2 acquisition
        try:
            recording = spre.phase_shift(recording)
        except AssertionError as e:
            print(
                f"""Phase shift failed for {recording_name} -
                this is likely because the recording is already phase shifted.
                Error: {e}"""
            )
        # Sort
        sorting = ss.run_sorter(
            "kilosort4",
            recording,
            folder=f"{sorting_path}",
            verbose=True,
            docker_image=False,
            remove_existing_folder=True,
            save_preprocessed_copy=True,
            use_binary_file=True,
        )

        print(f"Recording sorted!\n KS4 found {len(sorting.get_unit_ids())} units\n")

        # Automated curation of sorting output
        sorting = sorting.remove_empty_units()
        sorting = scur.remove_duplicated_spikes(
            sorting, censored_period_ms=0.3, method="keep_first_iterative"
        )
        sorting = scur.remove_excess_spikes(sorting, recording)
        sorting = scur.remove_redundant_units(
            sorting, align=False, remove_strategy="max_spikes"
        )

        sorting.save(folder=sorting_path / "sort")
        print(f"Sorting saved to {sorting_path}/sort\n")

    return sorting
