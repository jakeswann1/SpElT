from pathlib import Path
import spikeinterface as si
import spikeinterface.sorters as ss
import spikeinterface.preprocessing as spre
import spikeinterface.curation as scur


def sort_np2(recording, recording_name, base_folder, sorting_suffix, area):

    sorting_path = Path(f"{base_folder}/{recording_name[:6]}_{sorting_suffix}")

    if (sorting_path).is_dir():
        try:
            sorting = si.load_extractor(sorting_path / "sort")
            print(f"Sorting loaded from file {sorting_path}\n")
        except ValueError:
            print(
                f"Sorting at {sorting_path} failed to load - try deleting the folder and rerun"
            )
            raise ValueError
    else:
        # Account for phase shift in NP2 acquisition
        recording = spre.phase_shift(recording)
        # Sort
        sorting = ss.run_sorter(
            "kilosort4",
            recording,
            output_folder=f"{sorting_path}",
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

        # #Edit params.py file to point to new .dat file in the first line
        # with open(f'{sorting_path}/sorter_output/params.py', 'r+') as file:
        #     lines = file.readlines()
        #     file.seek(0)
        #     file.write(f'dat_path = "{base_folder}/concat_{area}.dat"\n')
        #     file.writelines(lines[1:])
        #     file.truncate()
