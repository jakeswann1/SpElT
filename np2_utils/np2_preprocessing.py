from pathlib import Path
import spikeinterface as si
import spikeinterface.sorters as ss

def sort_np2(recording, recording_name, base_folder, sorting_suffix, area):

    sorting_path = Path(f'{base_folder}/{recording_name[:6]}_{sorting_suffix}')

    if (sorting_path).is_dir():
        try:
            sorting = si.load_extractor(sorting_path / 'sort')
            print(f"Sorting loaded from file {sorting_path}\n")
        except ValueError:
            print(f"Sorting at {sorting_path} failed to load - try deleting the folder and rerun")
            raise ValueError
    else:
        # Sort
        sorting = ss.run_sorter('kilosort4', recording, output_folder=f'{sorting_path}',
                                        verbose = True, docker_image = False, remove_existing_folder = True)

        print(f'Recording sorted!\n KS4 found {len(sorting.get_unit_ids())} units\n')

        sorting = sorting.remove_empty_units()      
        sorting.save(folder=sorting_path / 'sort')
        print(f'Sorting saved to {sorting_path}/sort\n')

        #Edit params.py file to point to new .dat file in the first line
        with open(f'{sorting_path}/sorter_output/params.py', 'r+') as file:
            lines = file.readlines()
            file.seek(0)
            file.write(f'dat_path = "{base_folder}/concat_{area}.dat"\n')
            file.writelines(lines[1:])
            file.truncate()
            

        