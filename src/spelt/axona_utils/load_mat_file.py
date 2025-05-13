from pathlib import Path
from typing import Any

import h5py
import numpy as np


class EphysMat:
    def __init__(self, file_path: str | Path):
        """
        Initialize with a path to a MATLAB .mat file and load it as an h5py.File object

        Args:
            file_path: Path to the .mat file
        """
        self.file_path = Path(file_path)
        self.mat_file = h5py.File(self.file_path, "r")

    def close(self) -> None:
        """Close the h5py file handle"""
        if hasattr(self, "mat_file") and self.mat_file:
            self.mat_file.close()

    def __del__(self) -> None:
        """Ensure file is closed when object is deleted"""
        self.close()

    def __enter__(self) -> "EphysMat":
        """Support context manager protocol"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close file when exiting context"""
        self.close()

    def _resolve_reference(self, ref: Any) -> Any:
        """
        Resolve a single HDF5 object reference

        Args:
            ref: An HDF5 object reference

        Returns:
            The dereferenced data
        """
        if isinstance(ref, h5py.Reference):
            obj = self.mat_file[ref]
            return self._extract_data_from_object(obj)
        else:
            return np.array(ref)

    def _extract_data_from_object(self, obj: Any) -> Any:
        """
        Extract data from an HDF5 object, maintaining the original structure

        Args:
            obj: An HDF5 object (Group, Dataset, etc.)

        Returns:
            The extracted data, preserving the original nested structure
        """
        if isinstance(obj, h5py.Dataset):
            # For regular numeric datasets, return as numpy array
            if obj.dtype.kind != "O":
                return np.array(obj)

            # For object datasets (typically cell arrays), handle each element
            if obj.ndim == 0:  # Scalar object
                return self._resolve_reference(obj[()])

            # Create a list or nested list structure based on the dataset dimensions
            result = []

            # For 1D cell arrays
            if obj.ndim == 1:
                for i in range(obj.shape[0]):
                    result.append(self._resolve_reference(obj[i]))
                return result

            # For 2D cell arrays (common in MATLAB)
            elif obj.ndim == 2:
                # MATLAB typically uses column-major order, so transpose if needed
                # Check if it's a row or column vector
                if obj.shape[0] == 1:  # Row vector
                    for i in range(obj.shape[1]):
                        result.append(self._resolve_reference(obj[0, i]))
                    return result
                elif obj.shape[1] == 1:  # Column vector
                    for i in range(obj.shape[0]):
                        result.append(self._resolve_reference(obj[i, 0]))
                    return result
                else:  # True 2D array
                    for i in range(obj.shape[0]):
                        row = []
                        for j in range(obj.shape[1]):
                            row.append(self._resolve_reference(obj[i, j]))
                        result.append(row)
                    return result

            # For higher-dimensional cell arrays (less common)
            else:
                it = np.nditer(obj, flags=["multi_index", "refs_ok"])
                nested_result = np.empty(obj.shape, dtype=object)
                for x in it:
                    idx = it.multi_index
                    nested_result[idx] = self._resolve_reference(x[()])

                # Convert to nested lists
                return self._ndarray_to_nested_lists(nested_result)

        elif isinstance(obj, h5py.Group):
            # For groups, create a dictionary
            result = {}
            for key in obj.keys():
                result[key] = self._extract_data_from_object(obj[key])
            return result

        else:
            # Fallback for other types
            return np.array(obj)

    def _ndarray_to_nested_lists(self, arr: np.ndarray) -> list:
        """
        Convert a numpy ndarray of objects to nested Python lists

        Args:
            arr: numpy ndarray containing objects

        Returns:
            Nested Python lists representing the same structure
        """
        if arr.ndim == 1:
            return list(arr)
        else:
            return [self._ndarray_to_nested_lists(arr[i]) for i in range(arr.shape[0])]

    def get_data(self, keys: str | list[str]) -> Any:
        """
        Extract data for a given key or list of keys from the MATLAB .mat file,
        preserving the original nested structure

        Args:
            keys: A single key (str) or a list of keys to extract data for

        Returns:
            If keys is a string: The extracted data for that key
            If keys is a list: A dictionary mapping each key to its extracted data

        Raises:
            KeyError: If any key is not found in the file
        """
        # Handle single key
        if isinstance(keys, str):
            if keys not in self.mat_file:
                raise KeyError(f"Key '{keys}' not found in the .mat file")

            obj = self.mat_file[keys]
            return self._extract_data_from_object(obj)

        # Handle list of keys
        elif isinstance(keys, list):
            result = {}
            missing_keys = [key for key in keys if key not in self.mat_file]

            if missing_keys:
                if len(missing_keys) == 1:
                    raise KeyError(
                        f"Key '{missing_keys[0]}' not found in the .mat file"
                    )
                else:
                    raise KeyError(f"Keys {missing_keys} not found in the .mat file")

            for key in keys:
                obj = self.mat_file[key]
                result[key] = self._extract_data_from_object(obj)

            return result

        else:
            raise TypeError("'keys' must be a string or a list of strings")

    def get_all_data(self) -> dict[str, Any]:
        """
        Extract data for all keys in the MATLAB .mat file

        Returns:
            A dictionary mapping each key in the .mat file to its extracted data

        Note:
            This method may be slow and memory-intensive for large .mat files,
            as it loads all data into memory
        """
        # Get all top-level keys in the .mat file
        all_keys = list(self.mat_file.keys())

        # Handle empty file case
        if not all_keys:
            return {}

        # Use existing get_data method to load all keys
        return self.get_data(all_keys)

    def list_keys(self) -> list[str]:
        """
        List all top-level keys available in the .mat file

        Returns:
            A list of all keys in the .mat file
        """
        return list(self.mat_file.keys())

    def get_data_at_index(
        self, key: str, indices: list[int] | int | tuple[int, ...] | slice
    ) -> Any:
        """
        Extract data for a specific key at given indices only.

        Args:
            key: The key to extract data for
            indices: An index, list of indices, tuple of indices, or slice to extract
                For 1D arrays: single index or slice
                For multi-dimensional: tuple of indices for each dimension

        Returns:
            The extracted data at the specified indices

        Raises:
            KeyError: If the key is not found in the file
            IndexError: If the indices are out of bounds
            TypeError: If the indices are not compatible with the data structure
        """
        if key not in self.mat_file:
            raise KeyError(f"Key '{key}' not found in the .mat file")

        obj = self.mat_file[key]

        # Handle direct datasets (non-object type)
        if isinstance(obj, h5py.Dataset) and obj.dtype.kind != "O":
            # For regular numeric datasets, return slice directly
            try:
                if isinstance(indices, (int, slice)):
                    return np.array(obj[indices])
                else:
                    return np.array(obj[indices])
            except (IndexError, TypeError) as e:
                raise type(e)(
                    f"Invalid indices {indices} for dataset shape {obj.shape}: {e}"
                ) from obj

        # Handle object datasets (typically cell arrays)
        elif isinstance(obj, h5py.Dataset) and obj.dtype.kind == "O":
            try:
                # Get only the references we need
                if isinstance(indices, int):
                    # Single index for 1D array
                    if obj.ndim == 1:
                        return self._resolve_reference(obj[indices])
                    # Single index for column vector
                    elif obj.ndim == 2 and obj.shape[1] == 1:
                        return self._resolve_reference(obj[indices, 0])
                    # Single index for row vector
                    elif obj.ndim == 2 and obj.shape[0] == 1:
                        return self._resolve_reference(obj[0, indices])
                    else:
                        raise TypeError(
                            f"""
                            Single integer index not valid for
                            {obj.ndim}D array of shape {obj.shape}
                            """
                        )

                elif isinstance(indices, tuple) and all(
                    isinstance(i, int) for i in indices
                ):
                    # Tuple of indices for each dimension
                    if len(indices) != obj.ndim:
                        raise TypeError(
                            f"""Expected {obj.ndim} indices for
                            {obj.ndim}D array, got {len(indices)}"""
                        )
                    return self._resolve_reference(obj[indices])

                elif isinstance(indices, slice):
                    # Slice for 1D array
                    if obj.ndim == 1:
                        return [
                            self._resolve_reference(obj[i])
                            for i in range(*indices.indices(obj.shape[0]))
                        ]
                    # Slice for column vector
                    elif obj.ndim == 2 and obj.shape[1] == 1:
                        return [
                            self._resolve_reference(obj[i, 0])
                            for i in range(*indices.indices(obj.shape[0]))
                        ]
                    # Slice for row vector
                    elif obj.ndim == 2 and obj.shape[0] == 1:
                        return [
                            self._resolve_reference(obj[0, i])
                            for i in range(*indices.indices(obj.shape[1]))
                        ]
                    else:
                        raise TypeError(
                            f"""Slice not supported for {obj.ndim}D
                            array of shape {obj.shape}"""
                        )

                elif isinstance(indices, list):
                    # List of indices for 1D array
                    if obj.ndim == 1:
                        return [self._resolve_reference(obj[i]) for i in indices]
                    # List of indices for column vector
                    elif obj.ndim == 2 and obj.shape[1] == 1:
                        return [self._resolve_reference(obj[i, 0]) for i in indices]
                    # List of indices for row vector
                    elif obj.ndim == 2 and obj.shape[0] == 1:
                        return [self._resolve_reference(obj[0, i]) for i in indices]
                    else:
                        raise TypeError(
                            f"""List indices not supported for
                            {obj.ndim}D array of shape {obj.shape}"""
                        )

                else:
                    raise TypeError(f"Unsupported index type: {type(indices)}")

            except (IndexError, TypeError) as e:
                raise type(e)(
                    f"Invalid indices {indices} for dataset of shape {obj.shape}: {e}"
                ) from obj

        # Handle groups (structs in MATLAB)
        elif isinstance(obj, h5py.Group):
            raise TypeError(
                f"""Key '{key}' refers to a group/struct, not a dataset.
                Use get_data() instead."""
            )

        else:
            # Fallback for other types
            raise TypeError(f"Unsupported object type: {type(obj)}")
