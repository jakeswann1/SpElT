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

    def get_data(self, key: str) -> Any:
        """
        Extract data for a given key from the MATLAB .mat file,
        preserving the original nested structure

        Args:
            key: The key to extract data for

        Returns:
            The extracted data, maintaining the original structure

        Raises:
            KeyError: If the key is not found in the file
        """
        if key not in self.mat_file:
            raise KeyError(f"Key '{key}' not found in the .mat file")

        # Get the dataset or group
        obj = self.mat_file[key]

        # Extract data while preserving structure
        return self._extract_data_from_object(obj)
