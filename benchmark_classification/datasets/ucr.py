"""
UCR Archive Time Series Classification Dataset Loader for benchopt.

Loads UCR Archive datasets using tslearn and provides train/test splits
following the official UCR Archive train/test division.

Reference:
    https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
"""
import numpy as np
from pathlib import Path
from benchopt import BaseDataset
from benchopt.config import get_data_path


class Dataset(BaseDataset):
    """UCR Archive Time Series Classification dataset wrapper for benchopt.

    Parameters
    ----------
    dataset_name : str
        Name of the UCR dataset (e.g., "ECG5000", "FaceAll", "GunPoint").
        Dataset names are case-insensitive.
    debug : bool, optional
        If True, use only a subset of the data for fast iteration.
    """

    name = "UCR"

    # Default to a small dataset: ECG5000 (5000 training series, univariate)
    parameters = {
        "dataset_name": ["ECG5000"],
        "debug": [False],
    }

    def prepare(self):
        """Prepare data by downloading UCR datasets if needed."""
        try:
            from tslearn.datasets import UCR_UEA_datasets
        except ImportError:
            raise ImportError(
                "tslearn is required. Install with: pip install tslearn"
            )

        storage_path = Path(get_data_path("UCR"))
        storage_path.mkdir(parents=True, exist_ok=True)

        # The UCR_UEA_datasets utility will cache datasets locally
        # by default to the tslearn data directory, so no manual download needed

    def get_data(self):
        """Load the UCR dataset."""
        try:
            from tslearn.datasets import UCR_UEA_datasets
        except ImportError:
            raise ImportError(
                "tslearn is required. Install with: pip install tslearn"
            )

        # Load dataset using tslearn's built-in loader
        loader = UCR_UEA_datasets()
        
        try:
            X_train, y_train, X_test, y_test = loader.load_dataset(
                self.dataset_name
            )
        except Exception as e:
            raise ValueError(
                f"Could not load dataset '{self.dataset_name}'. "
                f"Make sure it exists in the UCR Archive. Error: {e}"
            )

        # Ensure we have numpy arrays
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int64)
        X_test = np.asarray(X_test, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.int64)

        # Normalize shape to (N, C, T) for downstream solvers.
        # tslearn returns UCR data as (N, T, C) for 3D arrays.
        if X_train.ndim == 2:
            X_train = X_train[:, np.newaxis, :]  # (N, 1, T)
        elif X_train.ndim == 3:
            X_train = X_train.transpose(0, 2, 1)  # (N, C, T)

        if X_test.ndim == 2:
            X_test = X_test[:, np.newaxis, :]  # (N, 1, T)
        elif X_test.ndim == 3:
            X_test = X_test.transpose(0, 2, 1)  # (N, C, T)

        # Apply debug mode if enabled
        if self.debug:
            n_debug = min(10, len(X_train))
            indices = np.random.choice(len(X_train), n_debug, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            n_debug_test = min(5, len(X_test))
            indices_test = np.random.choice(len(X_test), n_debug_test, replace=False)
            X_test = X_test[indices_test]
            y_test = y_test[indices_test]

        # Encode labels to 0-indexed if needed
        unique_labels = np.unique(np.concatenate([y_train, y_test]))
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        y_train = np.array([label_mapping[label] for label in y_train])
        y_test = np.array([label_mapping[label] for label in y_test])

        num_classes = len(unique_labels)
        
        # Create a simple container for train/test arrays and metadata.
        class UCRDataset:
            def __init__(self, X_train, y_train, X_test, y_test):
                self.X_train = X_train
                self.y_train = y_train
                self.X_test = X_test
                self.y_test = y_test
                self.num_classes = len(np.unique(y_test))
                
                # Compute basic properties
                self.sequence_length = X_train.shape[-1]
                self.num_variates = X_train.shape[1] if X_train.ndim > 2 else 1
                self.num_train = len(X_train)
                self.num_test = len(X_test)

        dataset = UCRDataset(X_train, y_train, X_test, y_test)

        return dict(
            ucr_dataset=dataset,
            num_classes=num_classes,
            dataset_name=self.dataset_name,
        )
