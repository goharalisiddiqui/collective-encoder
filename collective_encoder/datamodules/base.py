from abc import ABC
from typing import Dict, List, Any

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import ConcatDataset
import pytorch_lightning as pl

from collective_encoder.common.module import CEModule
import gslibs.validation as gsv


class BaseDataModule(CEModule, pl.LightningDataModule, ABC):
    """
    Base class for all PyTorch Lightning DataModules in the collective encoder framework.
    Provides common functionality for data normalization, dataloader creation, and batch handling.

    This is a generic base class that should work with any type of data (coordinates, COLVAR, KMC, etc.).

    Args:
        train_size (int): Number of training samples.
        batch_size (int): Batch size for training.
        validation_size (int, optional): Number of validation samples. Default is 0.
        val_batch_size (int, optional): Batch size for validation. Default is 0.
        test_size (int, optional): Number of test samples. Default is 0.
        test_batch_size (int, optional): Batch size for testing. Default is 0.
        norm_type (str, optional): Type of normalization to apply ('standard' or 'minmax'). Default is 'standard'.
        num_workers (int, optional): Number of workers for data loading. Default is 1.
        kwargs: Additional keyword arguments to be passed to base classes.
    """

    # To be overridden by subclasses
    _IDENTIFIER: str = None
    _COMPATIBLE_DATAREADERS: List[str] = []
    _COMPATIBLE_DATASETS: List[str] = []
    _COMPATIBLE_LABELERS: List[str] = []
    _REQUIRED_ARGS = ['train_size', 'batch_size']
    _OPTIONAL_ARGS = {
        'validation_size': 0,
        'val_batch_size': 0,
        'test_size': 0,
        'test_batch_size': 0,
        'norm_type': 'standard',
        'num_workers': 1,
    }

    def __init__(self,
                 args: Dict[str, Any] = None,
                 **kwargs,
                 ):
        pl.LightningDataModule.__init__(self)
        CEModule.__init__(self, args, **kwargs)

        # Input validation
        gsv.check_int(
            non_negative=True,
            train_size=self.train_size,
            batch_size=self.batch_size,
            validation_size=self.validation_size,
            val_batch_size=self.val_batch_size,
            test_size=self.test_size,
            test_batch_size=self.test_batch_size,
            num_workers=self.num_workers,
        )
        self.save_hyperparameters()


        # Initialize common data attributes
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.target_scaler = None
        self.num_frames = 0
        self.datapoint_shape = None
        self.label_list = []

        # Dataloader class to be set by subclasses
        self.dl_cls = None
        self.check_module_compatibility()

    def _check_batch_sizes(self):
        """Validate batch sizes against dataset sizes."""
        if self.val_batch_size == 0:
            self.val_batch_size = self.validation_size
        if self.test_batch_size == 0:
            self.test_batch_size = self.test_size

        assert self.batch_size <= self.train_size, "Batch size must be less than or equal to the training size"
        if self.validation_size > 0:
            assert self.val_batch_size <= self.validation_size, "Validation batch size must be less than or equal to the validation size"
        if self.test_size > 0:
            assert self.test_batch_size <= self.test_size, "Test batch size must be less than or equal to the test size"

    def set_batch_size(self, batch_size: int):
        """Set the batch size for training."""
        self.batch_size = batch_size
        self._check_batch_sizes()
    
    def set_val_batch_size(self, val_batch_size: int):
        """Set the batch size for validation."""
        self.val_batch_size = val_batch_size
        self._check_batch_sizes()
    
    def fit_target_scaler(self):
        """Fit the target scaler on training, validation, and test data."""
        if self.target_scaler is not None:
            return

        if self.norm_type == 'standard':
            self.target_scaler = StandardScaler()
        elif self.norm_type == 'minmax':
            self.target_scaler = MinMaxScaler()
        else:
            raise ValueError(f"Normalization type {self.norm_type} not supported")

        # Collect normalization data from all available datasets
        parts = [self.train_data.get_norm_data()]
        if self.val_data:
            parts.append(self.val_data.get_norm_data())
        if self.test_data:
            parts.append(self.test_data.get_norm_data())
        data_to_normalize = np.vstack(parts)
        data_to_normalize = data_to_normalize.reshape(data_to_normalize.shape[0], -1)  # Flatten if necessary
        self.target_scaler.fit(data_to_normalize)

    # Common dataloader methods
    def get_dataloader(self, data, batch_size, shuffle=False):
        """Create a dataloader for the given dataset."""
        if self.dl_cls is None:
            return torch.utils.data.DataLoader(
                data,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
        else:
            return self.dl_cls(
                data,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=True,
                num_workers=self.num_workers,
                pin_memory=True
            )

    def train_dataloader(self):
        """Return training dataloader."""
        return self.get_dataloader(self.train_data, self.batch_size, shuffle=True)

    def val_dataloader(self):
        """Return validation dataloader."""
        if self.val_data is None or len(self.val_data) == 0:
            raise ValueError("Validation data is not available")
        return self.get_dataloader(self.val_data, self.val_batch_size, shuffle=False)

    def test_dataloader(self):
        """Return test dataloader."""
        if self.test_data is None or len(self.test_data) == 0:
            raise ValueError("Test data is not available")
        return self.get_dataloader(self.test_data, self.test_batch_size, shuffle=False)

    def predict_dataloader(self):
        """Return prediction dataloader."""
        return self.get_dataloader(self.train_data, len(self.train_data), shuffle=False)

    def full_dataloader(self):
        """Return dataloader with all data combined."""
        all_data = self.train_data
        if self.val_data is not None and len(self.val_data) > 0:
            all_data = torch.utils.data.ConcatDataset([all_data, self.val_data])
        if self.test_data is not None and len(self.test_data) > 0:
            all_data = torch.utils.data.ConcatDataset([all_data, self.test_data])
        return self.get_dataloader(all_data, len(all_data), shuffle=False)

    def get_full_batch(self):
        """Get a full batch containing all data."""
        dl = self.full_dataloader()
        return next(iter(dl))

    # Scaler methods
    def target_scaler_transform(self, X):
        """Transform data using the fitted target scaler."""
        self.fit_target_scaler()
        return self.target_scaler.transform(X)

    def target_inverse_scaler(self, X):
        """Inverse transform data using the fitted target scaler."""
        self.fit_target_scaler()
        return self.target_scaler.inverse_transform(X)

    def get_scaler_mean(self):
        """Get the mean from the target scaler."""
        self.fit_target_scaler()
        if self.norm_type == 'minmax':
            return self.target_scaler.data_min_
        return self.target_scaler.mean_

    def get_scaler_var(self):
        """Get the variance from the target scaler."""
        self.fit_target_scaler()
        return self.target_scaler.var_

    def get_scaler_scale(self):
        """Get the scale from the target scaler."""
        self.fit_target_scaler()
        return self.target_scaler.scale_

    # Dataset information methods
    def get_datapoint_shape(self):
        """Get the shape of a single datapoint."""
        return self.datapoint_shape

    def get_dataset(self):
        """Get the combined train + val + test dataset for full-data operations."""
        parts = [self.train_data]
        if self.val_data is not None:
            parts.append(self.val_data)
        if self.test_data is not None:
            parts.append(self.test_data)
        return ConcatDataset(parts) if len(parts) > 1 else parts[0]
    
    def get_train_dataset(self):
        """Get the training dataset."""
        return self.train_data
    
    def get_val_dataset(self):
        """Get the validation dataset."""
        if self.val_data is None:
            raise ValueError("Validation data is not available")
        return self.val_data

    def get_test_dataset(self):
        """Get the test dataset."""
        if self.test_data is None:
            raise ValueError("Test data is not available")
        return self.test_data

    def set_num_workers(self, num_workers: int):
        """Set the number of workers for data loading."""
        self.num_workers = num_workers
    
    def get_label_names(self) -> List[str]:
        """Get the names of the label columns produced by the labeler."""
        return self.label_list

    @classmethod
    def get_compatible_datareaders(cls) -> List[str]:
        """Get list of compatible datareader types."""
        return cls._COMPATIBLE_DATAREADERS

    @classmethod
    def get_compatible_datasets(cls) -> List[str]:
        """Get list of compatible dataset types."""
        return cls._COMPATIBLE_DATASETS

    @classmethod
    def get_compatible_labelers(cls) -> List[str]:
        """Get list of compatible labeler types."""
        return cls._COMPATIBLE_LABELERS
    
    def check_module_compatibility(self):
        """Check if the provided datareader, dataset, and labeler types are compatible with this DataModule."""
        if hasattr(self, 'datareader_type') and self.datareader_type is not None:
            self.check_datareader_compatibility(self.datareader_type)
        if hasattr(self, 'dataset_type') and self.dataset_type is not None:
            self.check_dataset_compatibility(self.dataset_type)
        if hasattr(self, 'labeler_type') and self.labeler_type is not None:
            self.check_labeler_compatibility(self.labeler_type)

    def check_datareader_compatibility(self, datareader_id : str):
        """Check if the provided datareader type is compatible with this DataModule."""
        compatible_datareaders = self.get_compatible_datareaders()
        if datareader_id not in compatible_datareaders:
            raise ValueError(f"Datareader type {datareader_id} is not compatible "
                             f"with {type(self).__name__}, which supports "
                             f"{compatible_datareaders}")
    
    def check_dataset_compatibility(self, dataset_id : str):
        """Check if the provided dataset type is compatible with this DataModule."""
        compatible_datasets = self.get_compatible_datasets()
        if dataset_id not in compatible_datasets:
            raise ValueError(f"Dataset type {dataset_id} is not compatible "
                             f"with {type(self).__name__}, which supports "
                             f"{compatible_datasets}")
    
    def check_labeler_compatibility(self, labeler_id : str):
        """Check if the provided labeler type is compatible with this DataModule."""
        compatible_labelers = self.get_compatible_labelers()
        if labeler_id not in compatible_labelers:
            raise ValueError(f"Labeler type {labeler_id} is not compatible "
                             f"with {type(self).__name__}, which supports "
                             f"{compatible_labelers}")