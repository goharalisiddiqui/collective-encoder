import random
from typing import List, Dict, Any

import numpy as np
import torch

from collective_encoder.datamodules.base import BaseDataModule
from collective_encoder.datareaders.resolver import get_datareader
from collective_encoder.datasets.resolver import get_dataset_cls_dl

NON_READABILITY_TOLERANCE = 0.02  # Do not raise error if less than this fraction of frames are not readable

class CoordinatesDataModule(BaseDataModule):
    """
    PyTorch Lightning DataModule for molecular dynamics coordinate data.
    Handles data reading, dataset creation, normalization, and dataloader setup for coordinate-based data.

    Args:
        datareader_type (str, optional): Type of datareader to use. Default is 'XTC'.
        dataset_type (str, optional): Type of dataset to use. Default is 'DEFAULT'.
        train_size (int): Number of training samples.
        batch_size (int): Batch size for training.
        datareader_args (Dict[str, Any], optional): Arguments for the datareader. Default is {}.
        dataset_args (Dict[str, Any], optional): Arguments for the dataset. Default is {}.
        validation_size (int, optional): Number of validation samples. Default is 0.
        val_batch_size (int, optional): Batch size for validation. Default is 0.
        test_size (int, optional): Number of test samples. Default is 0.
        test_batch_size (int, optional): Batch size for testing. Default is 0.
        test_full_dataset (bool, optional): Whether to use the full dataset for testing. Default is False.
        sequential (bool, optional): Whether to split the dataset sequentially or randomly. Default is False.
        max_frames (int, optional): Maximum number of frames to read from the trajectory. Default is None (read all frames).
        labeler_type (str, optional): Type of labeler to use. Default is None.
        labeler_args (Dict[str, Any], optional): Arguments for the labeler. Default is {}.
        norm_type (str, optional): Type of normalization to apply ('standard' or 'minmax'). Default is 'standard'.
        num_workers (int, optional): Number of workers for data loading. Default is 1.
        kwargs: Additional keyword arguments to be passed to base classes, dataset and datareader.
    """

    # Compatible datareaders and datasets
    _IDENTIFIER = "COORDINATES"
    _COMPATIBLE_DATAREADERS = ["XTC", "XTC_CHUNKS", "XTC_CHUNKS_CG"]  # Add other coordinate-based readers
    _COMPATIBLE_DATASETS = ["DISTANCES", "POSITIONS", "GRAPH"]  # Add other coordinate-based datasets
    _COMPATIBLE_LABELERS = ["COORDINATION", "DIHEDRAL", "DISTANCE"]  # Add other coordinate-based labelers

    def __init__(self,
                 datareader_type: str,
                 dataset_type: str,
                 labeler_type : str = None,
                 datareader_args: Dict[str, Any] = {},
                 dataset_args: Dict[str, Any] = {},
                 labeler_args : Dict[str, Any] = {},
                 test_full_dataset: bool = False,
                 sequential: bool = False,
                 max_frames: int = None,
                 maximize_label_variance: bool = False,
                 **kwargs,
                 ):
        self.save_hyperparameters()
        
        if maximize_label_variance:
            assert labeler_type is not None, "labeler_type must be specified when maximize_label_variance is True"

        self.max_frames = max_frames
        
        self._initialize_datareader()

        if self.max_frames is None:
            self.max_frames = self.datareader.get_total_frames()

        super().__init__(**kwargs)

        # Add test_full_dataset validation
        if self.test_size > 0 and test_full_dataset:
            raise ValueError("test_size and test_full_dataset are mutually exclusive")
        
        if self.max_frames < (self.train_size + self.validation_size + 
                         (0 if self.hparams.test_full_dataset else self.test_size)):
            raise ValueError(f"Not enough frames ({self.max_frames}) in trajectory " 
                f"for the requested dataset sizes (train: {self.train_size}, "
                f"val: {self.validation_size}, test: {self.test_size})")
    
        if test_full_dataset:
            self.test_size = self.max_frames

        self._create_datasets()
    
    def _initialize_datareader(self):
        # Initialize the trajectory reader
        datareader_cls = get_datareader(self.hparams.datareader_type)
        self.datareader = datareader_cls(**self.hparams.datareader_args)

        # Store coordinate-specific information
        self.atomic_numbers = self.datareader.get_atomic_numbers()
        self.element_symbols = self.datareader.get_element_symbols()
        self.bonds = self.datareader.get_bonds()

    def _validate_traj_length(self, traj: List[Any], expected_length: int):
        """Validate that trajectory has expected number of frames."""
        if len(traj) < expected_length * (1 - NON_READABILITY_TOLERANCE):
            raise ValueError(f"Trajectory length {len(traj)} is less than expected {expected_length} frames")

    def _calculate_indices(self):
        """Calculate train, validation, and test indices based on split configuration."""
        if self.hparams.sequential:
            self._calculate_sequential_indices()
        else:
            self._calculate_random_indices()

    def _calculate_sequential_indices(self):
        """Calculate indices for sequential data splitting."""
        self.log_msg(f"Sequential split selected.")
        
        required_size = self.train_size + self.validation_size
        if not self.hparams.test_full_dataset:
            required_size += self.test_size

        train_start = random.randint(0, self.max_frames - required_size)
        self.train_indices = np.arange(train_start, train_start + self.hparams.train_size)
        
        val_start = random.randint(0, self.max_frames - required_size)
        if val_start >= self.train_indices[0] and val_start < self.train_indices[-1]:
            val_start += self.train_size
        self.val_indices = np.arange(val_start, val_start + self.validation_size)
        
        if self.hparams.test_full_dataset:
            self.test_indices = np.arange(self.max_frames)
        else:
            test_start = random.randint(0, self.max_frames - required_size)
            if len(self.train_indices) > 0 and \
                    test_start >= self.train_indices[0] and \
                    test_start < self.train_indices[-1]:
                test_start += self.train_size
            if len(self.val_indices) > 0 and \
                    test_start >= self.val_indices[0] and \
                    test_start < self.val_indices[-1]:
                test_start += self.validation_size
            self.test_indices = np.arange(test_start, test_start + self.test_size)

        self.log_msg(f"Train indices: {self.train_indices[0]} - {self.train_indices[-1]}")
        if len(self.val_indices) > 0:
            self.log_msg(f"Validation indices: {self.val_indices[0]} - {self.val_indices[-1]}")
        if len(self.test_indices) > 0:
            self.log_msg(f"Test indices: {self.test_indices[0]} - {self.test_indices[-1]}")

    def _calculate_random_indices(self):
        """Calculate indices for random data splitting."""
        all_indices = list(range(self.max_frames))
        self.train_indices = random.sample(all_indices, self.train_size)
        remainder_indices = [i for i in all_indices if i not in self.train_indices]
        self.val_indices = random.sample(remainder_indices, self.validation_size)
        remainder_indices = [i for i in remainder_indices if i not in self.val_indices]

        if self.hparams.test_full_dataset:
            self.test_indices = all_indices
        else:
            self.test_indices = random.sample(remainder_indices, self.test_size)

        self.log_msg(f"Random split selected. Train indices size: {len(self.train_indices)}, "
                     f"Validation indices size: {len(self.val_indices)}, "
                     f"Test indices size: {len(self.test_indices)}")

    def _create_datasets(self):
        """Create datasets for train, validation, and test splits."""

        # Calculate indices for data splitting
        self._calculate_indices()

        # Log sizes
        self.log_msg(f"Train size: {self.train_size}, "
                     f"Validation size: {self.validation_size}, "
                     f"Test size: {self.test_size}")

        # Read trajectory data
        trajs, labels = self.datareader.read_trajectory(
            indices=[self.train_indices, self.val_indices, self.test_indices],
            labeler_type=self.hparams.labeler_type,
            labeler_args=self.hparams.labeler_args,
        )

        # Validate trajectory lengths
        self._validate_traj_length(trajs[0], len(self.train_indices))
        self._validate_traj_length(trajs[1], len(self.val_indices))
        self._validate_traj_length(trajs[2], len(self.test_indices))

        # Create datasets
        dataset_class, dataset_args, self.dl_cls = \
            get_dataset_cls_dl(self.hparams.dataset_type, 
                               self.hparams.dataset_args, self.datareader)
        
        self.train_data = dataset_class(
            structures=trajs[0],
            labels=labels[0],
            dataset_args=dataset_args,
            **self.hparams
        )

        self.val_data = dataset_class(
            structures=trajs[1],
            labels=labels[1],
            dataset_args=dataset_args,
            **self.hparams
        ) if self.validation_size > 0 else []

        self.test_data = dataset_class(
            structures=trajs[2],
            labels=labels[2],
            dataset_args=dataset_args,
            **self.hparams
        ) if self.test_size > 0 else []

        # Set common attributes
        self.num_frames = len(self.train_data) + len(self.val_data) + len(self.test_data)
        self.datapoint_shape = self.train_data.get_datapoint_shape()
        self.label_list = self.datareader.label_list

        self.log_msg(f"Loaded dataset with {self.num_frames} frames -> "
                     f"Train: {len(self.train_data)}, "
                     f"Validation: {len(self.val_data)}, "
                     f"Test: {len(self.test_data)}")
        self.log_msg(f"Datapoint shape: {self.datapoint_shape}")

        # Check batch sizes
        self._check_batch_sizes()

    # Coordinate-specific methods
    def get_atns(self):
        """Get atomic numbers."""
        return self.atomic_numbers

    def get_bond_indices(self):
        """Get bond indices."""
        return self.bonds

    def get_element_symbols(self):
        """Get element symbols."""
        return self.element_symbols

    def get_fake_systems(self):
        """Get fake systems for testing purposes."""
        at_types = self.atomic_numbers
        mol_traj = self.datareader.mol_traj
        from metatomic.torch import System
        fake_systems = [
            System(
                types=torch.tensor(at_types, dtype=torch.long),
                positions=torch.tensor(mol_traj[0].positions, dtype=torch.float64),
                cell=torch.tensor(mol_traj[0].get_cell(), dtype=torch.float64),
                pbc=torch.tensor([True, True, True]),
            ),
            System(
                types=torch.tensor(at_types, dtype=torch.long),
                positions=torch.tensor(mol_traj[1].positions, dtype=torch.float64),
                cell=torch.tensor(mol_traj[1].get_cell(), dtype=torch.float64),
                pbc=torch.tensor([True, True, True]),
            )
        ]
        return fake_systems


# For backward compatibility
DefaultDatamodule = CoordinatesDataModule