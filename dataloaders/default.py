import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #FIXME: This is weird hack to import from parent dir. Need to write a proper package

from typing import List, Dict, Any

import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import pytorch_lightning as pl

from collective_encoder.common.module import CEModule

from collective_encoder.datareaders.solver import get_datareader
from collective_encoder.datasets.solver import get_dataset_cls_dl

NON_READABILITY_TOLERANCE = 0.02 # Do not raise error if less than this fraction of frames are not readable

class DefaultDatamodule(pl.LightningDataModule, CEModule):
    '''
    Default PyTorch Lightning DataModule for molecular dynamics data.
    Handles data reading, dataset creation, normalization, and dataloader setup.

    Args:
        train_size (int): Number of training samples.
        batch_size (int): Batch size for training.
        validation_size (int, optional): Number of validation samples. Default is 0.
        val_batch_size (int, optional): Batch size for validation. Default is 0.
        test_size (int, optional): Number of test samples. Default is 0.
        test_batch_size (int, optional): Batch size for testing. Default is 0.
        test_full_dataset (bool, optional): Whether to use the full dataset for testing. Default is False.
        datareader_type (str, optional): Type of datareader to use. Default is 'XTC'.
        datareader_args (Dict[str, Any], optional): Arguments for the datareader. Default is {}.
        dataset_type (str, optional): Type of dataset to use. Default is 'DEFAULT'.
        dataset_args (Dict[str, Any], optional): Arguments for the dataset. Default is {}.
        labeler_type (str, optional): Type of labeler to use. Default is None.
        labeler_args (Dict[str, Any], optional): Arguments for the labeler. Default is {}.
        norm_type (str, optional): Type of normalization to apply ('standard' or 'minmax'). Default is 'standard'.
        num_workers (int, optional): Number of workers for data loading. Default is 1.
        
    '''
    def __init__(self,
                 train_size : int,
                 batch_size : int,
                 validation_size : int = 0,
                 val_batch_size : int = 0,
                 test_size : int = 0,
                 test_batch_size : int = 0,
                 test_full_dataset : bool = False,
                 datareader_type : str = 'XTC',
                 datareader_args : Dict[str, Any] = {},
                 sequential : bool = False,
                 max_frames : int = None,
                 dataset_type : str = 'DEFAULT',
                 dataset_args : Dict[str, Any] = {},
                 labeler_type : str = None,
                 labeler_args : Dict[str, Any] = {},
                 norm_type : str = 'standard',
                 num_workers : int = 1,
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        
        # Input checks
        self.check_int(
            non_negative=True,
            train_size=train_size,
            batch_size=batch_size,
            validation_size=validation_size,
            val_batch_size=val_batch_size,
            test_size=test_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
        )
        self.check_mutually_exclusive(test_size=test_size, test_full_dataset=test_full_dataset)
        
        self.save_hyperparameters()

        # Save and log sizes
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size if not test_full_dataset else max_frames
        self.log_msg(f"Train size: {self.train_size}, Validation size: {self.validation_size}, Test size: {self.test_size}") 

        # Initialize the trajectory reader
        datareader_cls = get_datareader(datareader_type)
        datareader = datareader_cls(**datareader_args)
        if max_frames is None:
            max_frames = datareader.get_total_frames()
        self.sequential = sequential
        self.max_frames = max_frames
        self.atomic_numbers = datareader.get_atomic_numbers()
        self.element_symbols = datareader.get_element_symbols()
        self.bonds = datareader.get_bonds()
        
        if self.max_frames < (self.train_size + self.validation_size + (0 if test_full_dataset else self.test_size)):
            raise ValueError(f"Not enough frames ({self.max_frames}) in trajectory for the requested dataset sizes (train: {self.train_size}, val: {self.validation_size}, test: {self.test_size})")
        
        self._calculate_indices()

        trajs, labels = datareader.read_trajectory(
            indices = [self.train_indices, self.val_indices, self.test_indices],
            labeler_type=labeler_type,
            labeler_args=labeler_args,
        )

        self._validate_traj_length(trajs[0], len(self.train_indices))
        self._validate_traj_length(trajs[1], len(self.val_indices))
        self._validate_traj_length(trajs[2], len(self.test_indices))
        
        # Create dataset
        dataset_class, dataset_args, self.dl_cls = \
                get_dataset_cls_dl(dataset_type, dataset_args, datareader)
        self.train_data = dataset_class(
                structures=trajs[0],
                labels=labels[0],
                **dataset_args,
                **kwargs
                )
        self.val_data = dataset_class(
                structures=trajs[1],
                labels=labels[1],
                **dataset_args,
                **kwargs
                ) if validation_size > 0 else []
        self.test_data = dataset_class(
                structures=trajs[2],
                labels=labels[2],
                **dataset_args,
                **kwargs
                ) if test_size > 0 else []
        
        self.num_frames = len(self.train_data) + len(self.val_data) + len(self.test_data)
        self.datapoint_shape = self.train_data.get_datapoint_shape()
        self.log_msg(f"Loaded dataset with {self.num_frames} frames -> "
                     f"Train: {len(self.train_data)}, "
                     f"Validation: {len(self.val_data)}, "
                     f"Test: {len(self.test_data)}") 
        self.log_msg(f"Datapoint shape: {self.datapoint_shape}") 

        # Check batch sizes
        if self.hparams.val_batch_size == 0:
            self.hparams.val_batch_size = self.validation_size
        if self.hparams.test_batch_size == 0:
            self.hparams.test_batch_size = self.test_size
        assert self.hparams.batch_size <= self.train_size, "Batch size must be less than the training size"
        assert self.hparams.val_batch_size <= self.validation_size, "Validation batch size must be less than the validation size"
        assert self.hparams.test_batch_size <= self.test_size, "Test batch size must be less than the test size"

        self.target_scaler = None
    
    def _validate_traj_length(self, traj: List[Any], expected_length: int):
        if len(traj) < expected_length * (1 - NON_READABILITY_TOLERANCE):
            raise ValueError(f"Trajectory length {len(traj)} is less than expected {expected_length} frames")

    # def prepare_data(self): # only called on 1 GPU/TPU in distributed
    # def setup(self, stage):  # Called on every GPU/TPU in distributed

    def _calculate_indices(self):
        if self.sequential:
            required_size = self.train_size + self.validation_size
            if not self.hparams.test_full_dataset:
                required_size += self.test_size
            train_start = random.randint(0, self.max_frames - required_size)
            self.train_indices = np.arange(train_start, train_start + self.train_size)
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
            self.log_msg(f"Sequential split selected.")
            self.log_msg(f"Train indices: {self.train_indices[0]} - {self.train_indices[-1]}, ")
            if len(self.val_indices) > 0:
                self.log_msg(f"Validation indices: {self.val_indices[0]} - {self.val_indices[-1]}, ")
            if len(self.test_indices) > 0:
                self.log_msg(f"Test indices: {self.test_indices[0]} - {self.test_indices[-1]}")
        else:
            all_indices = range(self.max_frames)
            self.train_indices = random.sample(all_indices, self.train_size)
            remainder_indices = [i for i in all_indices if i not in self.train_indices]
            self.val_indices = random.sample(remainder_indices, self.validation_size)
            remainder_indices = [i for i in remainder_indices if i not in self.val_indices]
            if self.hparams.test_full_dataset:
                self.test_indices = all_indices
            else:
                self.test_indices = random.sample(list(remainder_indices), self.test_size)
            self.log_msg(f"Random split selected. Train indices size: {len(self.train_indices)}, "
                         f"Validation indices size: {len(self.val_indices)}, "
                         f"Test indices size: {len(self.test_indices)}")
        
    def get_atns(self):
        return self.atomic_numbers

    def get_bond_indices(self):
        return self.bonds

    def get_element_symbols(self):
        return self.element_symbols

    def get_dataset(self):
        return self.train_data

    def fit_target_scaler(self):
        if self.target_scaler is not None:
            return
        if self.hparams.norm_type == 'standard':
            self.target_scaler = StandardScaler()
        elif self.hparams.norm_type == 'minmax':
            self.target_scaler = MinMaxScaler()
        else:
            raise ValueError(f"Normalization type {self.hparams.norm_type} not supported")
        
        data_to_normalize = np.vstack([self.train_data.get_norm_data(),
                                      self.val_data.get_norm_data(),
                                      self.test_data.get_norm_data()])
        self.target_scaler.fit(data_to_normalize)

    # def output_trajectory(self, output_file, trajectory = None):
    #     if os.path.exists(output_file):
    #         Warning(f"File {output_file} already exists. Overwriting...")
    #         os.remove(output_file)
    #     mol_traj = self.datareader.mol_traj
    #     for i in range(len(self.datareader.mol_traj)):
    #         frame = mol_traj[i]
    #         if trajectory is not None:
    #             if i >= len(trajectory):
    #                 break
    #             frame.positions = trajectory[i]
    #         frame.write(output_file, append = True)

    def get_full_batch(self):
        dl = self.full_dataloader()
        return next(iter(dl))

    # called on every process in DDP
    def get_dataloader(self, data, batch_size):
        return self.dl_cls(
            data,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True)

    def train_dataloader(self):
        return self.get_dataloader(self.train_data, 
                                              self.hparams.batch_size)
    
    def predict_dataloader(self):
        return self.get_dataloader(self.train_data, 
                                              len(self.train_data))

    def val_dataloader(self):
        if self.val_data is None or len(self.val_data) == 0:
            raise ValueError("Validation data is not available")
        return self.get_dataloader(self.val_data, 
                                              self.hparams.val_batch_size)

    def test_dataloader(self):
        if self.test_data is None or len(self.test_data) == 0:
            raise ValueError("Test data is not available")
        return self.get_dataloader(self.test_data, 
                                              self.hparams.val_batch_size)

    def full_dataloader(self):
        all_data = self.train_data
        if self.val_data is not None and len(self.val_data) > 0:
            all_data = torch.utils.data.ConcatDataset([all_data, self.val_data])
        if self.test_data is not None and len(self.test_data) > 0:
            all_data = torch.utils.data.ConcatDataset([all_data, self.test_data])
        return self.get_dataloader(all_data, len(all_data))

    def target_scaler(self, X):
        self.fit_target_scaler()
        return self.target_scaler.transform(X)

    def target_inverse_scaler(self, X):
        self.fit_target_scaler()
        return self.target_scaler.inverse_transform(X)

    def get_scaler_mean(self):
        self.fit_target_scaler()
        if self.hparams.norm_type == 'minmax':
            return self.target_scaler.data_min_
        return self.target_scaler.mean_

    def get_scaler_var(self):
        self.fit_target_scaler()
        return self.target_scaler.var_

    def get_scaler_scale(self):
        self.fit_target_scaler()
        return self.target_scaler.scale_

    def get_datapoint_shape(self):
        return self.datapoint_shape

    def get_fake_systems(self):
        at_types = self.atns
        mol_traj = self.datareader.mol_traj
        from metatomic.torch import System
        fake_systems = [
            System(
                types=torch.tensor(at_types, dtype=torch.long),
                positions=torch.tensor(mol_traj[0].positions, dtype=torch.float64),
                cell=torch.tensor(mol_traj[0].get_cell(), dtype=torch.float64),
                pbc=torch.tensor([True, True, True]),),
            System(
                types=torch.tensor(at_types, dtype=torch.long),
                positions=torch.tensor(mol_traj[1].positions, dtype=torch.float64),
                cell=torch.tensor(mol_traj[1].get_cell(), dtype=torch.float64),
                pbc=torch.tensor([True, True, True]),)
            ]

        return fake_systems
