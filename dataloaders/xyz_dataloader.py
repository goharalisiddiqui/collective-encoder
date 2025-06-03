import os
import argparse
from typing import List, Dict
import random

import ase
from ase.io.extxyz import read_extxyz

import MDAnalysis as mda

from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

DOUBLE_PRECISION = False

# atomic_numbers = { 'C' : 6
#                 , 'H' : 1
#                 , 'O' : 8
#                 , 'N' : 7
#                 , 'S' : 16
#                 , 'P' : 15
#                 , 'F' : 9
#                 , 'Cl' : 17
#                 , 'Br' : 35
#                 , 'I' : 53
#                 , 'Si' : 14
# }

class XyzDataset(Dataset):
    """Default XYZ dataset"""

    def __init__(
        self,
        structures: List[ase.Atoms],
        labels: List[int],
        dtype=torch.float32,
    ):
        self.positions = [torch.tensor(s.positions, dtype=dtype) for s in structures]
        self.num_inputs = len(self.positions[0])
        self.labels = labels

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        x = ()
        x += (self.positions[index],self.labels[index])
        return x

def xyzdatset_args():
    desc = "XYZ DataLoader Arguments"
    parser = argparse.ArgumentParser(description=desc)


    parser.add_argument('--xyzfiles', required=True, nargs='+', help='Input coordinate file')
    parser.add_argument('--datasize', dest="dataset_size", type=int, default = None, help='Size (per file) of the dataset to use')
    # parser.add_argument('--labels',dest = 'label_list', nargs='+', help='Label columns in the data file')

    args, _ = parser.parse_known_args()

    return args


XYZ_args = xyzdatset_args

class XyzLoader(pl.LightningDataModule):
    def __init__(self,
                 xyzfiles : list,
                 dataset_size : int = None,
                 train_prop : float = 0.6,
                 validation_prop : float = 0.2,
                 batch_size : int = None,
                 val_batch_size : int = None,
                 test_batch_size : int = None,
                 num_workers : int = 1,
                 sequential : bool = False,
                 verbose : bool = False,
                 dataset_type : Dataset = XyzDataset,
                 dataset_args : Dict = None
                 ):
        super().__init__()
        print(f"\n\n[Initializing XyzDataloader Module]") if verbose else None
        print("==========================================") if verbose else None
        print(f"Loading coordinates from file:") if verbose else None
        for xyzfile in xyzfiles:
            print(f"  {xyzfile}") if verbose else None
        mol_traj = []
        labels = []
        for ind, xyzfile in enumerate(xyzfiles):
            if not os.path.exists(xyzfile):
                raise FileNotFoundError(f"File {xyzfile} not found")
            
            print(f"Reading trajectory from {xyzfile}...") if verbose else None
            full_traj = []
            with open(xyzfile, 'r') as f:
                for frame in read_extxyz(f, index=slice(0,-1)):
                    full_traj.append(frame)
            s, e = 0, len(full_traj)
            if dataset_size is not None and sequential:
                s = random.randint(0, len(full_traj) - dataset_size - 1)
                e = dataset_size + s
            mol_traj_current = full_traj[s:e]
            print(f"Finished reading trajectory.") if verbose else None
            if dataset_size is not None and not sequential:
                print(f"Subsampling dataset to {dataset_size} frames.") if verbose else None
                assert dataset_size <= len(mol_traj_current), f"Dataset size {dataset_size} must be less than the number of frames {len(mol_traj_current)} in {xyzfile}."
                mol_traj_current = random.choices(mol_traj_current, k=dataset_size)
            mol_traj.extend(mol_traj_current)
            labels.extend([ind] * len(mol_traj_current))
        
        self.label_list = ["Class Label"]

        FRAMES = len(mol_traj)
        assert train_prop + validation_prop < 1.0
        test_prop = 1.0 - train_prop - validation_prop
        self.train_size = int(FRAMES * train_prop)
        self.validation_size = int(FRAMES * validation_prop)
        self.test_size = FRAMES - self.train_size - self.validation_size


        self.xyzData_full = dataset_type(
                structures=mol_traj,
                labels=labels,
                dtype=torch.float64 if DOUBLE_PRECISION else torch.float32,
                **dataset_args
                )
        self.num_inputs = self.xyzData_full.num_inputs
        self.datapoint_shape = tuple(self.xyzData_full[0][0].shape)

        self.save_hyperparameters()

        if self.hparams.batch_size is None:
            self.hparams.batch_size = int(self.train_size * 0.1)
        if self.hparams.val_batch_size is None:
            self.hparams.val_batch_size = int(self.validation_size * 0.1)
        # Printing
        self.target_scaler = StandardScaler()
        # self.target_scaler.fit(self.xyzData_full)
        assert self.hparams.batch_size < self.train_size, "Batch size must be less than the training size"
        assert self.hparams.val_batch_size < self.validation_size, "Validation batch size must be less than the validation size"

        print(f"Total frames: {FRAMES}, Train size: {self.train_size}, Batch size: {self.hparams.batch_size}, Validation size: {self.validation_size}") if verbose else None
        print("==========================================") if verbose else None


    # def prepare_data(self): # only called on 1 GPU/TPU in distributed


    def setup(self, stage): # Called on every GPU/TPU in distributed
        # Assign train/val datasets for use in dataloaders
        self.mddata_train, self.mddata_val, self.mddata_test, _ = \
            torch.utils.data.random_split(
                self.xyzData_full,
                [
                    self.train_size,
                    self.validation_size,
                    self.test_size,
                    len(self.xyzData_full) - self.train_size - self.validation_size - self.test_size
                ])

    def get_full_batch(self):
        mddata = self.xyzData_full
        dl = DataLoader(
            mddata,
            batch_size=len(mddata),
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True)
        return next(iter(dl))

        # called on every process in DDP
    def train_dataloader(self):
        return DataLoader(
            self.mddata_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True)

    def val_dataloader(self):
        return DataLoader(
            self.mddata_val,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True)

    def test_dataloader(self):
        mddata_test = self.xyzData_full
        return DataLoader(
            mddata_test,
            batch_size=self.hparams.test_batch_size if self.hparams.test_batch_size is not None else len(mddata_test),
            shuffle=False,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True)

    def target_scaler(self, X):
        raise NotImplementedError("Normalization not implemented for XYZ data")
        return self.target_scaler.transform(X)

    def target_inverse_scaler(self, X):
        raise NotImplementedError("Normalization not implemented for XYZ data")
        return self.target_scaler.inverse_transform(X)

    def get_scaler_mean(self):
        raise NotImplementedError("Normalization not implemented for XYZ data")
        return self.target_scaler.mean_

    def get_scaler_var(self):
        raise NotImplementedError("Normalization not implemented for XYZ data")
        return self.target_scaler.var_

    def get_scaler_scale(self):
        raise NotImplementedError("Normalization not implemented for XYZ data")
        return self.target_scaler.scale_

    def get_datapoint_shape(self):
        return self.datapoint_shape

