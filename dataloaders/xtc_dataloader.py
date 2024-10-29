import os
import argparse
from typing import List, Dict
from tqdm import tqdm
import random

import numpy as np
import ase

import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis.rms import rmsd
import MDAnalysis.transformations as trans

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

DOUBLE_PRECISION = False

atomic_numbers = { 'C' : 6
                , 'H' : 1
                , 'O' : 8
                , 'N' : 7
                , 'S' : 16
                , 'P' : 15
                , 'F' : 9
                , 'Cl' : 17
                , 'Br' : 35
                , 'I' : 53
                , 'Si' : 14
}

class XtcData(Dataset):
    """XTC dataset"""

    def __init__(
        self,
        structures: List[ase.Atoms],
        dtype=torch.float32,
    ):
        self.positions = [torch.tensor(s.positions) for s in structures]
        self.num_inputs = len(self.positions[0])

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        x = ()
        x += (self.positions[index],)
        return x

def xtcdatset_args():
    desc = "Xtc Dataset Arguments"
    parser = argparse.ArgumentParser(description=desc)


    parser.add_argument('--xtcfile', required=True, type=str, help='Input compressed coordinate file')
    parser.add_argument('--tprfile', required=True, type=str, help='Input binary file containing the topology')
    parser.add_argument('--resname', required=True, type=str, help='Residue name to get the coordinates')
    parser.add_argument('--datasize', dest="dataset_size", type=int, default = None, help='Size of the dataset to use')
    # parser.add_argument('--labels',dest = 'label_list', nargs='+', help='Label columns in the data file')

    args, _ = parser.parse_known_args()

    return args


XTC_args = xtcdatset_args

class XtcDataset(pl.LightningDataModule):
    def __init__(self,
                 xtcfile : str,
                 tprfile : str,
                 resname : str = None,
                 dataset_size : int = None,
                 train_prop : float = 0.6,
                 validation_prop : float = 0.2,
                 batch_size : int = None,
                 val_batch_size : int = None,
                 test_batch_size : int = None,
                 num_workers : int = 1,
                 standardize_inputs : bool = False):
        super().__init__()
        print(f"\n\n[Initializing XtcDataset Module]")
        print("==========================================")
        print(f"Loading coordinates from file {xtcfile} and topology from file {tprfile}")
        if not os.path.exists(xtcfile):
            raise FileNotFoundError(f"File {xtcfile} not found")
        if not os.path.exists(tprfile):
            raise FileNotFoundError(f"File {tprfile} not found")
        u = mda.Universe(tprfile, xtcfile)
        mol = u.select_atoms(f"resname {resname}")
        transforms = [trans.unwrap(mol),
              trans.center_in_box(mol, wrap=True)]
        u.trajectory.add_transformations(*transforms)

        print(f"Reading trajectory of {len(u.trajectory)} frames...")
        mol_traj = []
        # for ind,ts in enumerate(tqdm(u.trajectory[:1000])):
        for ind,ts in enumerate(tqdm(u.trajectory)):
            mol_pos = mol.positions
            ch = [at.name[0] for at in mol]
            for at in ch:
                if at not in atomic_numbers:
                    raise ValueError(f"Atom {at} not found in atomic numbers")
            ch = [atomic_numbers[at] for at in ch]

            structure = ase.Atoms(numbers=ch, positions=mol_pos)
            mol_traj.append(structure)
        print(f"Finished reading trajectory.")
        if dataset_size is not None:
            print(f"Subsampling dataset to {dataset_size} frames.")
            assert dataset_size <= len(mol_traj), f"Dataset size {dataset_size} must be less than the number of frames {len(mol_traj)}."
            mol_traj = random.choices(mol_traj, k=dataset_size)

        FRAMES = len(mol_traj)
        assert train_prop + validation_prop < 1.0
        test_prop = 1.0 - train_prop - validation_prop
        self.train_size = int(FRAMES * train_prop)
        self.validation_size = int(FRAMES * validation_prop)
        self.test_size = FRAMES - self.train_size - self.validation_size

        self.xtcData_full = XtcData(
                structures=mol_traj,
                dtype=torch.float64 if DOUBLE_PRECISION else torch.float32)
        self.num_inputs = self.xtcData_full.num_inputs
        self.datapoint_shape = tuple(self.xtcData_full[0][0].shape)

        self.save_hyperparameters()

        if self.hparams.batch_size is None:
            self.hparams.batch_size = int(self.train_size * 0.1)
        if self.hparams.val_batch_size is None:
            self.hparams.val_batch_size = int(self.validation_size * 0.1)
        # Printing
        self.target_scaler = StandardScaler()
        # self.target_scaler.fit(self.xtcData_full)
        assert self.hparams.batch_size < self.train_size, "Batch size must be less than the training size"
        assert self.hparams.val_batch_size < self.validation_size, "Validation batch size must be less than the validation size"

        print(f"Total frames: {FRAMES}, Train size: {self.train_size}, Batch size: {self.hparams.batch_size}, Validation size: {self.validation_size}")
        print("==========================================")


    # def prepare_data(self): # only called on 1 GPU/TPU in distributed


    def setup(self, stage): # Called on every GPU/TPU in distributed
        # Assign train/val datasets for use in dataloaders
        self.mddata_train, self.mddata_val, self.mddata_test, _ = \
            torch.utils.data.random_split(
                self.xtcData_full,
                [
                    self.train_size,
                    self.validation_size,
                    self.test_size,
                    len(self.xtcData_full) - self.train_size - self.validation_size - self.test_size
                ])


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
        mddata_test = self.xtcData_full
        return DataLoader(
            mddata_test,
            batch_size=self.hparams.test_batch_size if self.hparams.test_batch_size is not None else len(mddata_test),
            shuffle=False,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True)

    def target_scaler(self, X):
        raise NotImplementedError("Normalization not implemented for XTC data")
        return self.target_scaler.transform(X)

    def target_inverse_scaler(self, X):
        raise NotImplementedError("Normalization not implemented for XTC data")
        return self.target_scaler.inverse_transform(X)

    def get_scaler_mean(self):
        raise NotImplementedError("Normalization not implemented for XTC data")
        return self.target_scaler.mean_

    def get_scaler_var(self):
        raise NotImplementedError("Normalization not implemented for XTC data")
        return self.target_scaler.var_

    def get_scaler_scale(self):
        raise NotImplementedError("Normalization not implemented for XTC data")
        return self.target_scaler.scale_

    def get_datapoint_shape(self):
        return self.datapoint_shape

