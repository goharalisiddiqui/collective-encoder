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

atomic_numbers = {'C': 6, 'H': 1, 'O': 8, 'N': 7, 'S': 16, 'P': 15, 'F': 9, 'Cl': 17, 'Br': 35, 'I': 53, 'Si': 14
                  }


class XtcData(Dataset):
    """XTC dataset"""

    def __init__(
        self,
        structures: List[ase.Atoms],
        dtype=torch.float32,
    ):
        self.positions = [torch.tensor(
            s.positions, dtype=dtype) for s in structures]
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

    parser.add_argument('--xtcfile', required=True, type=str,
                        help='Input compressed coordinate file')
    parser.add_argument('--tprfile', required=True, type=str,
                        help='Input binary file containing the topology')
    parser.add_argument('--resnames', required=True, nargs='+',
                        help='Residue names to get the coordinates')
    parser.add_argument('--datasize', dest="dataset_size",
                        type=int, default=None, help='Size of the dataset to use')
    parser.add_argument('--seq', dest="sequential",
                        action='store_true', help='Weather the trajectory should be sequential or not')
    # parser.add_argument('--labels',dest = 'label_list', nargs='+', help='Label columns in the data file')

    args, _ = parser.parse_known_args()

    return args


XTC_args = xtcdatset_args


class XtcDataset(pl.LightningDataModule):
    def __init__(self,
                 xtcfile: str,
                 tprfile: str,
                 resnames: List[str] = None,
                 dataset_size: int = None,
                 train_prop: float = 0.6,
                 validation_prop: float = 0.2,
                 batch_size: int = None,
                 val_batch_size: int = None,
                 test_batch_size: int = None,
                 num_workers: int = 1,
                 sequential: bool = False,
                 verbose: bool = True,
                 standardize_inputs: bool = False):
        super().__init__()
        print(f"\n\n[Initializing XtcDataset Module]") if verbose else None
        print("==========================================") if verbose else None
        print(
            f"Loading coordinates from file {xtcfile} and topology from file {tprfile}") if verbose else None
        if not os.path.exists(xtcfile):
            raise FileNotFoundError(f"File {xtcfile} not found")
        if not os.path.exists(tprfile):
            raise FileNotFoundError(f"File {tprfile} not found")
        u = mda.Universe(tprfile, xtcfile)
        for res in resnames:
            assert res in list(set(
                u.residues.resnames)), f"Residue \"{res}\" not found in the topology file. Available residues: {list(set(u.residues.resnames))}"
        mol = u.select_atoms(f"resname " + " ".join(resnames) + " and not name H*")
        transforms = [trans.unwrap(mol),
                      trans.center_in_box(mol, center='geometry', point=[0.0,0.0,0.0], wrap=False)]
        u.trajectory.add_transformations(*transforms)
        

        ch = [at.element for at in mol]
        for at in ch:
            if at not in atomic_numbers:
                raise ValueError(
                    f"Atom {at} not found in atomic numbers dictionary")
        ch = [atomic_numbers[at] for at in ch]
        self.loatn = ch
        self.bonds = mol.get_connections('bonds', outside=False).indices
        for i in range(len(self.bonds)):
            self.bonds[i] = [np.where(mol.atoms.indices == self.bonds[i][0])[0][0], np.where(mol.atoms.indices == self.bonds[i][1])[0][0]]
            
        mol_traj = []
        # for ind,ts in enumerate(tqdm(u.trajectory[:1000])):
        s, e = 0, len(u.trajectory)
        if dataset_size is not None:
            if dataset_size > len(u.trajectory):
                raise ValueError(
                    f"Dataset size {dataset_size} must be less than the number of frames {len(u.trajectory)}.")
            s = random.randint(0, len(u.trajectory) - dataset_size - 1)
            e = dataset_size + s
        else:
            dataset_size = len(u.trajectory)
        print(f"Reading trajectory of {e-s} frames...") if verbose else None
        if sequential:
            for ts in tqdm(u.trajectory[s:e], disable=not verbose):
                structure = ase.Atoms(numbers=ch, positions=mol.positions)
                
                residues = [str(r.residue.resname) for r in mol.atoms]
                resids = [r.residue.resid for r in mol.atoms]
                atomnames = [str(a.name) for a in mol.atoms]
                structure.set_array('residuenumbers', np.array(resids))
                structure.set_array('residuenames', np.array(residues))
                structure.set_array('atomtypes', np.array(atomnames))
                
                mol_traj.append(structure)
        else:
            for ts in tqdm(range(dataset_size), disable=not verbose):
                u.trajectory[random.randint(0, len(u.trajectory) - 1)]
                structure = ase.Atoms(numbers=ch, 
                                      positions=mol.positions)
                
                residues = [str(r.residue.resname) for r in mol.atoms]
                resids = [r.residue.resid for r in mol.atoms]
                atomnames = [str(a.name) for a in mol.atoms]
                structure.set_array('residuenumbers', np.array(resids))
                structure.set_array('residuenames', np.array(residues))
                structure.set_array('atomtypes', np.array(atomnames))
                
                mol_traj.append(structure)
        print(f"Finished reading trajectory.") if verbose else None
        FRAMES = len(mol_traj)
        assert train_prop + validation_prop < 1.0
        test_prop = 1.0 - train_prop - validation_prop
        self.train_size = int(FRAMES * train_prop)
        self.validation_size = int(FRAMES * validation_prop)
        self.test_size = FRAMES - self.train_size - self.validation_size
        
        self.mol_traj = mol_traj

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

        print(f"Total frames: {FRAMES}, Train size: {self.train_size}, Batch size: {self.hparams.batch_size}, Validation size: {self.validation_size}") if verbose else None
        print("==========================================") if verbose else None

    # def prepare_data(self): # only called on 1 GPU/TPU in distributed

    def setup(self, stage):  # Called on every GPU/TPU in distributed
        # Assign train/val datasets for use in dataloaders
        self.mddata_train, self.mddata_val, self.mddata_test, _ = \
            torch.utils.data.random_split(
                self.xtcData_full,
                [
                    self.train_size,
                    self.validation_size,
                    self.test_size,
                    len(self.xtcData_full) - self.train_size -
                    self.validation_size - self.test_size
                ])
    
    def get_atns(self):
        return self.loatn
    
    def get_bond_indices(self):
        return self.bonds
    
    def output_trajectory(self, output_file, trajectory = None):
        if os.path.exists(output_file):
            Warning(f"File {output_file} already exists. Overwriting...")
            os.remove(output_file)
        mol_traj = self.mol_traj
        for i in range(len(self.mol_traj)):
            frame = mol_traj[i]
            if trajectory is not None:
                if i >= len(trajectory):
                    break
                frame.positions = trajectory[i]
            frame.write(output_file, append = True)
            
        

    def get_full_batch(self):
        mddata = self.xtcData_full
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
        mddata_test = self.xtcData_full
        return DataLoader(
            mddata_test,
            batch_size=self.hparams.test_batch_size if self.hparams.test_batch_size is not None else len(
                mddata_test),
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
