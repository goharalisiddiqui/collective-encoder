import os
import argparse
from typing import List, Dict
from tqdm import tqdm
import random

import numpy as np
import ase
from ase.data import atomic_numbers

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

class XtcData(Dataset):
    """XTC dataset"""

    def __init__(
        self,
        structures: List[ase.Atoms],
        labels: List[int],
        dtype=torch.float32,
    ):
        self.positions = [torch.tensor(s.positions, dtype=dtype) for s in structures]
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
    parser.add_argument('--selection', required=True, type=str, help='Selection string of mdanalysis')
    parser.add_argument('--datasize', dest="dataset_size", type=int, default = None, help='Size of the dataset to use')
    parser.add_argument('--sequential', action='store_true', help='Take the trajectory sequentially, starting from a random frame')
    
    label_group = parser.add_mutually_exclusive_group()
    label_group.add_argument('--label_distance', dest='label_distance', type=str, default=None, help='Selection string for md analysis. Must select exactly 2 atoms. Distance between these atoms will be used as a label.')
    label_group.add_argument('--label_dihedrals', dest='label_dihedrals', type=str, default=None, help='Comma separated list of DIH_RESID e.g. "phi_1" will compute the phi dihedral angle for residue 1 and use it as a label')
    # parser.add_argument('--labels', dest = 'label_list', nargs='+', help='Label columns in the data file')

    args, _ = parser.parse_known_args()

    return args


XTC_args = xtcdatset_args

class XtcDataset(pl.LightningDataModule):
    def __init__(self,
                 xtcfile : str,
                 tprfile : str,
                 selection : str,
                 dataset_size : int = None,
                 train_prop : float = 0.8,
                 validation_prop : float = 0.2,
                 batch_size : int = None,
                 val_batch_size : int = None,
                 test_batch_size : int = None,
                 num_workers : int = 1,
                 sequential : bool = False,
                 verbose : bool = True,
                 dataset_type : Dataset = XtcData,
                 dataset_args : Dict = {},
                 label_distance : str = None,
                 label_dihedrals : str = None,
                 ):
        super().__init__()
        print(f"\n\n[Initializing XtcDataset Module]") if verbose else None
        print("==========================================") if verbose else None
        print(f"Loading coordinates from file {xtcfile} and topology from file {tprfile}") if verbose else None
        if not os.path.exists(xtcfile):
            raise FileNotFoundError(f"File {xtcfile} not found")
        if not os.path.exists(tprfile):
            raise FileNotFoundError(f"File {tprfile} not found")
        
        # Load the trajectory
        u = mda.Universe(tprfile, xtcfile)

        # Select the atoms
        try:
            mol = u.select_atoms(selection)
        except Exception as e:
            raise ValueError(f"Selection {selection} is not valid: {e}")
        if mol.n_atoms == 0:
            raise ValueError(f"Selection {selection} does not match any atoms in the trajectory")
        
        # Center the and unwrap the trajectory
        transforms = [trans.unwrap(mol),
              trans.center_in_box(mol, wrap=True)]
        u.trajectory.add_transformations(*transforms)
        
        # Extract the atomic numbers
        ch = [at.element for at in mol]
        for at in ch:
            if at not in atomic_numbers:
                raise ValueError(f"Atom {at} not found in atomic numbers dictionary")
        ch = [atomic_numbers[at] for at in ch]
        
        # Get the atom numbers in the trajectory
        atm_ids = [at.id + 1 for at in mol.atoms]
        
        # Get the labels to add to the dataset
        if label_distance is not None:
            print(f"Using label distance: {label_distance}") if verbose else None
            label_atoms = u.select_atoms(label_distance)
            if len(label_atoms) != 2:
                raise ValueError(f"Label distance selection {label_distance} must select exactly 2 atoms, found {len(label_atoms)}")
            self.label_list = ["Distance between " + label_atoms[0].name + " and " + label_atoms[1].name]
        elif label_dihedrals is not None:
            from MDAnalysis.analysis.dihedrals import Dihedral 
            print(f"Using label dihedrals: {label_dihedrals}") if verbose else None
            label_atoms = []
            self.label_list = []
            for res in label_dihedrals.split(','):
                res = res.strip()
                assert len(res) > 4, f"Label dihedral {res} must be at least 5 characters long"
                if res[:4] not in ['phi_', 'psi_']:
                    raise ValueError(f"Label dihedral {res} must start with 'phi_' or 'psi_'")
                resnum = int(res.split('_')[1])
                if res[:4] == 'phi_':
                    sel = u.residues[resnum-1].phi_selection()
                    assert sel is not None, f"Residue {resnum} does not have a phi dihedral"
                    label_atoms.append(sel)
                    self.label_list.append(f"phi_{resnum}")
                elif  res[:4] == 'psi_':
                    sel = u.residues[resnum-1].psi_selection()
                    assert sel is not None, f"Residue {resnum} does not have a psi dihedral"
                    label_atoms.append(sel)
                    self.label_list.append(f"psi_{resnum}")
            
        #label_atoms = u.select_atoms('(name CA and resnum 1) or (name CA and resnum 10)')
        
        labels = []
        mol_traj = []
        s, e = 0, len(u.trajectory)
        if dataset_size is not None and sequential:
            s = random.randint(0, len(u.trajectory) - dataset_size - 1)
            e = dataset_size + s
        print(f"Reading trajectory of {e-s} frames...") if verbose else None
        for ts in tqdm(u.trajectory[s:e], disable=not verbose):
            structure = ase.Atoms(numbers=ch, positions=mol.positions)
            mol_traj.append(structure)
            if label_distance is not None:
                labels.append(np.linalg.norm(label_atoms.positions[1] - label_atoms.positions[0]))
            elif label_dihedrals is not None:
                dih = Dihedral(label_atoms)
                res = dih.run(start=u.trajectory.frame, stop=u.trajectory.frame + 1)
                labels.extend(res.angles)
            else:
                labels.append(0.0)
        print(f"Finished reading trajectory.") if verbose else None
        if dataset_size is not None and not sequential:
            print(f"Subsampling dataset to {dataset_size} frames.") if verbose else None
            assert dataset_size <= len(mol_traj), f"Dataset size {dataset_size} must be less than the number of frames {len(mol_traj)}."
            mol_traj = random.choices(mol_traj, k=dataset_size)

        FRAMES = len(mol_traj)
        assert train_prop + validation_prop <= 1.0
        test_prop = 1.0 - train_prop - validation_prop
        self.train_size = int(FRAMES * train_prop)
        self.validation_size = int(FRAMES * validation_prop)
        self.test_size = FRAMES - self.train_size - self.validation_size

        dataset_args['atm_ids'] = atm_ids
        self.xtcData_full = dataset_type(
                structures=mol_traj,
                labels=labels,
                dtype=torch.float64 if DOUBLE_PRECISION else torch.float32,
                **dataset_args
                )
        self.num_inputs = self.xtcData_full.num_inputs
        self.datapoint_shape = tuple(self.xtcData_full[0][0].shape)
        
        
        
        self.save_hyperparameters()

        if self.hparams.batch_size is None:
            self.hparams.batch_size = int(self.train_size * 0.1)
        if self.hparams.val_batch_size is None:
            self.hparams.val_batch_size = int(self.validation_size * 0.1)
        # Printing
        self.target_scaler = StandardScaler()
        self.target_scaler.fit(self.xtcData_full.get_data()[0])
        assert self.hparams.batch_size < self.train_size, "Batch size must be less than the training size"
        assert self.hparams.val_batch_size < self.validation_size, "Validation batch size must be less than the validation size"

        print(f"Total frames: {FRAMES}, Train size: {self.train_size}, Batch size: {self.hparams.batch_size}, Validation size: {self.validation_size}") if verbose else None
        print("==========================================") if verbose else None


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
            batch_size=self.hparams.test_batch_size if self.hparams.test_batch_size is not None else len(mddata_test),
            shuffle=False,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True)

    def target_scaler(self, X):
        return self.target_scaler.transform(X)

    def target_inverse_scaler(self, X):
        return self.target_scaler.inverse_transform(X)

    def get_scaler_mean(self):
        return self.target_scaler.mean_

    def get_scaler_var(self):
        return self.target_scaler.var_

    def get_scaler_scale(self):
        return self.target_scaler.scale_

    def get_datapoint_shape(self):
        return self.datapoint_shape

