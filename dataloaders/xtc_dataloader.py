from glob import glob
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
from MDAnalysis.lib.distances import calc_dihedrals
import MDAnalysis.transformations as trans

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
import pytorch_lightning as pl



class XtcData(Dataset):
    """XTC dataset"""

    def __init__(
        self,
        structures: List[ase.Atoms],
        labels: List[int],
    ):
        self.positions = [torch.tensor(s.positions).flatten() for s in structures]
        self.num_inputs = len(self.positions[0])

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        x = ()
        x += (self.positions[index],)
        return x
    
    def get_data(self):
        return np.array([d.numpy() for d in self.positions])


class XtcDataset(pl.LightningDataModule):
    def __init__(self,
                 tprfile : str,
                 selection : str,
                 xtcfile : str = None,
                 multicoord : str = None,
                 dataset_size : int = None,
                 train_size : int = 0,
                 validation_size : int = 0,
                 batch_size : int = None,
                 val_batch_size : int = None,
                 test_batch_size : int = None,
                 num_workers : int = 1,
                 sequential : bool = False,
                 verbose : bool = True,
                 dataset_type : str = 'DEFAULT',
                 dataset_args : List[str] = [],
                 label_distance : List[str] = [],
                 label_dihedrals : List[str] = [],
                 norm_type : str = 'standard',
                 ):
        super().__init__()
        print(f"\n\n[Initializing {type(self).__name__} Module]") if verbose else None
        print("==========================================") if verbose else None
        print(f"Loading coordinates from file {xtcfile if xtcfile else multicoord} and topology from file {tprfile}") if verbose else None

        if not os.path.exists(tprfile):
            raise FileNotFoundError(f"File {tprfile} not found")
        
        dataset_args = {k: eval(v) for k, v in (arg.split('=') for arg in dataset_args)}
        
        # Load the trajectory
        assert any([a != None for a in [xtcfile, multicoord]]), "Either xtcfile or multicoord must be provided"
        assert not all([a != None for a in [xtcfile, multicoord]]), "Only one of xtcfile or multicoord can be provided"
        if multicoord:
            xtcfiles = glob(multicoord)
            if not xtcfiles:
                raise FileNotFoundError(f"No files found for pattern {multicoord}")
            u = mda.Universe(tprfile, *xtcfiles)
        else:
            if not os.path.exists(xtcfile):
                raise FileNotFoundError(f"File {xtcfile} not found")
            u = mda.Universe(tprfile, xtcfile)

        # Checks
        if dataset_size:
            assert dataset_size > 0, "Dataset size must be greater than 0"
            assert dataset_size <= len(u.trajectory), f"Dataset size {dataset_size} must be less than the number of frames in the trajectory {len(u.trajectory)}"
        
        # Select the atoms
        try:
            mol = u.select_atoms(selection)
        except Exception as e:
            raise ValueError(f"Selection {selection} is not valid: {e}")
        if mol.n_atoms == 0:
            raise ValueError(f"Selection {selection} does not match any atoms in the trajectory")
        
        # Center the and unwrap the trajectory
        transforms = [trans.unwrap(mol),
                      trans.center_in_box(mol, center='geometry', point=[0.0,0.0,0.0], wrap=False)]
        u.trajectory.add_transformations(*transforms)
    
        # Extract the atomic numbers
        at_elements = [at.element for at in mol]
        self.atns = []
        for elem in at_elements:
            assert elem in atomic_numbers, f"Atom {elem} not found in atomic numbers dictionary"
            self.atns.append(atomic_numbers[elem])

        # Get the atom numbers in the trajectory
        atm_ids = [at.id + 1 for at in mol.atoms]
        
        # Get the labels to add to the dataset
        if len(label_distance) > 0:
            print(f"Using label distance: {label_distance}") if verbose else None
            label_atoms = u.select_atoms(label_distance)
            if len(label_atoms) != 2:
                raise ValueError(f"Label distance selection {label_distance} must select exactly 2 atoms, found {len(label_atoms)}")
            self.label_list = ["Distance between " + label_atoms[0].name + " and " + label_atoms[1].name]
        elif len(label_dihedrals) > 0: 
            print(f"Using label dihedrals: {label_dihedrals}") if verbose else None
            label_atoms = []
            self.label_list = []
            for res in label_dihedrals:
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
        else:
            self.label_list = ['None']
        
        # Extract the bonds information
        self.bonds = mol.get_connections('bonds', outside=False).indices
        for i in range(len(self.bonds)): # remap to mol atoms indices (without hydrogens)
            self.bonds[i] = (np.where(mol.atoms.indices == self.bonds[i][0])[0][0], np.where(mol.atoms.indices == self.bonds[i][1])[0][0])

        # Read the trajectory and store the frames
        labels = []
        mol_traj = []
        if dataset_size is None:
            dataset_size = len(u.trajectory)
        assert dataset_size >= (train_size + validation_size), f"Dataset size {dataset_size} must be greater than or equal to train_size + validation_size = {train_size + validation_size}"
        if sequential:
            s = random.randint(0, len(u.trajectory) - dataset_size)
            e = dataset_size + s
            print(f"Reading trajectory of {e-s} frames...") if verbose else None
            print(f"Trajectory start frame (1-indexed): {s+1}, end: {e}") if verbose else None
            read_frame_seq = [i for i in range(s, e)]
        else:
            read_frame_seq = random.sample(range(len(u.trajectory)), dataset_size)
            read_frame_seq.sort()
            print(f"Reading trajectory of {len(read_frame_seq)} random frames...") if verbose else None

        for idx in tqdm(read_frame_seq, disable=not verbose):
            u.trajectory[idx]
            # Create the ASE structure
            structure = ase.Atoms(numbers=self.atns, positions=mol.atoms.positions)

            # Retain topology information
            residues = [str(r.residue.resname) for r in mol.atoms]
            resids = [r.residue.resid for r in mol.atoms]
            atomnames = [str(a.name) for a in mol.atoms]
            structure.set_array('residuenumbers', np.array(resids))
            structure.set_array('residuenames', np.array(residues))
            structure.set_array('atomtypes', np.array(atomnames))

            mol_traj.append(structure)

            # Get the labels
            if len(label_distance) > 0:
                labels.append(np.linalg.norm(label_atoms.positions[1] - label_atoms.positions[0]))
            elif len(label_dihedrals) > 0:
                dih = []
                for dih_group in label_atoms:
                    res = calc_dihedrals(
                        dih_group.positions[0],
                        dih_group.positions[1],
                        dih_group.positions[2],
                        dih_group.positions[3],
                        dih_group.dimensions
                    )
                    dih.append(res)
                labels.append(dih)
            else:
                labels.append(0.0)
        print(f"Finished reading trajectory.") if verbose else None
        
        
        FRAMES = len(mol_traj)
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = FRAMES - self.train_size - self.validation_size
        print(f"Train size: {self.train_size}, Validation size: {self.validation_size}, Test size: {self.test_size}") if verbose else None
        
        self.mol_traj = mol_traj

        if dataset_type == 'DEFAULT':
            dataset_class = XtcData
        elif dataset_type == 'DISTANCES':
            from datasets.distances_dataset import distancesDataset as dataset_class
            dataset_args['atm_ids'] = atm_ids
        elif dataset_type == 'GRAPH':
            from datasets.graph import graphDataset as dataset_class
            dataset_args['bond_indices'] = self.bonds
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        self.xtcData_full = dataset_class(
                structures=mol_traj,
                labels=labels,
                **dataset_args
                )
        
        self.target_scaler = None

        if dataset_type == 'GRAPH':
            print(f"Loaded graph dataset with {len(self.xtcData_full)} graphs") if verbose else None
            print(f"Number of bonds (nodes): {len(self.bonds)}") if verbose else None
            print(f"Node feature size: {self.xtcData_full[0].x.shape[1]}") if verbose else None
            print(f"Edge feature size: {self.xtcData_full[0].edge_attr.shape[1]}") if verbose else None
            print(f"Total number of edges: {self.xtcData_full[0].edge_index.shape[1]}") if verbose else None
            self.num_inputs = len(self.bonds)
            self.datapoint_shape = (len(self.bonds), self.xtcData_full[0].x.shape[1])
        else:
            self.num_inputs = self.xtcData_full.num_inputs
            self.datapoint_shape = tuple(self.xtcData_full[0][0].shape)
            print(f"Loaded dataset with {len(self.xtcData_full)} frames") if verbose else None
            print(f"Number of atoms: {self.num_inputs}") if verbose else None
            print(f"Datapoint shape: {self.datapoint_shape}") if verbose else None

        self.save_hyperparameters()

        if self.hparams.batch_size is None:
            self.hparams.batch_size = int(self.train_size * 0.1)
        if self.hparams.val_batch_size is None:
            self.hparams.val_batch_size = int(self.validation_size * 0.1)
        # Printing
        assert self.hparams.batch_size < self.train_size, "Batch size must be less than the training size"
        assert self.hparams.val_batch_size < self.validation_size, "Validation batch size must be less than the validation size"

        print(f"Total frames: {FRAMES}, Train size: {self.train_size}, Batch size: {self.hparams.batch_size}, Validation size: {self.validation_size}") if verbose else None
        print("==========================================") if verbose else None

    # def prepare_data(self): # only called on 1 GPU/TPU in distributed

    def setup(self, stage):  # Called on every GPU/TPU in distributed
        # Assign train/val datasets for use in dataloaders
        if self.hparams.sequential:
            self.mddata_train = torch.utils.data.Subset(
                self.xtcData_full, list(range(0, self.train_size)))
            self.mddata_val = torch.utils.data.Subset(
                self.xtcData_full, list(range(self.train_size, self.train_size + self.validation_size)))
            self.mddata_test = torch.utils.data.Subset(
                self.xtcData_full, list(range(self.train_size + self.validation_size, self.train_size + self.validation_size + self.test_size)))
        else:
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
        return self.atoms

    def get_bond_indices(self):
        return self.bonds
    
    def fit_target_scaler(self):
        if self.target_scaler is not None:
            return
        if self.hparams.norm_type == 'standard':
            self.target_scaler = StandardScaler()
        elif self.hparams.norm_type == 'minmax':
            self.target_scaler = MinMaxScaler()
        else:
            raise ValueError(f"Normalization type {self.hparams.norm_type} not supported")

        if self.hparams.dataset_type in ['DEFAULT', 'DISTANCES']:
            self.num_inputs = self.xtcData_full.num_inputs
            self.datapoint_shape = tuple(self.xtcData_full[0][0].shape)
            self.target_scaler.fit(self.xtcData_full.get_data()[0])
        elif self.hparams.dataset_type == 'GRAPH':
            data_to_normalize = [] 
            for g in self.xtcData_full:
                node_feat = g.x.numpy()
                edge_feat = g.edge_attr.numpy()
                data_to_normalize.append(np.hstack([node_feat.mean(axis=0), edge_feat.mean(axis=0)]))
            data_to_normalize = np.vstack(data_to_normalize)
            self.target_scaler.fit(data_to_normalize)
        else:
            raise ValueError(f"Unsupported dataset type for normalization {self.hparams.dataset_type}")
    
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
        dl = self.full_dataloader()
        return next(iter(dl))
    
    def get_dataset(self):
        return self.xtcData_full

        # called on every process in DDP
    def train_dataloader(self):
        if self.hparams.dataset_type == 'GRAPH':
            # For graph dataset we use pyg DataLoader
            return GeoDataLoader(
                self.mddata_train,
                batch_size=self.hparams.batch_size,
                shuffle=not self.hparams.sequential,
                drop_last=True,
                num_workers=self.hparams.num_workers,
                pin_memory=True)
        return DataLoader(
            self.mddata_train,
            batch_size=self.hparams.batch_size,
            shuffle=not self.hparams.sequential,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True)

    def val_dataloader(self):
        if self.hparams.dataset_type == 'GRAPH':
            # For graph dataset we use pyg DataLoader
            return GeoDataLoader(
                self.mddata_val,
                batch_size=self.hparams.val_batch_size,
                shuffle=not self.hparams.sequential,
                drop_last=True,
                num_workers=self.hparams.num_workers,
                pin_memory=True)
        return DataLoader(
            self.mddata_val,
            batch_size=self.hparams.val_batch_size,
            shuffle=not self.hparams.sequential,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True)

    def test_dataloader(self):
        if self.test_size == 0:
            raise ValueError("Test size is 0, cannot create test dataloader")
        if self.hparams.dataset_type == 'GRAPH':
            # For graph dataset we use pyg DataLoader
            return GeoDataLoader(
                self.mddata_test,
                batch_size=self.hparams.test_batch_size if self.hparams.test_batch_size not in [None, 0] else len(
                    self.mddata_test),
                shuffle=not self.hparams.sequential,
                drop_last=True,
                num_workers=self.hparams.num_workers,
                pin_memory=True)
        return DataLoader(
            self.mddata_test,
            batch_size=self.hparams.test_batch_size if self.hparams.test_batch_size not in [None, 0] else len(
                self.mddata_test),
            shuffle=not self.hparams.sequential,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True)

    def full_dataloader(self):
        mddata_test = self.xtcData_full
        if self.hparams.test_batch_size not in [None, 0]:
            batch_size = self.hparams.test_batch_size
        elif self.hparams.batch_size not in [None, 0]:
            batch_size = self.hparams.batch_size
        else:
            batch_size = len(mddata_test)
        if self.hparams.dataset_type == 'GRAPH':
            # For graph dataset we use pyg DataLoader
            return GeoDataLoader(
                mddata_test,
                batch_size=batch_size,
                shuffle=not self.hparams.sequential,
                drop_last=True,
                num_workers=self.hparams.num_workers,
                pin_memory=True)
        return DataLoader(
            mddata_test,
            batch_size=batch_size,
            shuffle=not self.hparams.sequential,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True)

    def target_scaler(self, X):
        self.fit_target_scaler()
        return self.target_scaler.transform(X)

    def target_inverse_scaler(self, X):
        self.fit_target_scaler()
        return self.target_scaler.inverse_transform(X)

    def get_scaler_mean(self):
        self.fit_target_scaler()
        return self.target_scaler.mean_

    def get_scaler_var(self):
        self.fit_target_scaler()
        return self.target_scaler.var_

    def get_scaler_scale(self):
        self.fit_target_scaler()
        return self.target_scaler.scale_

    def get_datapoint_shape(self):
        return self.datapoint_shape
