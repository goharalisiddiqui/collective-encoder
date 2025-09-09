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
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from datasets.distances_dataset import distancesDataset
from datasets.graph import graphDataset

DOUBLE_PRECISION = False

class XtcData(Dataset):
    """XTC dataset"""

    def __init__(
        self,
        structures: List[ase.Atoms],
        labels: List[int],
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
    
    def get_data(self):
        return np.array([d.numpy() for d in self.positions])


def xtcdatset_args():
    desc = "Xtc Dataset Arguments"
    parser = argparse.ArgumentParser(description=desc)


    parser.add_argument('--xtcfile', required=True, type=str,
                        help='Input compressed coordinate file')
    parser.add_argument('--tprfile', required=True, type=str,
                        help='Input binary file containing the topology')
    parser.add_argument('--selection', required=True, type=str,
                        help='Selection string of mdanalysis')
    parser.add_argument('--datasize', dest="dataset_size",
                        type=int, default=None, help='Size of the dataset to use')
    parser.add_argument('--sequential', dest="sequential",
                        action='store_true', 
                        help='Weather the trajectory should be sequential or not')
    
    label_group = parser.add_mutually_exclusive_group()
    label_group.add_argument('--label_distance', dest='label_distance', 
                             type=str, default=None, 
                             help='Selection string for md analysis. ' \
                             'Must select exactly 2 atoms. ' \
                             'Distance between these atoms will be used as a label.')
    label_group.add_argument('--label_dihedrals', dest='label_dihedrals', 
                             type=str, default=None, 
                             help='Comma separated list of DIH_RESID ' \
                             'e.g. "phi_1" will compute the phi dihedral angle ' \
                             'for residue 1 and use it as a label')
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
        print(
            f"Loading coordinates from file {xtcfile} and topology from file {tprfile}") if verbose else None
        if not os.path.exists(xtcfile):
            raise FileNotFoundError(f"File {xtcfile} not found")
        if not os.path.exists(tprfile):
            raise FileNotFoundError(f"File {tprfile} not found")
        
        # Load the trajectory
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
        if label_distance is not None:
            print(f"Using label distance: {label_distance}") if verbose else None
            label_atoms = u.select_atoms(label_distance)
            if len(label_atoms) != 2:
                raise ValueError(f"Label distance selection {label_distance} must select exactly 2 atoms, found {len(label_atoms)}")
            self.label_list = ["Distance between " + label_atoms[0].name + " and " + label_atoms[1].name]
        elif label_dihedrals is not None: 
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
        else:
            self.label_list = ['None']
        
        # Extract the bonds information
        self.bonds = mol.get_connections('bonds', outside=False).indices
        for i in range(len(self.bonds)):
            self.bonds[i] = (np.where(mol.atoms.indices == self.bonds[i][0])[0][0], np.where(mol.atoms.indices == self.bonds[i][1])[0][0])

        # Read the trajectory and store the frames
        labels = []
        mol_traj = []
        if dataset_size is None:
            dataset_size = len(u.trajectory)
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
            if label_distance is not None:
                labels.append(np.linalg.norm(label_atoms.positions[1] - label_atoms.positions[0]))
            elif label_dihedrals is not None:
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
        assert train_prop + validation_prop <= 1.0
        self.train_size = int(FRAMES * train_prop)
        self.validation_size = int(FRAMES * validation_prop)
        self.test_size = FRAMES - self.train_size - self.validation_size
        
        self.mol_traj = mol_traj

        if dataset_type == distancesDataset:
            dataset_args['atm_ids'] = atm_ids
        if dataset_type == graphDataset:
            dataset_args['bond_indices'] = self.bonds
        self.xtcData_full = dataset_type(
                structures=mol_traj,
                labels=labels,
                dtype=torch.float64 if DOUBLE_PRECISION else torch.float32,
                **dataset_args
                )
        
        self.target_scaler = StandardScaler()
        if dataset_type in [XtcData, distancesDataset]:
            self.num_inputs = self.xtcData_full.num_inputs
            self.datapoint_shape = tuple(self.xtcData_full[0][0].shape)
            self.target_scaler.fit(self.xtcData_full.get_data()[0])
        
        if dataset_type == graphDataset:
            print(f"Loaded graph dataset with {len(self.xtcData_full)} graphs") if verbose else None
            print(f"Number of bonds (nodes): {len(self.bonds)}") if verbose else None
            print(f"Node feature size: {self.xtcData_full[0].x.shape[1]}") if verbose else None
            print(f"Edge feature size: {self.xtcData_full[0].edge_attr.shape[1]}") if verbose else None
            print(f"Total number of edges: {self.xtcData_full[0].edge_index.shape[1]}") if verbose else None
            self.num_inputs = len(self.bonds)
            self.datapoint_shape = (len(self.bonds), self.xtcData_full[0].x.shape[1])

            data_to_normalize = []
            for g in self.xtcData_full:
                node_feat = g.x.numpy()
                edge_feat = g.edge_attr.numpy()
                data_to_normalize.append(np.hstack([node_feat.flatten(), edge_feat.flatten()]))
            data_to_normalize = np.vstack(data_to_normalize)
            self.target_scaler.fit(data_to_normalize)

        print(self.get_scaler_mean().shape)
        print(self.get_scaler_var().shape) 
        
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
