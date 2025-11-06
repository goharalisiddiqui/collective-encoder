import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) #FIXME: This is weird hack to import from parent dir. Need to write a proper package

from glob import glob
from typing import List, Dict, Union
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
from MDAnalysis.exceptions import NoDataError

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
import pytorch_lightning as pl

from warnings import warn

class XtcData(Dataset):
    """XTC dataset"""

    def __init__(
        self,
        structures: List[ase.Atoms],
        labels: List[List[float]],
    ):
        self.positions = [torch.tensor(s.positions).flatten() for s in structures]
        self.labels = [torch.tensor(l).flatten() for l in labels]
        self.num_inputs = len(self.positions[0])

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        x = ()
        x += (self.positions[index],self.labels[index])
        return x
    
    def get_data(self):
        return np.array([d.numpy() for d in self.positions]), np.array([l.numpy() for l in self.labels])

class XtcDataset(pl.LightningDataModule):
    def __init__(self,
                 tprfile : str,
                 selection : str,
                 xtcfile : str = None,
                 xtcfiles : List[str] = None,
                 coord_glob : str = None,
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
                 dataset_args : Dict[str, Union[str, int]] = {},
                 labeler: str = None,
                 labeler_args: Dict[str, Union[str, int]] = {},
                 norm_type : str = 'standard',
                 test_full_dataset : bool = False,
                 type_to_elements : List[int] = None,
                 ):
        super().__init__()
        print(f"\n\n[Initializing {type(self).__name__} Module]") if verbose else None
        print("==========================================") if verbose else None
        print(f"Loading topology from file {tprfile}") if verbose else None

        # Checks
        if not os.path.exists(tprfile):
            raise FileNotFoundError(f"File {tprfile} not found")
        if train_size <= 0 or validation_size <= 0:
            raise ValueError("Train size and validation size must be greater than 0")
        
        # dataset_args = {k: eval(v) for k, v in (arg.split('=') for arg in dataset_args)}
        
        # Load the trajectory
        assert any([a != None for a in [xtcfile, coord_glob, xtcfiles]]), "One of xtcfile, coord_glob or xtcfiles must be provided"
        if coord_glob:
            print(f"Loading trajectory files matching glob pattern: {coord_glob}") if verbose else None
            xtcfiles = glob(coord_glob)
            if not xtcfiles:
                raise FileNotFoundError(f"No files found for pattern {coord_glob}")
            print(f"Found {len(xtcfiles)} files") if verbose else None
            u = mda.Universe(tprfile, *xtcfiles)
        elif xtcfiles:
            print(f"Loading trajectory from multiple files: \n\t - {('\n\t - '.join(xtcfiles))}") if verbose else None
            for xf in xtcfiles:
                if not os.path.exists(xf):
                    raise FileNotFoundError(f"File {xf} not found")
            u = mda.Universe(tprfile, *xtcfiles)
        elif xtcfile:
            print(f"Loading trajectory from file: {xtcfile}") if verbose else None
            if not os.path.exists(xtcfile):
                raise FileNotFoundError(f"File {xtcfile} not found")
            u = mda.Universe(tprfile, xtcfile)
        else:
            raise ValueError("One of xtcfile, coord_glob or xtcfiles must be provided")

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
        try:
            at_elements = [at.element for at in mol]
        except NoDataError:
            if type_to_elements is None:
                raise ValueError("Atom elements not found in trajectory. Please provide type_to_elements mapping.")
            at_elements = []
            for at in mol.atoms:
                type_index = int(at.type) - 1
                if type_index < 0 or type_index >= len(type_to_elements):
                    raise ValueError(f"Atom type {at.type} is out of bounds for provided type_to_elements mapping")
                at_elements.append(type_to_elements[type_index])
        self.atns = []
        for elem in at_elements:
            assert elem in atomic_numbers, f"Atom {elem} not found in atomic numbers dictionary"
            self.atns.append(atomic_numbers[elem])

        # Get the atom numbers in the trajectory
        atm_ids = [at.id + 1 for at in mol.atoms]
        
        # Get the labels from labeler
        if labeler is not None:
            print(f"Using custom labeler: {labeler} with args {labeler_args}") if verbose else None
            if labeler == 'CoordinationCountLabeler':
                from labels.coordination import CoordinationCountLabeler as labeler_class
            elif labeler == 'DistanceValueLabeler':
                from labels.distance import DistanceValueLabeler as labeler_class
            elif labeler == 'DihedralValueLabeler':
                from labels.dihedral import DihedralValueLabeler as labeler_class
            else:
                raise ValueError(f"Unknown labeler {labeler}")
            coord_labeler = labeler_class(
                universe=u,
                args=labeler_args,
            )
            self.label_list = coord_labeler.get_names()
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
            dataset_size = (train_size + validation_size)
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

        top_warn_flags = [False, False, False]  # resname, resid, atomname
        for idx in tqdm(read_frame_seq, disable=not verbose):
            try:
                u.trajectory[idx]
            except OSError:
                print(f"[{type(self).__name__}] Warning: Could not read frame {idx} from trajectory. Skipping frame.") if verbose else None
                continue
            # Create the ASE structure
            structure = ase.Atoms(numbers=self.atns, 
                                  positions=mol.atoms.positions,
                                  cell=mol.dimensions[:3],)
            if not np.all(mol.dimensions[:3] == 0):
                structure.set_pbc([True, True, True])

            # Retain topology information if available
            try:
                residues = [str(r.residue.resname) for r in mol.atoms]
            except NoDataError:
                print(f"[{type(self).__name__}] Warning: Residue names not found in trajectory. Using UNK for all") if verbose and not top_warn_flags[0] else None
                top_warn_flags[0] = True
                residues = ['UNK' for _ in mol.atoms]
            try:
                resids = [r.residue.resid for r in mol.atoms]
            except NoDataError:
                print(f"[{type(self).__name__}] Warning: Residue IDs not found in trajectory. Using 0 for all") if verbose and not top_warn_flags[1] else None
                top_warn_flags[1] = True
                resids = [0 for _ in mol.atoms]
            try:
                atomnames = [str(a.name) for a in mol.atoms]
            except NoDataError:
                print(f"[{type(self).__name__}] Warning: Atom names not found in trajectory. Using element names.") if verbose and not top_warn_flags[2] else None
                top_warn_flags[2] = True
                atomnames = at_elements

            structure.set_array('residuenames', np.array(residues))
            structure.set_array('residuenumbers', np.array(resids))
            structure.set_array('atomtypes', np.array(atomnames))

            mol_traj.append(structure)

            # Get the labels
            if labeler is not None:
                labels.append(coord_labeler.compute())
            else:
                labels.append([0.0])
        print(f"Finished reading trajectory.") if verbose else None
        
        FRAMES = len(mol_traj)
        print(f"Total frames read: {FRAMES}") if verbose else None
        if FRAMES < (train_size + validation_size): # In case some frames could not be read
            print(f"[{type(self).__name__}] Warning: Only {FRAMES} frames read, "
                  f"which is less than train_size + validation_size "
                  f"= {train_size + validation_size}. "
                  f"Adjusting train size accordingly.") if verbose else None
            train_size = int(FRAMES * (train_size / (train_size + validation_size)))
            validation_size = FRAMES - train_size
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = FRAMES - self.train_size - self.validation_size
        print(f"Train size: {self.train_size}, Validation size: {self.validation_size}, Test size: {self.test_size}") if verbose else None
        
        self.mol_traj = mol_traj

        if dataset_type == 'DEFAULT':
            dataset_class = XtcData
        elif dataset_type == 'DISTANCES':
            from datasets.distances import DistancesDataset as dataset_class
            dataset_args['atm_ids'] = atm_ids
        elif dataset_type == 'GRAPH':
            from datasets.bondgraph import BondGraphDataset as dataset_class
            dataset_args['bond_indices'] = self.bonds
        elif dataset_type == 'SOAP':
            from datasets.soap import SOAPDataset as dataset_class
        elif dataset_type == 'SOAP_PS':
            from datasets.soap_ps import SoapPowerSpectrumDataset as dataset_class
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # To enable selecting specific atoms for SOAP descriptors using MDAnalysis selections
        if 'SOAP' in dataset_type:
            if dataset_args.get('atoms_selections', None) is not None:
                selected_indices = []
                for selection in dataset_args['atoms_selections']:
                    sel_atoms = u.select_atoms(selection)
                    if sel_atoms.n_atoms != 1:
                        print(f"[{type(self).__name__}] WARNING! Selection {selection} does not select exactly one atom, selected {sel_atoms.n_atoms} atoms")
                    n_types = len(set([at.type for at in sel_atoms]))
                    if n_types > 1:
                        print(f"[{type(self).__name__}] WARNING! Selection {selection} selects more than one atom type.)")
                    print(f"Selected {sel_atoms.n_atoms} atoms of total {n_types} types") if verbose else None
                    for at in sel_atoms:
                        mol_index = np.where(mol.atoms.indices == at.index)[0]
                        if len(mol_index) == 0:
                            raise ValueError(f"Atom {at.index} in selection {selection} not found in selected molecule atoms")
                        selected_indices.append(int(mol_index[0]))
                dataset_args['selected_atoms'] = dataset_args.get('selected_atoms', []) + selected_indices
                dataset_args.pop('atoms_selections')

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
        assert self.hparams.batch_size <= self.train_size, "Batch size must be less than the training size"
        assert self.hparams.val_batch_size <= self.validation_size, "Validation batch size must be less than the validation size"

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
        return self.atns

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

        if self.hparams.dataset_type in ['DEFAULT', 'DISTANCES', 'SOAP', 'SOAP_PS']:
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
                shuffle=False,
                drop_last=True,
                num_workers=self.hparams.num_workers,
                pin_memory=True)
        return DataLoader(
            self.mddata_val,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True)

    def test_dataloader(self):
        if self.hparams.test_full_dataset:
            return self.full_dataloader()
        if self.test_size == 0:
            raise ValueError("Test size is 0, cannot create test dataloader")
        if self.hparams.dataset_type == 'GRAPH':
            # For graph dataset we use pyg DataLoader
            return GeoDataLoader(
                self.mddata_test,
                batch_size=self.hparams.test_batch_size if self.hparams.test_batch_size not in [None, 0] else len(
                    self.mddata_test),
                shuffle=False,
                drop_last=True,
                num_workers=self.hparams.num_workers,
                pin_memory=True)
        return DataLoader(
            self.mddata_test,
            batch_size=self.hparams.test_batch_size if self.hparams.test_batch_size not in [None, 0] else len(
                self.mddata_test),
            shuffle=False,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True)

    def full_dataloader(self):
        mddata_test = self.xtcData_full
        batch_size = len(mddata_test)
        if self.hparams.dataset_type == 'GRAPH':
            # For graph dataset we use pyg DataLoader
            return GeoDataLoader(
                mddata_test,
                batch_size=batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=self.hparams.num_workers,
                pin_memory=True)
        return DataLoader(
            mddata_test,
            batch_size=batch_size,
            shuffle=False,
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
        from metatomic.torch import System
        fake_systems = [
            System(
                types=torch.tensor(at_types, dtype=torch.long),
                positions=torch.tensor(self.mol_traj[0].positions, dtype=torch.float64),
                cell=torch.tensor(self.mol_traj[0].get_cell(), dtype=torch.float64),
                pbc=torch.tensor([True, True, True]),),
            System(
                types=torch.tensor(at_types, dtype=torch.long),
                positions=torch.tensor(self.mol_traj[1].positions, dtype=torch.float64),
                cell=torch.tensor(self.mol_traj[1].get_cell(), dtype=torch.float64),
                pbc=torch.tensor([True, True, True]),)
            ]

        return fake_systems