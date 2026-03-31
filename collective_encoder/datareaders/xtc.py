import os

from glob import glob
from typing import List, Dict, Union
from tqdm import tqdm

import numpy as np
import ase

import MDAnalysis as mda

from MDAnalysis.exceptions import NoDataError

from .base import TrajectoryReaderBase

from collective_encoder.labels.resolver import get_labeler

class XTCReader(TrajectoryReaderBase):
    def __init__(self,
                 tprfile : str,
                 xtcfile : str = None,
                 xtcfiles : List[str] = None,
                 coord_glob : str = None,
                 selection : str = "all",
                 type_to_elements : List[int] = None,
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        
        # Check files
        self.check_exists(tprfile=tprfile)
        self.log_msg(f"Loading topology from file {tprfile}")
        self.check_mutually_exclusive(xtcfile=xtcfile, 
                                      coord_glob=coord_glob, 
                                      xtcfiles=xtcfiles, 
                                      require_one=True)
        if coord_glob:
            self.log_msg(f"Loading trajectory files matching glob pattern: {coord_glob}") 
            xtcfiles = glob(coord_glob)
            if not xtcfiles:
                raise FileNotFoundError(f"No files found for pattern {coord_glob}")
            self.log_msg(f"Found {len(xtcfiles)} files") 
            u = mda.Universe(tprfile, *xtcfiles)
        elif xtcfiles:
            self.log_msg(f"Loading trajectory from multiple files: \n\t - {('\n\t - '.join(xtcfiles))}") 
            for xf in xtcfiles:
                if not os.path.exists(xf):
                    raise FileNotFoundError(f"File {xf} not found")
            u = mda.Universe(tprfile, *xtcfiles)
        else:
            self.log_msg(f"Loading trajectory from file: {xtcfile}") 
            if not os.path.exists(xtcfile):
                raise FileNotFoundError(f"File {xtcfile} not found")
            u = mda.Universe(tprfile, xtcfile)

        # Select the atoms
        self.mol = self.mda_select_atoms(u, selection)
        self.u = u
        
        # Extract the atomic numbers
        self.atns, self.at_elements = self.mda_get_atomic_numbers_and_elements(
                                                self.mol, type_to_elements)

        # Get the atom numbers in the trajectory
        self.atm_ids = self.mda_get_atom_ids(self.mol)
        
        # Extract the bonds information
        self.bonds = self.mda_get_bonds(self.mol)
    
    def read_trajectory(self, 
                        indices: List[List[int]],
                        labeler_type : str = 'Dummy',
                        labeler_args : Dict[str, Union[str, float, List[int]]] = {},
                        ):
        # Center the and unwrap the trajectory
        self.u = self.mda_add_default_transforms(self.u, self.mol)
    
        # Get the labels from labeler
        labeler_cls = get_labeler(labeler_type)
        labeler = labeler_cls(
            universe=self.u,
            args=labeler_args,
        )
        self.label_list = labeler.get_names()
        
        self.no_residues, self.no_resids, self.no_atomnames = False, False, False

        trajs, labels = (), ()
        self.log_msg(f"Reading trajectories...")
        for seq in tqdm(indices, disable=not self.verbose, leave=False):
            traj, label = self._read_and_label(seq, labeler)
            trajs += (traj,)
            labels += (label,)
        self.log_msg(f"Finished reading trajectories.")
        return trajs, labels

    def get_total_frames(self):
        return len(self.u.trajectory)
    
    def get_residue_info(self):
        if not self.no_residues:
            try:
                residues = [str(r.residue.resname) for r in self.mol.atoms]
                self.no_residues = False
            except NoDataError:
                self.log_msg(f"[{type(self).__name__}] Warning: Residue names not found in trajectory. Using UNK for all") if self.verbose else None
                self.no_residues = True
                residues = ['UNK' for _ in self.mol.atoms]
        else:
            residues = ['UNK' for _ in self.mol.atoms]
        
        if not self.no_resids:
            try:
                resids = [r.residue.resid for r in self.mol.atoms]
                self.no_resids = False
            except NoDataError:
                self.log_msg(f"[{type(self).__name__}] Warning: Residue IDs not found in trajectory. Using 0 for all") if self.verbose else None
                self.no_resids = True
                resids = [0 for _ in self.mol.atoms]
        else:
            resids = [0 for _ in self.mol.atoms]
        if not self.no_atomnames:
            try:
                atomnames = [str(a.name) for a in self.mol.atoms]
                self.no_atomnames = False
            except NoDataError:
                self.log_msg(f"[{type(self).__name__}] Warning: Atom names not found in trajectory. Using element names.") if self.verbose else None
                self.no_atomnames = True
                atomnames = self.at_elements
        else:
            atomnames = self.at_elements
        
        return residues, resids, atomnames

    def _read_and_label(self, indices, labeler):
        # Read the trajectory and store the frames
        labels = []
        mol_traj = []
        
        for idx in tqdm(indices, disable=not self.verbose):
            try:
                self.u.trajectory[idx]
            except OSError:
                self.log_msg(f"[{type(self).__name__}] Warning: Could not read frame {idx} from trajectory. Skipping frame.") if self.verbose else None
                continue

            # Create the ASE structure
            structure = ase.Atoms(numbers=self.atns, 
                                  positions=self.mol.atoms.positions,
                                  cell=self.mol.dimensions[:3],)
            # Set periodic boundary conditions if box dimensions are non-zero
            if not np.all(self.mol.dimensions[:3] == 0):
                structure.set_pbc([True, True, True])

            # Retain topology information if available
            residues, resids, atomnames = self.get_residue_info()
            structure.set_array('residuenames', np.array(residues))
            structure.set_array('residuenumbers', np.array(resids))
            structure.set_array('atomtypes', np.array(atomnames))

            # Compute the labels for this frame
            label = labeler.compute()
            
            mol_traj.append(structure)
            labels.append(label)
        
        return mol_traj, labels
