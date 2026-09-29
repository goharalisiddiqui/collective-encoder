from abc import ABC, abstractmethod
from typing import List, Tuple, Tuple, Dict, Any

import numpy as np
import ase

import MDAnalysis.transformations as trans
from MDAnalysis.exceptions import NoDataError

from collective_encoder.datareaders.base import BaseDataReader

class TrajectoryReaderBase(BaseDataReader, ABC):
    """
    Abstract base class for trajectory readers.

    Provides common MDAnalysis functionality for selecting atoms,
    unwrapping trajectories, and extracting topological data (elements,
    atomic numbers, bonds) for downstream models.
    """
    _OPTIONAL_ARGS = {
        'selection': "all",
        'type_to_elements': None,
        'parallel': True,
    }

    @abstractmethod
    def read_trajectory(self) -> Tuple[List[ase.Atoms], List[List[float]]]:
        """
        Abstract method to read the trajectory and return 
        a tuple of ASE Atoms objects and their corresponding labels.

        Returns
        -------
        tuple
            Tuple containing list of `ase.Atoms` and their corresponding labels.
        """
        pass
    
    def get_atomic_numbers(self) -> List[int]:
        """
        Get the extracted atomic numbers.

        Returns
        -------
        list of int
            List of atomic numbers for selected atoms.
        """
        return self.atns
    
    def get_element_symbols(self) -> List[str]:
        """
        Get the extracted element symbols.

        Returns
        -------
        list of str
            List of element symbols for selected atoms.
        """
        return self.at_elements
    
    def get_atom_ids(self) -> List[int]:
        """
        Get the original atom IDs.

        Returns
        -------
        list of int
            List of 1-indexed atom IDs from the topology.
        """
        return self.atm_ids
    
    def get_bonds(self) -> List[Tuple[int, int]]:
        """
        Get the bond connectivity for selected atoms.

        Returns
        -------
        list of tuple
            List of (atom_i, atom_j) index pairs.
        """
        return self.bonds
    
    def select_atoms(self):
        """
        Select atoms from the MDAnalysis universe based on `self.selection`.

        Raises
        ------
        ValueError
            If the selection string is invalid or matches zero atoms.
        """
        try:
            mol = self.u.select_atoms(self.selection)
        except Exception as e:
            raise ValueError(f"Selection {self.selection} is not valid: {e}")
        if mol.n_atoms == 0:
            raise ValueError(f"Selection {self.selection} does not match any atoms in the trajectory")
        self.mol = mol
    
    def mda_add_default_transforms(self, universe, mol):
        """
        Add default transformations (unwrapping, centering) to the universe.

        Parameters
        ----------
        universe : MDAnalysis.Universe
            The MDAnalysis universe.
        mol : MDAnalysis.AtomGroup
            The selected atom group used for centering.

        Returns
        -------
        MDAnalysis.Universe
            The modified universe object.
        """
        transforms = [trans.unwrap(mol),
                      trans.center_in_box(mol, center='geometry', point=[0.0,0.0,0.0], wrap=False)]
        universe.trajectory.add_transformations(*transforms)
        return universe
    
    def extract_topology_info(self):
        """
        Extract atomic numbers, elements, atom ids, and bonds from the MDAnalysis atom group.
        """
        self._extract_elements()
        self._extract_atomic_numbers()
        self._extract_atom_ids()
        self._extract_bonds()
        
        self.ce_log_dict("Extracted topology: ", {
            "Number of atoms": len(self.atns),
            "Number of bonds": len(self.bonds),
            "Element Counts": {
                a: self.at_elements.count(a) for a in set(self.at_elements)
            }
        })
            
    def _extract_elements(self):
        """
        Extract atomic elements from the MDAnalysis atom group.
        Falls back to `type_to_elements` if MDAnalysis cannot deduce them.
        """
        try:
            at_elements = [at.element for at in self.mol]
        except NoDataError:
            if self.type_to_elements is None:
                raise ValueError("Atom elements not found in trajectory. Please provide type_to_elements mapping.")
            at_types = self.mol.types
            at_elements = [self.type_to_elements[at] for at in at_types]
            # Add elements information back to the universe
            try:
                if hasattr(self.u.atoms, "types"):
                    all_elements = [self.type_to_elements.get(t, t) for t in self.u.atoms.types]
                    self.u.add_TopologyAttr("elements", all_elements)
            except Exception as e:
                self.log_warn(f"Could not add elements information to the universe: {e}",)
        self.at_elements = at_elements

    def _extract_atomic_numbers(self):
        """
        Get atomic numbers derived from elements.
        """
        from ase.data import atomic_numbers
        at_numbers = [atomic_numbers[el] for el in self.at_elements]
        self.atns = at_numbers
    
    def _extract_atom_ids(self):
        """
        Get 1-based atom IDs from the MDAnalysis atom group.
        """
        atm_ids = [int(at.id) + 1 for at in self.mol.atoms]
        self.atm_ids = atm_ids
    
    def _extract_bonds(self):
        """
        Get bonds from the MDAnalysis atom group and remap indices relative to the selection.
        """
        bonds = self.mol.get_connections('bonds', outside=False).indices
        for i in range(len(bonds)): # remap to mol atoms indices (without hydrogens)
            bonds[i] = (np.where(self.mol.atoms.indices == bonds[i][0])[0][0], np.where(self.mol.atoms.indices == bonds[i][1])[0][0])
        self.bonds = bonds