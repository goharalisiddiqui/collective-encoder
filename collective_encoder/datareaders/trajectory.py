from abc import ABC, abstractmethod
from typing import List, Tuple, Tuple, Dict, Any

import numpy as np
import ase

import MDAnalysis.transformations as trans
from MDAnalysis.exceptions import NoDataError

from collective_encoder.datareaders.base import BaseDataReader

class TrajectoryReaderBase(BaseDataReader, ABC):
    '''
    Abstract base class for trajectory readers.
    '''
    def __init__(self,
                 args: Dict[str, Any] = None,
                 **kwargs):
        super().__init__(args=args, **kwargs)

    @abstractmethod
    def read_trajectory(self) -> Tuple[List[ase.Atoms], List[List[float]]]:
        '''
        Abstract method to read the trajectory and return 
        a tuple of ASE Atoms objects and their corresponding labels.
        '''
        raise NotImplementedError("Subclasses must implement read_trajectory method")
    
    def get_atomic_numbers(self) -> List[int]:
        return self.atns
    
    def get_element_symbols(self) -> List[str]:
        return self.at_elements
    
    def get_atom_ids(self) -> List[int]:
        return self.atm_ids
    
    def get_bonds(self) -> List[Tuple[int, int]]:
        return self.bonds
    
    def mda_select_atoms(self, universe, selection: str):
        '''
        Select atoms from the MDAnalysis universe.
        '''
        try:
            mol = universe.select_atoms(selection)
        except Exception as e:
            raise ValueError(f"Selection {selection} is not valid: {e}")
        if mol.n_atoms == 0:
            raise ValueError(f"Selection {selection} does not match any atoms in the trajectory")
        return mol
    
    def mda_add_default_transforms(self, universe, mol):
        '''
        Add default transformations to the MDAnalysis universe.
        '''
        transforms = [trans.unwrap(mol),
                      trans.center_in_box(mol, center='geometry', point=[0.0,0.0,0.0], wrap=False)]
        universe.trajectory.add_transformations(*transforms)
        return universe
    
    def mda_get_atomic_numbers_and_elements(self, mol, type_to_elements: list = None):
        '''
        Get atomic numbers from the MDAnalysis atom group.
        '''
        from ase.data import atomic_numbers
        try:
            at_elements = [at.element for at in mol]
        except NoDataError:
            if type_to_elements is None:
                raise ValueError("Atom elements not found in trajectory. Please provide type_to_elements mapping.")
            at_types = mol.types
            at_elements = [type_to_elements[at] for at in at_types]
        at_numbers = [atomic_numbers[el] for el in at_elements]
        return at_numbers, at_elements
    
    def mda_get_atom_ids(self, mol):
        '''
        Get atom IDs from the MDAnalysis atom group.
        '''
        atm_ids = [at.id + 1 for at in mol.atoms]
        return atm_ids
    
    def mda_get_bonds(self, mol):
        '''
        Get bonds from the MDAnalysis atom group.
        '''
        bonds = mol.get_connections('bonds', outside=False).indices
        for i in range(len(bonds)): # remap to mol atoms indices (without hydrogens)
            bonds[i] = (np.where(mol.atoms.indices == bonds[i][0])[0][0], np.where(mol.atoms.indices == bonds[i][1])[0][0])
        return bonds