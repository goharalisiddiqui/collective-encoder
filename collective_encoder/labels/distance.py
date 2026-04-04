import MDAnalysis as mda
from typing import Dict, List, Union

from .base import BaseLabeler

class DistanceValueLabeler(BaseLabeler):
    '''
    Docstring for DistanceValueLabeler
    '''
    _IDENTIFIER = "DISTANCE"
    
    def __init__(self, 
                 universe: mda.Universe,
                 args: Dict[str, Union[str, List[str]]]
                 ) -> None:
        selections = args.get("selections", None)
        if selections is None or len(selections.strip()) == 0:
            raise ValueError(f"[{type(self).__name__}] 'selections' must be provided in args")
        self.dist_atoms = []
        for selection in selections:
            atoms = universe.select_atoms(selection)
            if len(atoms) != 2:
                raise ValueError(f"[{type(self).__name__}] Each selection must select exactly 2 atoms, but selection '{selection}' selected {len(atoms)} atoms")
            self.dist_atoms.append(atoms)

        if len(self.dist_atoms) == 0:
            raise ValueError(f"[{type(self).__name__}] No valid atom pairs found")


    def get_names(self) -> List[str]:
        return [f'distance_value_{i+1}' for i in range(len(self.dist_atoms))]

    def compute(self) -> List[float]:
        from MDAnalysis.lib.distances import distance_array

        dists = []
        for atoms in self.dist_atoms:
            dist = distance_array(
                atoms[0].position.reshape(1, 3),
                atoms[1].position.reshape(1, 3)
            ).flatten()[0]
            dists.append(dist)
        return dists
