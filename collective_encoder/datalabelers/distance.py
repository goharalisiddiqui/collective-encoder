import MDAnalysis as mda
from typing import Dict, List, Union

from .base import FrameLabeler


class DistanceValueLabeler(FrameLabeler):
    """Compute inter-atomic distances for the current trajectory frame.

    Each entry in ``selections`` must select exactly two atoms; the Euclidean
    distance between them is returned as a label column.

    Args:
        universe: MDAnalysis Universe positioned at the target frame.
        args: Configuration dict with key ``selections`` (required) — a list
            of MDAnalysis selection strings, each selecting exactly 2 atoms
            (e.g. ``['resid 1 and name CA', 'resid 5 and name CA']``).
    """

    _IDENTIFIER = "DISTANCE"
    _REQUIRED_ARGS = ["selections"]
    _OPTIONAL_ARGS = {}

    def __init__(self,
                 universe: mda.Universe,
                 args: Dict[str, Union[str, List[str]]],
                 **kwargs
                 ) -> None:
        super().__init__(args=args, **kwargs)
        if len(self.selections) == 0:
            self.raise_error("'selections' must be provided in args")
        self.dist_atoms = []
        for selection in self.selections:
            atoms = universe.select_atoms(selection)
            if len(atoms) != 2:
                self.raise_error(f"Each selection must select exactly 2 atoms, but selection '{selection}' selected {len(atoms)} atoms")
            self.dist_atoms.append(atoms)

        if len(self.dist_atoms) == 0:
            self.raise_error("No valid atom pairs found")

    def get_label_names(self) -> List[str]:
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
