import numpy as np
import MDAnalysis as mda
from typing import Optional, Dict, List, Union

from .base import BaseLabeler
from .mda import MDALabeler


class DistanceValueLabeler(MDALabeler):
    """
    Compute inter-atomic distances for the current trajectory frame.

    Each entry in ``selections`` must select exactly two atoms; the Euclidean
    distance between them is returned in the label vector under the labeler's key.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        MDAnalysis Universe positioned at the target frame.
    args : dict
        Configuration dict with key ``selections`` (required) — a list
        of MDAnalysis selection strings, each selecting exactly 2 atoms
        (e.g. ``['resid 1 and name CA', 'resid 5 and name CA']``) — and
        optional ``name`` (str).
    kwargs : dict
        Additional keyword arguments forwarded to the base class.
    """

    _IDENTIFIER = "DISTANCE"
    _REQUIRED_ARGS = ["selections"]
    _OPTIONAL_ARGS = BaseLabeler._OPTIONAL_ARGS.copy()

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
        """
        Return the dictionary key name for this labeler.

        Returns
        -------
        list of str
            Single-element list containing ``self.get_default_label_name()``.
        """
        return [self.get_default_label_name()]

    def compute(self, indices: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
        """
        Compute inter-atomic distances for the current frame.

        Parameters
        ----------
        indices : list of int, optional
            Unused frame indices argument for interface compatibility.

        Returns
        -------
        dict of str to numpy.ndarray
            Dictionary mapping the label name to a 1-D NumPy array of Euclidean
            distances in Ångstroms.
        """
        from MDAnalysis.lib.distances import distance_array

        dists = []
        for atoms in self.dist_atoms:
            dist = distance_array(
                atoms[0].position.reshape(1, 3),
                atoms[1].position.reshape(1, 3)
            ).flatten()[0]
            dists.append(dist)

        return {self.get_default_label_name(): np.array(dists, dtype=float)}
