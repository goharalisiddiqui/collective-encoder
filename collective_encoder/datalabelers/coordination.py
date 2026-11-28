import numpy as np
import MDAnalysis as mda
from typing import Optional, Dict, List, Union

from .base import BaseLabeler
from .mda import MDALabeler


class CoordinationCountLabeler(MDALabeler):
    """
    Count atoms within a cutoff distance for the current trajectory frame.

    For each center atom, counts how many neighbour atoms lie within
    ``cutoff_distance`` Å and compares the count against each threshold in
    ``bins``.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        MDAnalysis Universe positioned at the target frame.
    args : dict
        Configuration dict with the following keys:
        - ``selection_centers`` (str, required): MDAnalysis selection string
          for center atoms.
        - ``selection_neighbors`` (str, required): MDAnalysis selection string
          for neighbour atoms.
        - ``cutoff_distance`` (float, optional): Distance cutoff in Å.
          Defaults to ``5.0``.
        - ``bins`` (list of int, optional): Coordination number thresholds to
          count against. Defaults to ``[6]``.
        - ``name`` (str, optional): Custom label dictionary key.
    kwargs : dict
        Additional keyword arguments forwarded to the base class.
    """

    _IDENTIFIER = "COORDINATION"
    _REQUIRED_ARGS = ["selection_centers", "selection_neighbors"]
    _OPTIONAL_ARGS = BaseLabeler._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        "cutoff_distance": 5.0,
        "bins": [6],
    })

    def __init__(self,
                 universe: mda.Universe,
                 args: Dict[str, Union[str, float, List[int]]],
                 **kwargs
                 ) -> None:
        super().__init__(args=args, **kwargs)
        if self.selection_centers is None:
            self.raise_error(f"'selection_centers' must be provided in args")
        if self.selection_neighbors is None:
            self.raise_error(f"'selection_neighbors' must be provided in args")
        self.centers = universe.select_atoms(self.selection_centers)
        self.neighbors = universe.select_atoms(self.selection_neighbors)
        self.box = universe.dimensions

        if len(self.centers) == 0:
            self.raise_error(f"No atoms selected for centers \
                             with selection: {self.selection_centers}")
        if len(self.neighbors) == 0:
            self.raise_error(f"No atoms selected for neighbors \
                             with selection: {self.selection_neighbors}")

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
        Compute the coordination count labels for the current frame.

        Parameters
        ----------
        indices : list of int, optional
            Unused frame indices argument for interface compatibility.

        Returns
        -------
        dict of str to numpy.ndarray
            Dictionary mapping the label name to a 1-D NumPy array of counts of
            central atoms meeting each coordination threshold.
        """
        centers = self.centers
        neighbors = self.neighbors

        counts = [0] * len(self.bins)
        for center in centers:
            dists = mda.lib.distances.distance_array(
                center.position.reshape(1, 3),
                neighbors.positions,
                box=self.box
            ).flatten()
            count = (dists <= self.cutoff_distance).sum()

            for i, b in enumerate(self.bins):
                if count >= b:
                    counts[i] += 1
        
        return {self.get_default_label_name(): np.array(counts, dtype=float)}

