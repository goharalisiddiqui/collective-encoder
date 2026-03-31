import MDAnalysis as mda
from typing import Dict, List, Union

from .base import BaseLabeler

class CoordinationCountLabeler(BaseLabeler):
    def __init__(self, 
                 universe: mda.Universe,
                 args: Dict[str, Union[str, float, List[int]]],
    ) -> None:
        self.cutoff_distance = args.get("cutoff_distance", 5.0)
        selection_centers = args.get("selection_centers", None)
        if selection_centers is None:
            raise ValueError(f"[{type(self).__name__}] 'selection_centers' must be provided in args")
        selection_neighbors = args.get("selection_neighbors", None)
        if selection_neighbors is None:
            raise ValueError(f"[{type(self).__name__}] 'selection_neighbors' must be provided in args")
        self.centers = universe.select_atoms(selection_centers)
        self.neighbors = universe.select_atoms(selection_neighbors) 
        self.bins = args.get("bins", [6])

        if len(self.centers) == 0:
            raise ValueError(f"[{type(self).__name__}] No atoms selected for centers with selection: {selection_centers}")
        if len(self.neighbors) == 0:
            raise ValueError(f"[{type(self).__name__}] No atoms selected for neighbors with selection: {selection_neighbors}")

    def get_names(self) -> List[str]:
        return [f'coordination_count_{b}' for b in self.bins]

    def compute(self) -> List[float]:
        centers = self.centers
        neighbors = self.neighbors


        counts = [0] * len(self.bins)
        for center in centers:
            dists = mda.lib.distances.distance_array(
                center.position.reshape(1, 3),
                neighbors.positions
            ).flatten()
            count = (dists <= self.cutoff_distance).sum()

            for i, b in enumerate(self.bins):
                if count == b:
                    counts[i] += 1

        return counts

