from typing import List, Tuple, Any
import ase
from collective_encoder.datareaders.processors.base import BaseProcessor

class FilterProcessor(BaseProcessor):
    """
    Filters frames based on label values.
    """
    def __init__(self, label_index: int = 0, min_val: float = None, max_val: float = None, **kwargs):
        super().__init__(**kwargs)
        self.label_index = label_index
        self.min_val = min_val
        self.max_val = max_val

    def postprocess_seq(self, mol_traj: List[ase.Atoms], labels: List[Any]) -> Tuple[List[ase.Atoms], List[Any]]:
        idx_to_remove = []
        for i in range(len(labels)):
            val = labels[i][self.label_index]
            remove = False
            if self.min_val is not None and val <= self.min_val:
                remove = True
            if self.max_val is not None and val >= self.max_val:
                remove = True

            if remove:
                idx_to_remove.append(i)

        traj = [frame for i, frame in enumerate(mol_traj) if i not in idx_to_remove]
        filtered_labels = [label for i, label in enumerate(labels) if i not in idx_to_remove]

        return traj, filtered_labels
