from typing import List, Tuple, Any
import numpy as np
import ase
from collective_encoder.datareaders.processors.base import BaseProcessor

class CoarseGrainProcessor(BaseProcessor):
    """
    Coarse-grains raw frames by averaging positions and labels over a window.
    """
    def __init__(self, cg_window: int, sequence_length: int = None, **kwargs):
        super().__init__(**kwargs)
        self.cg_window = cg_window
        self.sequence_length = sequence_length

    def prepare_seq(self, seq: List[int]) -> List[int]:
        """
        Expand start indices into frame ranges of length (sequence_length * cg_window).
        """
        if self.sequence_length is None:
            # If not explicitly given, assume seq is already expanded and we just don't expand further here
            # But normally, chunking and coarse graining were combined.
            return seq

        expanded_length = self.sequence_length * self.cg_window
        return [j for i in seq for j in range(i, i + expanded_length)]

    def adjust_total_frames(self, total_frames: int) -> int:
        """
        Adjust the total frames because a full coarse-grained sequence requires multiple frames.
        """
        if self.sequence_length is not None:
            return total_frames - (self.sequence_length * self.cg_window)
        return total_frames

    def postprocess_seq(self, mol_traj: List[ase.Atoms], labels: List[Any]) -> Tuple[List[ase.Atoms], List[Any]]:
        """
        Average positions and labels over cg_window.
        """
        cg_window = self.cg_window
        cg_traj = []
        for i in range(0, len(mol_traj), cg_window):
            window_frames = mol_traj[i:i + cg_window]
            cg_frame = window_frames[0].copy()
            cg_frame.set_positions(
                np.mean([frame.get_positions() for frame in window_frames], axis=0)
            )
            cg_traj.append(cg_frame)

        if labels:
            cg_labels = [np.mean(labels[i:i + cg_window], axis=0)
                         for i in range(0, len(labels), cg_window)]
        else:
            cg_labels = labels

        return cg_traj, cg_labels
