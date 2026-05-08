import os
from typing import Any, Dict

from tqdm import tqdm

import numpy as np

from .xtc_chunks import XTCChunksReader

class XTCChunksCGReader(XTCChunksReader):
    _IDENTIFIER = "XTC_CHUNKS_CG"
    _REQUIRED_ARGS = XTCChunksReader._REQUIRED_ARGS + [
        "cg_window"
    ]

    def __init__(self,
                 args: Dict[str, Any] = None,
                 **kwargs,
                 ):
        super().__init__(args=args, **kwargs)
    
    def get_total_frames(self):
        return len(self.u.trajectory) - self.sequence_length * self.cg_window
    
    def _prepare_seq(self, seq):
        """Expand start indices using sequence_length * cg_window frames per start."""
        expanded_length = self.sequence_length * self.cg_window
        return [j for i in seq for j in range(i, i + expanded_length)]

    def _postprocess_seq(self, mol_traj, labels):
        """Coarse-grain raw frames by averaging positions over cg_window windows."""
        cg_window = self.cg_window
        cg_traj = []
        for i in range(0, len(mol_traj), cg_window):
            window_frames = mol_traj[i:i + cg_window]
            cg_frame = window_frames[0]
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