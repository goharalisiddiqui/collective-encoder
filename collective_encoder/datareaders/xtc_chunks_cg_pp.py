import os

from tqdm import tqdm

import numpy as np

from .xtc_chunks_cg import XTCChunksCGReader

class XTCChunksCGReaderPP(XTCChunksCGReader):
    _IDENTIFIER = "XTC_CHUNKS_CG_PP"

    def _postprocess_seq(self, mol_traj, labels):
        """Coarse-grain raw frames by averaging positions over cg_window windows."""
        traj, labels = super()._postprocess_seq(mol_traj, labels)
        idx_to_remove = []
        for i in range(len(labels) - 1):
            if labels[i][0] > 0 and labels[i][0] < 2.5:
                idx_to_remove.append(i)
        traj = [frame for i, frame in enumerate(traj) if i not in idx_to_remove]
        labels = [label for i, label in enumerate(labels) if i not in idx_to_remove]
        
        return traj, labels