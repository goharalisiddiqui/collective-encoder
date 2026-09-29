import os

from tqdm import tqdm

import numpy as np

from .xtc_chunks_cg import XTCChunksCGReader

class XTCChunksCGReaderPP(XTCChunksCGReader):
    """
    Reader for splitting, coarse-graining, and filtering a trajectory.

    Extends :class:`XTCChunksCGReader` by adding an additional filtering
    step in `_postprocess_seq` to remove certain frames based on label values.
    """
    _IDENTIFIER = "XTC_CHUNKS_CG_PP"

    def _postprocess_seq(self, mol_traj, labels):
        """
        Coarse-grain raw frames and subsequently filter them.

        Parameters
        ----------
        mol_traj : list of ase.Atoms
            Raw frames read from the trajectory.
        labels : list of list of float
            Labels computed per raw frame.

        Returns
        -------
        tuple
            Tuple containing averaged and filtered `(mol_traj, labels)`.
        """
        traj, labels = super()._postprocess_seq(mol_traj, labels)
        idx_to_remove = []
        for i in range(len(labels) - 1):
            if labels[i][0] > 0 and labels[i][0] < 2.5:
                idx_to_remove.append(i)
        traj = [frame for i, frame in enumerate(traj) if i not in idx_to_remove]
        labels = [label for i, label in enumerate(labels) if i not in idx_to_remove]
        if len(idx_to_remove) > 0:
            self.log_debug(f"Removed {len(idx_to_remove)} frames from the trajectory.")
        
        return traj, labels