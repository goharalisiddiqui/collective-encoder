import os
from typing import Any, Dict

from tqdm import tqdm

import numpy as np

from .xtc_chunks import XTCChunksReader

class XTCChunksCGReader(XTCChunksReader):
    """
    Reader for splitting a trajectory into chunks and applying coarse-graining.

    Extends :class:`XTCChunksReader` by reading an expanded range of frames
    (``sequence_length * cg_window``) and averaging positions/labels over
    the `cg_window` to yield `sequence_length` frames per chunk.

    Parameters
    ----------
    args : dict, optional
        Dictionary of configuration options. Must include:
        - ``cg_window`` (int): The number of consecutive frames to average over.
        - ``sequence_length`` (int): The final sequence length after averaging.
    kwargs : dict
        Additional arguments forwarded to `XTCChunksReader`.
    """
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
        """
        Get the total number of starting frames that can yield a full coarse-grained sequence.

        Returns
        -------
        int
            Number of frames minus the expanded sequence length.
        """
        return len(self.u.trajectory) - self.sequence_length * self.cg_window
    
    def _prepare_seq(self, seq):
        """
        Expand start indices using sequence_length * cg_window frames per start.

        Parameters
        ----------
        seq : list of int
            List of starting indices.

        Returns
        -------
        list of int
            Flattened list of all raw frame indices needed prior to averaging.
        """
        self.log_info(f"Expanding sequence start indices into frame ranges of length "
                      f"{self.sequence_length * self.cg_window} (sequence_length "
                      f"{self.sequence_length} * cg_window {self.cg_window}).")
        expanded_length = self.sequence_length * self.cg_window
        return [j for i in seq for j in range(i, i + expanded_length)]

    def _postprocess_seq(self, mol_traj, labels):
        """
        Coarse-grain raw frames by averaging positions over cg_window windows.

        Parameters
        ----------
        mol_traj : list of ase.Atoms
            Raw frames read from the trajectory.
        labels : list of list of float
            Labels computed per raw frame.

        Returns
        -------
        tuple
            Tuple containing averaged `(mol_traj, labels)`.
        """
        cg_window = self.cg_window
        cg_traj = []
        self.log_info(f"Coarse-graining trajectory with window size {cg_window}.")
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