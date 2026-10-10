from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict
import ase

class BaseProcessor(ABC):
    """
    Abstract base class for sequence processors.

    Processors modify the behavior of trajectory reading either by
    expanding/altering the indices before reading (via prepare_seq)
    or transforming the trajectory frames and labels after reading
    (via postprocess_seq).
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def prepare_seq(self, seq: List[int]) -> List[int]:
        """
        Transform a sequence of indices before passing to the parallel worker.

        Parameters
        ----------
        seq : list of int
            Original indices.

        Returns
        -------
        list of int
            Transformed indices.
        """
        return seq

    def postprocess_seq(self, mol_traj: List[ase.Atoms], labels: List[Any]) -> Tuple[List[ase.Atoms], List[Any]]:
        """
        Post-process a single sequence's (mol_traj, labels) after reading.

        Parameters
        ----------
        mol_traj : list of ase.Atoms
            The read trajectory frames.
        labels : list of list of float
            The computed labels.

        Returns
        -------
        tuple
            Processed `(mol_traj, labels)`.
        """
        return mol_traj, labels

    def adjust_total_frames(self, total_frames: int) -> int:
        """
        Adjust the total number of frames based on the processor's requirements.
        By default, it doesn't change anything.
        """
        return total_frames
