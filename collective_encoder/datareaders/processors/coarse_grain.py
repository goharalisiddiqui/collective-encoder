from typing import List, Tuple, Any, Dict, Union
import numpy as np
import torch
import ase
import einops
from collective_encoder.datareaders.processors.base import BaseDataProcessor

class CoarseGrainProcessor(BaseDataProcessor):
    """
    Coarse-grains raw frames by averaging positions and labels over a window.
    """
    _IDENTIFIER = "COARSE_GRAIN"
    _REQUIRED_ARGS = [
        'cg_window',
    ]

    def adjust_total_datapoints(self, total: int) -> int:
        return total
    
    def _coarsen_atoms(self, atoms: List[ase.Atoms]) -> List[ase.Atoms]:
        """
        Coarsen a list of ase.Atoms objects by averaging positions over cg_window.
        """
        cg_window = self.cg_window
        cg_traj = []
        for i in range(0, len(atoms), cg_window):
            window_frames = atoms[i:i + cg_window]
            cg_frame = window_frames[0].copy()
            cg_frame.set_positions(
                np.mean([frame.get_positions() for frame in window_frames], axis=0)
            )
            cg_traj.append(cg_frame)
        return cg_traj
    
    def _coarsen_array(self, arr: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Coarsen a 3-D array by averaging over cg_window.
        """
        cg_window = self.cg_window
        
        if arr.ndim >= 3:
            arr = einops.reduce(arr, '... (a b) feature -> ... a feature', 'mean', b=cg_window)
        elif arr.ndim == 2:
            arr = einops.reduce(arr, '... (a b) -> ... a', 'mean', b=cg_window)
        else:
            self.raise_error(f"Unsupported array shape for coarsening: {arr.shape}")
        return arr
    
    def postprocess_seq(self, 
        targets: Union[List[Any], np.ndarray, torch.Tensor], 
        labels: Union[List[Any], np.ndarray, torch.Tensor]
        ) -> Tuple[Union[List[Any], np.ndarray, torch.Tensor], Union[List[Any], np.ndarray, torch.Tensor]]:
        """
        Average positions and labels over cg_window.
        """
        if isinstance(targets, list) and len(targets) > 0:
            if isinstance(targets[0], ase.Atoms):
                targets = self._coarsen_atoms(targets)
            elif isinstance(targets[0], (np.ndarray, torch.Tensor)):
                targets = [self._coarsen_array(t) for t in targets]
        elif isinstance(targets, (np.ndarray, torch.Tensor)):
            targets = self._coarsen_array(targets)
        else:
            self.raise_error(f"Unsupported type for targets: {type(targets)}")

        if isinstance(labels, list) and len(labels) > 0 and isinstance(labels[0], (np.ndarray, torch.Tensor)):
            labels = self._coarsen_array(labels)
        elif isinstance(labels, (np.ndarray, torch.Tensor)):
            labels = self._coarsen_array(labels)
        else:
            self.raise_error(f"Unsupported type for labels: {type(labels)}")

        return targets, labels
