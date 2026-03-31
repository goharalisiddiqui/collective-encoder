import os

from tqdm import tqdm

import numpy as np

from .xtc_chunks import XTCChunksReader

class XTCChunksCGReader(XTCChunksReader):
    def __init__(self,
                 cg_window : int,
                 **kwargs,
                 ):
        self.cg_window = cg_window
        super().__init__(**kwargs)
    
    def get_total_frames(self):
        return len(self.u.trajectory) - self.sequence_length * self.cg_window
    
    def _read_and_label(self, indices, labeler):
        temp = self.sequence_length # Store original sequence length
        self.sequence_length = self.sequence_length * self.cg_window
        mol_traj, labels = super()._read_and_label(indices, labeler)
        self.sequence_length = temp

        cg_window = self.cg_window
        print(f"Coarse-graining trajectory with window size {cg_window}...") if self.verbose else None
        cg_traj = []
        # Coarse-grain the trajectory
        for i in tqdm(range(0, len(mol_traj), cg_window), desc="Coarse-graining frames"):
            window_frames = mol_traj[i:i+cg_window]
            cg_frame = window_frames[0]
            cg_frame.set_positions(
                np.mean([frame.get_positions() for frame in window_frames], axis=0)
            )
            cg_traj.append(cg_frame)
        mol_traj = cg_traj
        # Coarse-grain labels if they exist
        if labels is not None:
            cg_labels = []
            for i in range(0, len(labels), cg_window):
                window_labels = labels[i:i+cg_window]
                cg_label = np.mean(window_labels, axis=0)
                cg_labels.append(cg_label)
            labels = cg_labels
        print(f"Coarse-grained trajectory has {len(mol_traj)} frames.") if self.verbose else None
        return mol_traj, labels