import os

import numpy as np

from .xtc import XTCReader

class XTCChunksReader(XTCReader):
    def __init__(self,
                 sequence_length : int,
                 **kwargs,
                 ):
        self.sequence_length = sequence_length
        super().__init__(**kwargs)

        if self.verbose:
            print("Atom index mapping (original -> selected):")
            for at in self.mol.atoms:
                print(f"  Original index: {at.id + 1} -> Selected index: "
                      f"{np.where(self.mol.atoms.indices == at.index)[0][0]} "
                      f"(Element: {at.element}, Name: {at.name}, "
                      f"Type: {at.type}, "
                      f"Residue: {at.residue.resname}{at.residue.resid})")
    
    def get_total_frames(self):
        return len(self.u.trajectory) - self.sequence_length
    
    def _prepare_seq(self, seq):
        """Expand start indices into consecutive frame ranges of length sequence_length."""
        return [j for i in seq for j in range(i, i + self.sequence_length)]
