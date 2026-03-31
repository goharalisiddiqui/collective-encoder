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

        if kwargs.get('verbose', True):
            print("Atom index mapping (original -> selected):")
            for at in self.mol.atoms:
                print(f"  Original index: {at.id + 1} -> Selected index: "
                      f"{np.where(self.mol.atoms.indices == at.index)[0][0]} "
                      f"(Element: {at.element}, Name: {at.name}, "
                      f"Type: {at.type}, "
                      f"Residue: {at.residue.resname}{at.residue.resid})")
    
    def get_total_frames(self):
        return len(self.u.trajectory) - self.sequence_length
    
    def _read_and_label(self, indices, labeler):
        chunked_indices = []
        for i in indices:
            s = i
            e = s + self.sequence_length
            chunked_indices.extend([j for j in range(s, e)])
        indices = chunked_indices
        return super()._read_and_label(indices, labeler)
    
    