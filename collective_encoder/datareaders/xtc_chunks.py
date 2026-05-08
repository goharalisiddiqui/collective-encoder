import os
from typing import Any, Dict

import numpy as np

from .xtc import XTCReader

class XTCChunksReader(XTCReader):
    _IDENTIFIER = "XTC_CHUNKS"
    _REQUIRED_ARGS = XTCReader._REQUIRED_ARGS + [
        "sequence_length",
    ]

    def __init__(self,
                 args: Dict[str, Any] = None,
                 **kwargs,
                 ):
        super().__init__(args=args, **kwargs)

        if self.verbose:
            self.log_info("Atom index mapping (original -> selected):")
            for at in self.mol.atoms:
                self.log_info(f"  Original index: {at.id + 1} -> Selected index: "
                      f"{np.where(self.mol.atoms.indices == at.index)[0][0]} "
                      f"(Element: {at.element}, Name: {at.name}, "
                      f"Type: {at.type}, "
                      f"Residue: {at.residue.resname}{at.residue.resid})")
    
    def get_total_frames(self):
        return len(self.u.trajectory) - self.sequence_length
    
    def _prepare_seq(self, seq):
        """Expand start indices into consecutive frame ranges of length sequence_length."""
        return [j for i in seq for j in range(i, i + self.sequence_length)]
