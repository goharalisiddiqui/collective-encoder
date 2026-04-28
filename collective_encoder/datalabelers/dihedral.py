import MDAnalysis as mda
from typing import Dict, List, Union

from .base import FrameLabeler


class DihedralValueLabeler(FrameLabeler):
    """Compute backbone dihedral angles (phi/psi) for the current trajectory frame.

    Expects an MDAnalysis ``Universe`` already positioned at the desired frame.
    For each entry in ``label_dihedrals``, the dihedral is computed from the
    four atoms selected by MDAnalysis's built-in ``phi_selection()`` /
    ``psi_selection()`` helpers.

    Args:
        universe: MDAnalysis Universe loaded with the full topology and
            trajectory.  The trajectory must be positioned at the target
            frame before calling :meth:`compute`.
        args: Configuration dict with key ``label_dihedrals`` (required) — a
            list of strings of the form ``'phi_<resnum>'`` or ``'psi_<resnum>'``
            (e.g. ``['phi_2', 'psi_2']``).
    """

    _IDENTIFIER = "DIHEDRAL"
    _REQUIRED_ARGS = ["label_dihedrals"]
    _OPTIONAL_ARGS = {}

    def __init__(self,
                 universe: mda.Universe,
                 args: Dict[str, Union[str, List[float]]],
                 **kwargs   
                 ) -> None:
        super().__init__(args=args, **kwargs)

        if len(self.label_dihedrals) == 0:
            self.raise_error(f"'label_dihedrals' must be provided in args")
        self.label_atoms = []
        self.label_list = []
        for sel in self.label_dihedrals:
            sel = sel.strip()
            assert len(sel) > 4, f"Label dihedral {sel} must be at least 5 characters long"
            if sel[:4] not in ['phi_', 'psi_']:
                raise ValueError(f"Label dihedral {sel} must start with 'phi_' or 'psi_'")
            resnum = int(sel.split('_')[1])
            if sel[:4] == 'phi_':
                sel = universe.residues[resnum-1].phi_selection()
                assert sel is not None, f"Residue {resnum} does not have a phi dihedral"
                self.label_atoms.append(sel)
                self.label_list.append(f"phi_{resnum}")
            elif  sel[:4] == 'psi_':
                sel = universe.residues[resnum-1].psi_selection()
                assert sel is not None, f"Residue {resnum} does not have a psi dihedral"
                self.label_atoms.append(sel)
                self.label_list.append(f"psi_{resnum}")

    def get_label_names(self) -> List[str]:
        return self.label_list

    def compute(self) -> List[float]:

        dihedrals = []
        for atoms in self.label_atoms:
            dihedral = mda.lib.distances.calc_dihedrals(
                atoms.positions[0],
                atoms.positions[1],
                atoms.positions[2],
                atoms.positions[3],
            )
            dihedrals.append(dihedral)

        return dihedrals

