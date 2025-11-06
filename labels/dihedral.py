import MDAnalysis as mda
from typing import Dict, List, Union

class DihedralValueLabeler:
    def __init__(self, 
                 universe: mda.Universe,
                 args: Dict[str, Union[str, List[float]]]
                 ) -> None:
        self.dihedrals = args.get("label_dihedrals", [])
        if len(self.dihedrals) == 0:
            raise ValueError(f"[{type(self).__name__}] 'label_dihedrals' must be provided in args")
        
        self.label_atoms = []
        self.label_list = []
        for sel in self.dihedrals:
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

    def get_names(self) -> List[str]:
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

