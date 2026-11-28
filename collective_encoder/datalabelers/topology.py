from typing import Any, Dict, List, Optional, Tuple

import ase
import MDAnalysis as mda
import numpy as np

from collective_encoder.datasets.utils.bondgraph import (
    angle_and_torsion_edges,
    assemble_bonds,
)
from .base import BaseLabeler
from .mda import MDALabeler


class TopologyLabeler(MDALabeler):
    """
    Compute internal topological coordinates (bond distances, bond angles,
    cosine of torsions, and sine of torsions) for the current trajectory frame.

    Extracts bond connectivity for the selected atom group from the MDAnalysis
    ``Universe`` and computes:
    - ``"bonds"``: 1-D NumPy array of bond lengths ``(n_bonds,)``
    - ``"angles"``: 1-D NumPy array of bond angles in radians ``(n_angles,)``
    - ``"cos_torsions"``: 1-D NumPy array of cosine of torsion angles ``(n_torsions,)``
    - ``"sin_torsions"``: 1-D NumPy array of sine of torsion angles ``(n_torsions,)``

    Parameters
    ----------
    universe : MDAnalysis.Universe
        MDAnalysis Universe loaded with the molecular topology and trajectory.
    args : dict, optional
        Configuration dictionary. Optional keys:
        - ``selection`` (str, default ``"all"``): MDAnalysis atom selection string.
        - ``name`` (str, optional): Optional labeler name.
    kwargs : dict
        Additional keyword arguments forwarded to :class:`MDALabeler`.
    """

    _IDENTIFIER = "TOPOLOGY"
    _REQUIRED_ARGS: List[str] = []
    _OPTIONAL_ARGS: Dict[str, Any] = {
        **BaseLabeler._OPTIONAL_ARGS,
        "selection": "all",
    }

    def __init__(
        self,
        universe: mda.Universe,
        args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(args=args, **kwargs)
        self.u = universe
        try:
            self.mol = universe.select_atoms(self.selection)
        except Exception as exc:
            self.raise_error(
                f"Invalid selection '{self.selection}' in TopologyLabeler: {exc}"
            )
        if self.mol.n_atoms == 0:
            self.raise_error(
                f"Selection '{self.selection}' matched 0 atoms in TopologyLabeler."
            )

        self.bonds = self._extract_bonds()
        self.atns = [1] * self.mol.n_atoms

    def _extract_bonds(self) -> List[Tuple[int, int]]:
        """
        Extract internal bond index pairs remapped to ``0 .. self.mol.n_atoms - 1``.
        """
        raw_bonds = self.mol.get_connections("bonds", outside=False).indices
        remapped: List[Tuple[int, int]] = []
        mol_indices = self.mol.atoms.indices
        for b0, b1 in raw_bonds:
            i0 = int(np.where(mol_indices == b0)[0][0])
            i1 = int(np.where(mol_indices == b1)[0][0])
            remapped.append((i0, i1))
        return remapped

    def get_label_names(self) -> List[str]:
        """
        Return the dictionary keys produced by :meth:`compute`.

        Returns
        -------
        list of str
            ``["bonds", "angles", "cos_torsions", "sin_torsions"]``.
        """
        return ["bonds", "angles", "cos_torsions", "sin_torsions"]

    def compute(self, indices: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
        """
        Compute bond lengths, bond angles, and torsion cosines/sines for the current frame.

        Parameters
        ----------
        indices : list of int, optional
            Unused frame indices argument for :class:`BaseLabeler` interface compatibility.

        Returns
        -------
        dict of str to numpy.ndarray
            Dictionary containing ``"bonds"``, ``"angles"``, ``"cos_torsions"``,
            and ``"sin_torsions"`` 1-D NumPy arrays.
        """
        structure = ase.Atoms(
            numbers=self.atns,
            positions=self.mol.atoms.positions.copy(),
        )
        assembled_bonds = assemble_bonds(structure, self.bonds)
        _, edge_attr, _, _, _ = angle_and_torsion_edges(structure, assembled_bonds)

        bonds_arr = np.array([b[2] for b in assembled_bonds], dtype=np.float64)
        if edge_attr.size(0) > 0:
            ea_np = edge_attr.detach().cpu().numpy()
            angle_mask = ea_np[:, 0] == 1.0
            torsion_mask = ea_np[:, 1] == 1.0
            angles_arr = ea_np[angle_mask, 2].astype(np.float64)
            torsion_vals = ea_np[torsion_mask, 2].astype(np.float64)
            cos_torsions_arr = np.cos(torsion_vals).astype(np.float64)
            sin_torsions_arr = np.sin(torsion_vals).astype(np.float64)
        else:
            angles_arr = np.empty((0,), dtype=np.float64)
            cos_torsions_arr = np.empty((0,), dtype=np.float64)
            sin_torsions_arr = np.empty((0,), dtype=np.float64)

        return {
            "bonds": bonds_arr,
            "angles": angles_arr,
            "cos_torsions": cos_torsions_arr,
            "sin_torsions": sin_torsions_arr,
        }
