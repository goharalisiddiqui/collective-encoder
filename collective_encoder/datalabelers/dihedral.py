from itertools import permutations
from typing import Any, Dict, List, Optional, Tuple, Union

import MDAnalysis as mda
import numpy as np

from .mda import MDALabeler


class DihedralValueLabeler(MDALabeler):
    """
    Compute dihedral (torsion) angles for the current trajectory frame from
    user-defined MDAnalysis atom selection strings.

    Each dihedral in ``label_dihedrals`` is specified as a mapping
    ``{label_name: selection_string}`` (either as a list of single-entry/multi-entry
    dictionaries or as a flat dictionary). During initialization, the labeler
    validates that:
    1. Each selection resolves to **exactly 4 unique atoms** in ``universe``.
    2. The 4 selected atoms are connected by **3 sequential chemical bonds** forming
       a valid 4-atom dihedral chain ``a0 - a1 - a2 - a3`` (and automatically orders
       the 4 atoms along the connected bond path if ``select_atoms`` returned them
       in topology-index order).
    3. The 4 atoms have non-degenerate 3-D coordinates producing a finite dihedral angle.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        MDAnalysis Universe loaded with the molecular topology and trajectory.
    args : dict
        Configuration dictionary containing ``label_dihedrals`` (required), e.g.::

            label_dihedrals:
              - phi: "resid 2 and (name C or name N or name CA or name O)"
              - psi: "(resid 2 and (name N or name CA or name C)) or (resid 3 and name N)"

    kwargs : dict
        Additional keyword arguments forwarded to :class:`MDALabeler`.
    """

    _IDENTIFIER = "DIHEDRAL"
    _REQUIRED_ARGS = ["label_dihedrals"]
    _OPTIONAL_ARGS = {}

    def __init__(
        self,
        universe: mda.Universe,
        args: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(args=args, **kwargs)

        if not self.label_dihedrals:
            self.raise_error("'label_dihedrals' must be a non-empty list or dict in args.")

        parsed_specs = self._parse_dihedral_specs(self.label_dihedrals)
        if len(parsed_specs) == 0:
            self.raise_error("'label_dihedrals' must contain at least one dihedral specification.")

        self.label_atoms: List[mda.AtomGroup] = []
        self.label_list: List[str] = []

        for label_name, selection in parsed_specs:
            atoms = self._select_and_validate_four_atoms(universe, label_name, selection)
            ordered_atoms = self._validate_and_order_dihedral_chain(
                universe, atoms, label_name, selection
            )
            self._validate_dihedral_geometry(ordered_atoms, label_name, selection)
            self.label_atoms.append(ordered_atoms)
            self.label_list.append(label_name)

    def _parse_dihedral_specs(
        self, raw_specs: Union[List[Any], Dict[str, Any]]
    ) -> List[Tuple[str, Union[str, List[str]]]]:
        """
        Normalize ``label_dihedrals`` into an ordered list of ``(label_name, selection)`` pairs.

        Parameters
        ----------
        raw_specs : list of dict or dict
            Raw ``label_dihedrals`` configuration from YAML / args.

        Returns
        -------
        list of tuple of (str, str or list of str)
            Ordered ``(label_name, selection)`` pairs.
        """
        specs: List[Tuple[str, Union[str, List[str]]]] = []
        if isinstance(raw_specs, dict):
            for name, sel in raw_specs.items():
                specs.append((str(name).strip(), sel))
            return specs

        if isinstance(raw_specs, (list, tuple)):
            for item in raw_specs:
                if not isinstance(item, dict) or len(item) == 0:
                    self.raise_error(
                        f"Invalid entry {item!r} in 'label_dihedrals'. Expected a dictionary "
                        "mapping dihedral label name to an MDAnalysis selection string, e.g. "
                        "{'phi': 'resid 2 and (name C or name N or name CA or name O)'}."
                    )
                for name, sel in item.items():
                    specs.append((str(name).strip(), sel))
            return specs

        self.raise_error(
            f"'label_dihedrals' must be a list of dicts or a dict, got {type(raw_specs).__name__}."
        )

    @staticmethod
    def _format_atoms(atoms: mda.AtomGroup) -> List[str]:
        """Format an AtomGroup into human-readable atom descriptors for error messages."""
        out: List[str] = []
        for a in atoms:
            resname = getattr(a, "resname", "RES")
            resid = getattr(a, "resid", "?")
            name = getattr(a, "name", "ATOM")
            out.append(f"{resname}{resid}:{name} (index {int(a.index)})")
        return out

    def _select_and_validate_four_atoms(
        self,
        universe: mda.Universe,
        label_name: str,
        selection: Union[str, List[str], Tuple[str, ...]],
    ) -> mda.AtomGroup:
        """
        Execute the MDAnalysis selection and verify that it selects exactly 4 unique atoms.

        Parameters
        ----------
        universe : MDAnalysis.Universe
            Active MDAnalysis Universe.
        label_name : str
            Name of the dihedral label.
        selection : str or list of str
            MDAnalysis selection string (or list of selection strings).

        Returns
        -------
        MDAnalysis.AtomGroup
            Selected 4-atom group.
        """
        if isinstance(selection, str):
            sel_str = selection.strip()
            if not sel_str:
                self.raise_error(
                    f"Dihedral '{label_name}' has an empty selection string."
                )
            try:
                atoms = universe.select_atoms(sel_str)
            except Exception as exc:
                self.raise_error(
                    f"Invalid MDAnalysis selection string for dihedral '{label_name}': "
                    f"'{selection}' ({exc})."
                )
        elif isinstance(selection, (list, tuple)) and all(
            isinstance(s, str) and s.strip() for s in selection
        ):
            try:
                atoms = universe.select_atoms(*[s.strip() for s in selection])
            except Exception as exc:
                self.raise_error(
                    f"Invalid MDAnalysis selection list for dihedral '{label_name}': "
                    f"{selection!r} ({exc})."
                )
        else:
            self.raise_error(
                f"Selection for dihedral '{label_name}' must be a non-empty MDAnalysis "
                f"selection string (or list of strings), got {selection!r}."
            )

        if len(atoms) != 4:
            self.raise_error(
                f"Dihedral '{label_name}' selection '{selection}' must select exactly 4 atoms, "
                f"but selected {len(atoms)} atom(s): {self._format_atoms(atoms)}."
            )

        unique_indices = set(int(a.index) for a in atoms)
        if len(unique_indices) != 4:
            self.raise_error(
                f"Dihedral '{label_name}' selection '{selection}' selected duplicate atoms: "
                f"{self._format_atoms(atoms)}. All 4 atoms in a dihedral must be distinct."
            )

        return atoms

    def _get_induced_bonds(
        self, universe: mda.Universe, atoms: mda.AtomGroup
    ) -> set:
        """
        Return the set of undirected bond pairs ``{(i, j), (j, i)}`` among the 4 selected atoms.

        Uses topology bonds from ``universe.atoms.bonds`` when present, falling back
        to MDAnalysis covalent bond guessing when the universe lacks bond topology.
        """
        idx_set = {int(a.index) for a in atoms}
        adj: set = set()

        has_topol_bonds = (
            hasattr(universe.atoms, "bonds") and len(universe.atoms.bonds) > 0
        )
        if has_topol_bonds:
            for bond in atoms.bonds:
                i0, i1 = int(bond.atoms[0].index), int(bond.atoms[1].index)
                if i0 in idx_set and i1 in idx_set and i0 != i1:
                    adj.add((i0, i1))
                    adj.add((i1, i0))
        else:
            from MDAnalysis.topology.guessers import guess_bonds

            guessed = guess_bonds(atoms, atoms.positions)
            for i0, i1 in guessed:
                i0_int, i1_int = int(i0), int(i1)
                if i0_int in idx_set and i1_int in idx_set and i0_int != i1_int:
                    adj.add((i0_int, i1_int))
                    adj.add((i1_int, i0_int))

        return adj

    def _validate_and_order_dihedral_chain(
        self,
        universe: mda.Universe,
        atoms: mda.AtomGroup,
        label_name: str,
        selection: Any,
    ) -> mda.AtomGroup:
        """
        Verify that the 4 selected atoms form a connected 3-bond dihedral path
        ``a0 - a1 - a2 - a3`` and return them ordered along that bond chain.

        Parameters
        ----------
        universe : MDAnalysis.Universe
            Active MDAnalysis Universe.
        atoms : MDAnalysis.AtomGroup
            4 unique selected atoms.
        label_name : str
            Name of the dihedral label.
        selection : Any
            Original selection specification (for error reporting).

        Returns
        -------
        MDAnalysis.AtomGroup
            4-atom group ordered sequentially along the dihedral bond chain.
        """
        adj = self._get_induced_bonds(universe, atoms)
        orig_indices = [int(a.index) for a in atoms]

        # 1. If the current atom order already forms a connected chain a0-a1-a2-a3, preserve it
        if (
            (orig_indices[0], orig_indices[1]) in adj
            and (orig_indices[1], orig_indices[2]) in adj
            and (orig_indices[2], orig_indices[3]) in adj
        ):
            return atoms

        # 2. Otherwise search for a valid 3-bond simple path (Hamiltonian path) across the 4 atoms
        valid_paths: List[Tuple[int, int, int, int]] = []
        for p in permutations(orig_indices, 4):
            if (p[0], p[1]) in adj and (p[1], p[2]) in adj and (p[2], p[3]) in adj:
                # Canonicalize path vs its reversal (p0-p1-p2-p3 and p3-p2-p1-p0 have identical dihedral value)
                if p[0] < p[3]:
                    valid_paths.append(p)

        if not valid_paths:
            idx_to_atom = {int(a.index): a for a in atoms}
            unique_bonds = sorted(
                {(min(i, j), max(i, j)) for (i, j) in adj}
            )
            bond_strs = [
                f"{idx_to_atom[i].resname}{idx_to_atom[i].resid}:{idx_to_atom[i].name}"
                f"--{idx_to_atom[j].resname}{idx_to_atom[j].resid}:{idx_to_atom[j].name}"
                for (i, j) in unique_bonds
            ]
            self.raise_error(
                f"Selected atoms for dihedral '{label_name}' (selection: '{selection}') -> "
                f"{self._format_atoms(atoms)} do not form a valid 4-atom dihedral chain "
                f"(3 sequential bonds a0-a1-a2-a3). "
                f"Bonds found among the selected atoms: {bond_strs or 'none'}."
            )

        best_path = valid_paths[0]
        return universe.atoms[list(best_path)]

    def _validate_dihedral_geometry(
        self,
        atoms: mda.AtomGroup,
        label_name: str,
        selection: Any,
    ) -> None:
        """
        Verify that the 4 atoms have non-degenerate 3-D positions on the initial frame.
        """
        pos = atoms.positions
        v1 = pos[1] - pos[0]
        v2 = pos[2] - pos[1]
        v3 = pos[3] - pos[2]

        n1 = np.cross(v1, v2)
        n2 = np.cross(v2, v3)
        if np.linalg.norm(n1) < 1e-8 or np.linalg.norm(n2) < 1e-8:
            self.raise_error(
                f"Selected atoms for dihedral '{label_name}' (selection: '{selection}') -> "
                f"{self._format_atoms(atoms)} have collinear or coincident positions "
                "and cannot define a valid dihedral plane."
            )

        val = mda.lib.distances.calc_dihedrals(pos[0], pos[1], pos[2], pos[3])
        if not np.isfinite(val):
            self.raise_error(
                f"Computed dihedral angle for '{label_name}' (selection: '{selection}') is non-finite ({val})."
            )

    def get_label_names(self) -> List[str]:
        """
        Return the names of the configured dihedral labels.

        Returns
        -------
        list of str
            Dihedral label names in configuration order.
        """
        return list(self.label_list)

    def compute(self, indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Compute the dihedral angles (in radians) for the current trajectory frame.

        Parameters
        ----------
        indices : list of int, optional
            Unused frame indices argument for :class:`BaseLabeler` interface compatibility.

        Returns
        -------
        numpy.ndarray
            1-D array of computed dihedral angles in radians of shape ``(n_dihedrals,)``.
        """
        dihedrals = []
        for atoms in self.label_atoms:
            dihedral = mda.lib.distances.calc_dihedrals(
                atoms.positions[0],
                atoms.positions[1],
                atoms.positions[2],
                atoms.positions[3],
            )
            dihedrals.append(float(dihedral))

        return np.array(dihedrals, dtype=float)
