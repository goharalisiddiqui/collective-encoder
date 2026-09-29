import logging
from typing import Dict, List, Union, Optional

import MDAnalysis as mda
import numpy as np
from ase.data import atomic_numbers

from .base import FrameLabeler

_log = logging.getLogger(__name__)


class DebyeStructureFactorLabeler(FrameLabeler):
    """
    Compute the multi-component element-weighted Debye structure factor S(q) or intensity I(q).

    Evaluates:
        S_Debye(q) = 1 + (2 / sum(b_i^2)) * sum_{j < k} b_j * b_k * sin(q * r_{jk}) / (q * r_{jk})

    Periodic boundary conditions (PBC) are automatically applied if box dimensions are
    present in the MDAnalysis universe and ``use_pbc`` is True.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        MDAnalysis Universe loaded with topology and trajectory.
        The trajectory must be positioned at the target frame before calling :meth:`compute`.
    args : dict
        Configuration dict with keys:
        - ``selection`` (str, optional): MDAnalysis atom selection string. Defaults to ``'all'``.
        - ``q_values`` (list of float, optional): Explicit list of scattering wavenumbers in Å⁻¹.
        - ``q_min`` (float, optional): Minimum q value in Å⁻¹ (default: ``0.5``).
        - ``q_max`` (float, optional): Maximum q value in Å⁻¹ (default: ``5.0``).
        - ``q_num`` (int, optional): Number of q values (default: ``10``).
        - ``scattering_lengths`` (dict, optional): Map of element/atom name to scattering length b_i.
        - ``output_mode`` (str, optional): ``'total'`` (normalized S(q)) or ``'intensity'`` (I(q)). Defaults to ``'total'``.
        - ``use_pbc`` (bool, optional): Whether to apply PBC (default: ``True``).
    kwargs : dict
        Additional keyword arguments forwarded to the base class.
    """

    _IDENTIFIER = "DEBYE_STRUCTURE_FACTOR"
    _REQUIRED_ARGS = []
    _OPTIONAL_ARGS = {
        "selection": "all",
        "q_values": None,
        "q_min": 0.5,
        "q_max": 5.0,
        "q_num": 10,
        "scattering_lengths": None,
        "output_mode": "total",
        "use_pbc": True,
    }

    def __init__(
        self,
        universe: mda.Universe,
        args: Dict[str, Union[str, float, int, List[float], Dict[str, float], bool]] = None,
        **kwargs,
    ) -> None:
        super().__init__(args=args, **kwargs)

        self.atoms = universe.select_atoms(self.selection)
        if len(self.atoms) == 0:
            self.raise_error(f"No atoms selected with selection '{self.selection}'")

        if self.q_values is not None and len(self.q_values) > 0:
            self.q_arr = np.array(self.q_values, dtype=np.float64)
        else:
            self.q_arr = np.linspace(
                float(self.q_min),
                float(self.q_max),
                int(self.q_num),
                dtype=np.float64,
            )

        # Precompute atomic scattering lengths/weights b_i for selected atoms
        self.b_weights = np.zeros(len(self.atoms), dtype=np.float64)
        custom_weights = self.scattering_lengths if isinstance(self.scattering_lengths, dict) else {}

        for idx, atom in enumerate(self.atoms):
            try:
                elem = (atom.element or atom.name or "").upper().strip()
            except Exception as e:
                self.log_warn(f"Could not identify element of the atom at index {idx}: {e}")
                elem = ""
            try:
                name = (atom.name or "").upper().strip()
            except Exception as e:
                self.log_warn(f"Could not identify name of the atom at index {idx}: {e}", once=True)
                name = ""

            if name in custom_weights:
                w = float(custom_weights[name])
            elif elem in custom_weights:
                w = float(custom_weights[elem])
            else:
                w = atomic_numbers.get(elem, 1.0)
            self.b_weights[idx] = w

        # Precompute pair product b_j * b_k for all j < k
        n_atoms = len(self.atoms)
        if n_atoms >= 2:
            j_idx, k_idx = np.triu_indices(n_atoms, k=1)
            self.b_pairs = self.b_weights[j_idx] * self.b_weights[k_idx]
            self.sum_b_sq = np.sum(self.b_weights ** 2)
        else:
            self.b_pairs = np.array([])
            self.sum_b_sq = 1.0

        prefix = "debye_sq" if self.output_mode.lower() == "total" else "debye_iq"
        self.label_names = [f"{prefix}_q{q:.2f}" for q in self.q_arr]

    def get_label_names(self) -> List[str]:
        """
        Return the names of the label columns.

        Returns
        -------
        list of str
            Label names for each wavenumber q.
        """
        return self.label_names

    def compute(self) -> List[float]:
        """
        Compute the Debye structure factor or intensity for the current frame.

        Returns
        -------
        list of float
            Values of S(q) or I(q) evaluated at the specified wavenumbers.
        """
        n_atoms = len(self.atoms)
        if n_atoms < 2:
            return [1.0] * len(self.q_arr)

        positions = self.atoms.positions
        box = (
            self.atoms.universe.dimensions
            if (self.use_pbc and hasattr(self.atoms.universe, "dimensions"))
            else None
        )

        dists = mda.lib.distances.self_distance_array(positions, box=box)

        # q_arr shape: (K, 1), dists shape: (1, M) -> QR shape: (K, M)
        QR = self.q_arr[:, None] * dists[None, :]
        sinc_QR = np.where(QR > 1e-7, np.sin(QR) / QR, 1.0)

        # Weighted sum over pairs
        pair_sum = np.sum(self.b_pairs[None, :] * sinc_QR, axis=1)

        if self.output_mode.lower() == "intensity":
            res = (self.sum_b_sq + 2.0 * pair_sum) / float(n_atoms)
        else:
            # Normalized structure factor S(q)
            res = (self.sum_b_sq + 2.0 * pair_sum) / self.sum_b_sq

        return res.tolist()
