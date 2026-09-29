import logging
from typing import Dict, List, Union, Optional

import MDAnalysis as mda
import numpy as np
import scipy.special as sp

from .base import FrameLabeler

_log = logging.getLogger(__name__)


def _sph_harm(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Computes complex spherical harmonic Y_l^m(theta, phi) safely across SciPy versions."""
    if hasattr(sp, "sph_harm_y"):
        return sp.sph_harm_y(l, m, theta, phi)
    elif hasattr(sp, "sph_harm"):
        return sp.sph_harm(m, l, phi, theta)
    else:
        raise RuntimeError("scipy.special does not have sph_harm_y or sph_harm.")


class SteinhardtOrderParameterLabeler(FrameLabeler):
    """
    Compute Steinhardt bond-orientational order parameters q_l and Q_l.

    Calculates local per-atom rotational invariants q_l(i), mean local order parameter
    q_l_mean_local = (1/N) * sum_i q_l(i), and global bond order parameter Q_l_global
    for specified spherical harmonic degrees l (e.g., l=4, 6).

    Periodic boundary conditions (PBC) are automatically applied if box dimensions are
    present in the MDAnalysis universe and ``use_pbc`` is True.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        MDAnalysis Universe loaded with topology and trajectory.
    args : dict
        Configuration dict with keys:
        - ``selection_centers`` (str, optional): Atom selection for center atoms. Defaults to ``'all'``.
        - ``selection_neighbors`` (str, optional): Atom selection for neighbor atoms. Defaults to ``'all'``.
        - ``cutoff_distance`` (float, optional): Neighbor cutoff distance in Å. Defaults to ``3.5``.
        - ``degrees`` (list of int, optional): Spherical harmonic degrees l (e.g., ``[4, 6]``). Defaults to ``[4, 6]``.
        - ``average_type`` (str, optional): ``'mean_local'``, ``'global'``, or ``'both'``. Defaults to ``'both'``.
        - ``use_pbc`` (bool, optional): Whether to apply PBC (default: ``True``).
    kwargs : dict
        Additional keyword arguments forwarded to the base class.
    """

    _IDENTIFIER = "STEINHARDT"
    _REQUIRED_ARGS = []
    _OPTIONAL_ARGS = {
        "selection_centers": "all",
        "selection_neighbors": "all",
        "cutoff_distance": 3.5,
        "degrees": [4, 6],
        "average_type": "both",
        "use_pbc": True,
    }

    def __init__(
        self,
        universe: mda.Universe,
        args: Dict[str, Union[str, float, int, List[int], bool]] = None,
        **kwargs,
    ) -> None:
        super().__init__(args=args, **kwargs)

        self.centers = universe.select_atoms(self.selection_centers)
        self.neighbors = universe.select_atoms(self.selection_neighbors)

        if len(self.centers) == 0:
            self.raise_error(f"No atoms selected for centers with '{self.selection_centers}'")
        if len(self.neighbors) == 0:
            self.raise_error(f"No atoms selected for neighbors with '{self.selection_neighbors}'")

        self.degrees_list = [int(d) for d in self.degrees]
        self.avg_type = str(self.average_type).lower()

        label_names = []
        for l in self.degrees_list:
            if self.avg_type in ["mean_local", "both"]:
                label_names.append(f"q{l}_mean_local")
            if self.avg_type in ["global", "both"]:
                label_names.append(f"Q{l}_global")

        self.label_names = label_names

    def get_label_names(self) -> List[str]:
        """
        Return the names of the label columns.

        Returns
        -------
        list of str
            Label names for the selected Steinhardt order parameters.
        """
        return self.label_names

    def compute(self) -> List[float]:
        """
        Compute the Steinhardt order parameters for the current frame.

        Returns
        -------
        list of float
            The evaluated order parameter values based on configured degrees and average types.
        """
        n_centers = len(self.centers)
        cutoff = float(self.cutoff_distance)

        box = (
            self.centers.universe.dimensions
            if (self.use_pbc and hasattr(self.centers.universe, "dimensions"))
            else None
        )

        # Capped distance neighbor search
        pairs, dists = mda.lib.distances.capped_distance(
            self.centers.positions,
            self.neighbors.positions,
            max_cutoff=cutoff,
            box=box,
        )

        if len(pairs) == 0:
            return [0.0] * len(self.label_names)

        c_indices = pairs[:, 0]
        n_indices = pairs[:, 1]

        # Filter out self-interactions if centers and neighbors overlap
        mask = (self.centers[c_indices].indices != self.neighbors[n_indices].indices)
        if not np.any(mask):
            return [0.0] * len(self.label_names)

        c_indices = c_indices[mask]
        n_indices = n_indices[mask]
        dists = dists[mask]

        # Compute relative displacement vectors r_ij = r_j - r_i
        vecs = self.neighbors.positions[n_indices] - self.centers.positions[c_indices]
        if box is not None:
            vecs = mda.lib.distances.minimize_vectors(vecs, box)

        r = np.linalg.norm(vecs, axis=1)
        r = np.maximum(r, 1e-8)

        cos_theta = np.clip(vecs[:, 2] / r, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        phi = np.arctan2(vecs[:, 1], vecs[:, 0])

        results = []
        n_total_bonds = len(vecs)

        for l in self.degrees_list:
            norm_factor = 4.0 * np.pi / (2.0 * float(l) + 1.0)
            
            # Store q_lm per bond
            q_lm_bonds = {}
            for m in range(-l, l + 1):
                q_lm_bonds[m] = _sph_harm(l, m, theta, phi)

            if self.avg_type in ["mean_local", "both"]:
                # Compute q_lm(i) for each center atom i
                # Accumulate Y_lm values for each center atom
                q_lm_centers = np.zeros((n_centers, 2 * l + 1), dtype=np.complex128)
                bond_counts = np.zeros(n_centers, dtype=np.float64)

                np.add.at(bond_counts, c_indices, 1.0)

                for idx_m, m in enumerate(range(-l, l + 1)):
                    np.add.at(q_lm_centers[:, idx_m], c_indices, q_lm_bonds[m])

                # Average per center atom with at least 1 neighbor
                valid_centers = bond_counts > 0
                if np.any(valid_centers):
                    q_lm_centers[valid_centers] /= bond_counts[valid_centers, None]
                    q_l_atoms = np.sqrt(norm_factor * np.sum(np.abs(q_lm_centers) ** 2, axis=1))
                    mean_q_l = float(np.mean(q_l_atoms[valid_centers]))
                else:
                    mean_q_l = 0.0

                results.append(mean_q_l)

            if self.avg_type in ["global", "both"]:
                # Compute global Q_l over all bonds in the frame
                Q_lm = np.zeros(2 * l + 1, dtype=np.complex128)
                for idx_m, m in enumerate(range(-l, l + 1)):
                    Q_lm[idx_m] = np.mean(q_lm_bonds[m])

                Q_l = float(np.sqrt(norm_factor * np.sum(np.abs(Q_lm) ** 2)))
                results.append(Q_l)

        return results
