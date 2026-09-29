import logging
from typing import Dict, List, Union, Optional

import MDAnalysis as mda
import numpy as np

from .base import FrameLabeler

_log = logging.getLogger(__name__)


class StaticStructureFactorLabeler(FrameLabeler):
    """Compute the isotropic static structure factor S(q) for the current trajectory frame.

    Evaluates S(q) using the Debye scattering formula:
        S(q) = 1 + (2 / N) * sum_{j < k} sin(q * r_{jk}) / (q * r_{jk})

    Periodic boundary conditions (PBC) are automatically applied if box dimensions are
    present in the MDAnalysis universe and ``use_pbc`` is True.

    Args:
        universe: MDAnalysis Universe loaded with topology and trajectory.
            The trajectory must be positioned at the target frame before calling :meth:`compute`.
        args: Configuration dict with keys:
            - ``selection`` (str, optional): MDAnalysis atom selection string.
              Defaults to ``'all'``.
            - ``q_values`` (List[float], optional): Explicit list of scattering wavenumbers in Å⁻¹.
            - ``q_min`` (float, optional): Minimum q value in Å⁻¹ (used if ``q_values`` is omitted).
              Defaults to ``0.5``.
            - ``q_max`` (float, optional): Maximum q value in Å⁻¹ (used if ``q_values`` is omitted).
              Defaults to ``5.0``.
            - ``q_num`` (int, optional): Number of q values (used if ``q_values`` is omitted).
              Defaults to ``10``.
            - ``use_pbc`` (bool, optional): Whether to apply PBC using universe box dimensions.
              Defaults to ``True``.
    """

    _IDENTIFIER = "STATIC_STRUCTURE_FACTOR"
    _REQUIRED_ARGS = []
    _OPTIONAL_ARGS = {
        "selection": "all",
        "q_values": None,
        "q_min": 0.5,
        "q_max": 5.0,
        "q_num": 10,
        "use_pbc": True,
    }

    def __init__(
        self,
        universe: mda.Universe,
        args: Dict[str, Union[str, float, int, List[float], bool]] = None,
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

        self.label_names = [f"sq_q{q:.2f}" for q in self.q_arr]
        self.log_info(f"Initialized StaticStructureFactorLabeler with "
                      f"{len(self.atoms)} atoms and q values: {self.q_arr}")

    def get_label_names(self) -> List[str]:
        return self.label_names

    def compute(self) -> List[float]:
        n_atoms = len(self.atoms)
        if n_atoms < 2:
            return [1.0] * len(self.q_arr)

        positions = self.atoms.positions
        box = (
            self.atoms.universe.dimensions
            if (self.use_pbc and hasattr(self.atoms.universe, "dimensions"))
            else None
        )

        # Compute all N(N-1)/2 pairwise distances
        dists = mda.lib.distances.self_distance_array(positions, box=box)

        # Vectorized Debye formula computation for all q values
        # q_arr shape: (K, 1), dists shape: (1, M) -> QR shape: (K, M)
        QR = self.q_arr[:, None] * dists[None, :]
        sinc_QR = np.where(QR > 1e-7, np.sin(QR) / QR, 1.0)
        
        # S(q) = 1 + (2 / N) * sum_{j < k} sin(q * r_{jk}) / (q * r_{jk})
        S_q = 1.0 + (2.0 / float(n_atoms)) * np.sum(sinc_QR, axis=1)
        return S_q.tolist()
