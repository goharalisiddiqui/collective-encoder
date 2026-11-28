import os
from typing import Dict, Any, List, Tuple

import ase
import matplotlib
matplotlib.use("Agg")
import MDAnalysis as mda
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def run_dir(tmp_path) -> str:
    """Provide a clean temporary run directory path as a string."""
    d = tmp_path / "run"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


@pytest.fixture
def metargs(run_dir) -> Dict[str, Any]:
    """Provide standard CEModule runtime metadata arguments."""
    return {
        "run_dir": run_dir,
        "verbose": False,
        "root_logger_name": "collective_encoder_test",
    }


@pytest.fixture
def synthetic_universe() -> mda.Universe:
    """
    Create a deterministic in-memory MDAnalysis Universe with 20 frames and 6 atoms,
    including topology attributes (names, types, elements, resnames, resids, bonds)
    and periodic box dimensions.
    """
    n_frames = 20
    n_atoms = 6
    u = mda.Universe.empty(
        n_atoms,
        n_residues=2,
        atom_resindex=np.array([0, 0, 0, 1, 1, 1]),
        residue_segindex=np.array([0, 0]),
        trajectory=True,
    )
    u.add_TopologyAttr("name", ["N", "CA", "C", "N", "CA", "C"])
    u.add_TopologyAttr("type", ["N", "C", "C", "N", "C", "C"])
    u.add_TopologyAttr("elements", ["N", "C", "C", "N", "C", "C"])
    u.add_TopologyAttr("resname", ["ALA", "ALA"])
    u.add_TopologyAttr("resid", [1, 2])
    u.add_TopologyAttr("segid", ["SYSTEM"])
    u.add_TopologyAttr("id", np.arange(n_atoms))
    u.add_TopologyAttr("bonds", [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])

    rng = np.random.default_rng(42)
    base_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.45, 0.0, 0.0],
            [2.10, 1.35, 0.0],
            [3.40, 1.50, 0.2],
            [4.15, 2.75, 0.1],
            [5.50, 2.80, -0.2],
        ],
        dtype=np.float32,
    )
    coords = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    for f in range(n_frames):
        coords[f] = base_coords + 0.05 * f + rng.normal(0.0, 0.02, size=base_coords.shape)

    u.load_new(coords, format="MEMORY")
    for ts in u.trajectory:
        ts.dimensions = np.array([25.0, 25.0, 25.0, 90.0, 90.0, 90.0], dtype=np.float32)
    u.trajectory[0]
    return u


@pytest.fixture
def synthetic_peptide_universe() -> mda.Universe:
    """
    Create a deterministic 3-residue peptide backbone universe (9 atoms: N, CA, C per residue)
    with 15 frames so residue 2 has well-defined phi_2 and psi_2 dihedral angles.
    """
    n_frames = 15
    n_atoms = 9
    u = mda.Universe.empty(
        n_atoms,
        n_residues=3,
        atom_resindex=np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
        residue_segindex=np.array([0, 0, 0]),
        trajectory=True,
    )
    u.add_TopologyAttr("name", ["N", "CA", "C", "N", "CA", "C", "N", "CA", "C"])
    u.add_TopologyAttr("type", ["N", "C", "C", "N", "C", "C", "N", "C", "C"])
    u.add_TopologyAttr("elements", ["N", "C", "C", "N", "C", "C", "N", "C", "C"])
    u.add_TopologyAttr("resname", ["ALA", "ALA", "ALA"])
    u.add_TopologyAttr("resid", [1, 2, 3])
    u.add_TopologyAttr("segid", ["SYSTEM"])
    u.add_TopologyAttr("id", np.arange(n_atoms))
    u.add_TopologyAttr(
        "bonds",
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)],
    )

    rng = np.random.default_rng(99)
    base_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.46, 0.0, 0.0],
            [2.05, 1.40, 0.0],
            [3.35, 1.55, 0.25],
            [4.05, 2.85, 0.15],
            [5.45, 2.95, -0.20],
            [6.15, 4.15, 0.10],
            [7.55, 4.25, 0.35],
            [8.25, 5.55, 0.05],
        ],
        dtype=np.float32,
    )
    coords = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    for f in range(n_frames):
        coords[f] = base_coords + 0.04 * f + rng.normal(0.0, 0.03, size=base_coords.shape)

    u.load_new(coords, format="MEMORY")
    for ts in u.trajectory:
        ts.dimensions = np.array([30.0, 30.0, 30.0, 90.0, 90.0, 90.0], dtype=np.float32)
    u.trajectory[0]
    return u


@pytest.fixture
def synthetic_atoms_list() -> Tuple[List[ase.Atoms], List[np.ndarray]]:
    """
    Return a tuple of (structures, labels) with 20 deterministic 4-atom ASE Atoms
    objects and 2-D labels of shape (2,).
    """
    rng = np.random.default_rng(123)
    base_pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 1.4, 0.0],
            [3.2, 1.6, 0.5],
        ],
        dtype=np.float64,
    )
    numbers = [6, 6, 8, 1]
    structures: List[ase.Atoms] = []
    labels: List[np.ndarray] = []

    for i in range(20):
        pos = base_pos + 0.03 * i + rng.normal(0.0, 0.01, size=base_pos.shape)
        atoms = ase.Atoms(
            numbers=numbers,
            positions=pos,
            cell=[20.0, 20.0, 20.0],
            pbc=True,
        )
        atoms.set_array("residuenames", np.array(["MOL"] * 4))
        atoms.set_array("residuenumbers", np.array([1] * 4))
        atoms.set_array("atomtypes", np.array(["C", "C", "O", "H"]))
        structures.append(atoms)
        labels.append(np.array([np.sin(0.2 * i), np.cos(0.2 * i)], dtype=np.float64))

    return structures, labels


@pytest.fixture
def plumed_colvar_file(tmp_path) -> str:
    """
    Write a valid PLUMED COLVAR file with `#! FIELDS time phi psi ene` and 30 rows.
    """
    colvar_path = tmp_path / "COLVAR"
    rng = np.random.default_rng(7)
    lines = ["#! FIELDS time phi psi ene"]
    for i in range(30):
        t = float(i) * 0.5
        phi = float(np.sin(0.2 * i) + rng.normal(0, 0.01))
        psi = float(np.cos(0.2 * i) + rng.normal(0, 0.01))
        ene = float(-50.0 + 0.5 * i)
        lines.append(f"{t:.4f} {phi:.6f} {psi:.6f} {ene:.6f}")
    colvar_path.write_text("\n".join(lines) + "\n")
    return str(colvar_path)


@pytest.fixture
def plumed_dataframe(plumed_colvar_file) -> pd.DataFrame:
    """Return a pandas DataFrame matching the synthetic PLUMED COLVAR file."""
    df = pd.read_csv(
        plumed_colvar_file,
        sep=r"\s+",
        comment="#",
        names=["time", "phi", "psi", "ene"],
    )
    return df


@pytest.fixture
def trajectory_files(tmp_path, synthetic_universe) -> Tuple[str, str]:
    """
    Write a synthetic `.pdb` topology and `.xtc` trajectory to `tmp_path`
    and return `(pdb_path, xtc_path)`.
    """
    pdb_path = str(tmp_path / "topol.pdb")
    xtc_path = str(tmp_path / "traj.xtc")

    synthetic_universe.trajectory[0]
    synthetic_universe.atoms.write(pdb_path)

    with mda.Writer(xtc_path, synthetic_universe.atoms.n_atoms) as writer:
        for _ in synthetic_universe.trajectory:
            writer.write(synthetic_universe.atoms)

    synthetic_universe.trajectory[0]
    return pdb_path, xtc_path
