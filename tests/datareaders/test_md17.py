import os
import tarfile
import pytest
import numpy as np

from collective_encoder.datareaders.datasets.md17 import MD17Reader
from collective_encoder.common.registry import get_module


def test_resolver_md17():
    cls = get_module("datareaders", "MD17")
    assert cls is MD17Reader


def test_md17_molecule_validation():
    # Valid molecules
    for mol in [
        "aspirin",
        "azobenzene",
        "benzene",
        "ethanol",
        "malonaldehyde",
        "naphthalene",
        "paracetamol",
        "salicylic",
        "salicylic_acid",
        "toluene",
        "uracil",
    ]:
        reader = MD17Reader.__new__(MD17Reader)
        reader.molecule_name = mol
        reader._validate_molecule()
        assert reader.molecule_name == mol.lower()

    # Invalid molecule
    with pytest.raises(Exception) as exc_info:
        MD17Reader(args={"molecule_name": "unknown_molecule"})
    assert "Invalid molecule" in str(exc_info.value)


def test_md17_target_filepath_resolution(tmp_path):
    reader = MD17Reader.__new__(MD17Reader)
    reader.molecule_name = "salicylic_acid"
    reader._molecule_file_key = "salicylic"
    reader.data_dir = str(tmp_path)

    # When file doesn't exist, returns default extract path
    default_path = os.path.join(str(tmp_path), "rmd17", "npz_data", "rmd17_salicylic.npz")
    assert reader._get_target_filepath() == default_path

    # When file exists in data_dir directly
    direct_file = tmp_path / "rmd17_salicylic.npz"
    direct_file.touch()
    assert reader._get_target_filepath() == str(direct_file)


def test_md17_offline_archive_extraction(tmp_path, monkeypatch):
    """Test that archive extraction and loading work with a simulated rMD17 tarball."""
    monkeypatch.setattr("collective_encoder.common.module.EXECUTION_DIR", str(tmp_path))
    data_dir = tmp_path / "md17_data"
    data_dir.mkdir(parents=True)

    # Create dummy .npz
    npz_dir = tmp_path / "pack" / "rmd17" / "npz_data"
    npz_dir.mkdir(parents=True)
    npz_path = npz_dir / "rmd17_benzene.npz"

    n_frames, n_atoms = 5, 12
    coords = np.zeros((n_frames, n_atoms, 3), dtype=np.float64)
    nuclear_charges = np.array([6] * 6 + [1] * 6, dtype=np.uint8)
    energies = np.linspace(-100.0, -90.0, n_frames)
    forces = np.ones((n_frames, n_atoms, 3), dtype=np.float64)

    np.savez(
        npz_path,
        coords=coords,
        nuclear_charges=nuclear_charges,
        energies=energies,
        forces=forces,
    )

    # Create tar.bz2
    tar_path = data_dir / "rmd17.tar.bz2"
    with tarfile.open(tar_path, "w:bz2") as tar:
        tar.add(tmp_path / "pack" / "rmd17", arcname="rmd17")

    reader = MD17Reader(
        args={"molecule_name": "benzene"},
        run_dir=str(tmp_path),
    )

    assert reader.get_total_frames() == n_frames
    assert len(reader.u.atoms) == n_atoms
    assert list(reader.raw_data["nuclear_charges"]) == list(nuclear_charges)
    assert np.allclose(reader.raw_data["energies"], energies)


def test_md17_custom_data_dir_arg_respected(tmp_path, monkeypatch):
    monkeypatch.setattr("collective_encoder.common.module.EXECUTION_DIR", str(tmp_path / "exec_root"))
    custom_dir = tmp_path / "custom_md17"
    npz_dir = custom_dir / "rmd17" / "npz_data"
    npz_dir.mkdir(parents=True)
    np.savez(
        npz_dir / "rmd17_benzene.npz",
        coords=np.zeros((3, 12, 3), dtype=np.float64),
        nuclear_charges=np.array([6] * 6 + [1] * 6, dtype=np.uint8),
        energies=np.zeros(3),
        forces=np.zeros((3, 12, 3)),
    )
    reader = MD17Reader(
        args={"molecule_name": "benzene", "data_dir": str(custom_dir)},
        exec_dir=str(tmp_path / "other_exec_dir"),
    )
    assert reader.get_total_frames() == 3


@pytest.mark.skipif(
    not os.path.exists("md17_data/rmd17/npz_data/rmd17_benzene.npz"),
    reason="Cached rMD17 benzene dataset not present locally in md17_data/",
)
def test_md17_integration_with_cached_dataset(tmp_path):
    """Integration test using the locally cached rMD17 benzene dataset."""
    reader = MD17Reader(
        args={"molecule_name": "benzene"},
    )

    assert reader.get_total_frames() == 100000
    assert len(reader.u.atoms) == 12
    assert list(reader.u.atoms.elements[:6]) == ["C"] * 6
    assert list(reader.u.atoms.elements[6:]) == ["H"] * 6

    # Verify native rMD17 keys
    assert "coords" in reader.raw_data
    assert "nuclear_charges" in reader.raw_data
    assert "energies" in reader.raw_data
    assert "forces" in reader.raw_data

    # Read trajectory with FromExternalLabeler using native rMD17 keys
    trajs, labels, failed = reader.read_trajectory(
        indices=[[0, 1]],
        labeler_type="FROM_EXTERNAL",
        labeler_args={"keys": ["energies", "forces"]},
    )

    assert len(trajs) == 1
    assert len(trajs[0]) == 2
    assert len(labels) == 1
    assert len(labels[0]) == 2

    # Check label names: 1 energy + 12*3 forces = 37 columns
    label_names = reader.get_label_names()
    assert len(label_names) == 37
    assert label_names[0] == "energies"
    assert "forces_0_0" in label_names

    # Check label values
    l0 = np.asarray(labels[0][0]).flatten()
    assert np.isclose(l0[0], reader.raw_data["energies"][0])
    assert np.allclose(l0[1:], reader.raw_data["forces"][0].flatten())
