import os
import pytest
import numpy as np
import MDAnalysis as mda

from collective_encoder.datalabelers.from_external import FromExternalLabeler
from collective_encoder.datalabelers.resolver import get_labeler
from collective_encoder.datareaders.external_dataset import ExternalDatasetReaderBase


def create_dummy_universe(n_frames=10, n_atoms=4):
    """Create a minimal in-memory MDAnalysis Universe for testing."""
    u = mda.Universe.empty(n_atoms, trajectory=True)
    coords = np.zeros((n_frames, n_atoms, 3))
    for i in range(n_frames):
        coords[i] = float(i)
    u.load_new(coords, format="MEMORY")
    return u


def test_from_external_1d_and_2d():
    n_frames = 10
    dataset = {
        "E": np.linspace(1.0, 10.0, n_frames),  # 1D array (N,)
        "dipole": np.arange(n_frames * 3).reshape(n_frames, 3),  # 2D array (N, 3)
    }
    u = create_dummy_universe(n_frames=n_frames, n_atoms=3)

    labeler = FromExternalLabeler(
        dataset=dataset,
        universe=u,
        args={"keys": ["E", "dipole"]},
    )

    expected_names = ["E", "dipole_0", "dipole_1", "dipole_2"]
    assert labeler.get_label_names() == expected_names

    # Test compute
    val = labeler.compute([2])
    assert isinstance(val, np.ndarray)
    assert val.shape == (1, 4)
    assert val[0, 0] == 3.0
    assert np.allclose(val[0, 1:], [6.0, 7.0, 8.0])

    # Test batch compute
    batch_val = labeler.compute([0, 1, 2])
    assert isinstance(batch_val, np.ndarray)
    assert batch_val.shape == (3, 4)
    assert np.allclose(batch_val[2], [3.0, 6.0, 7.0, 8.0])


def test_from_external_3d_multi_index_naming():
    n_frames = 5
    n_atoms = 3
    dataset = {
        "F": np.arange(n_frames * n_atoms * 3, dtype=float).reshape(n_frames, n_atoms, 3)
    }
    u = create_dummy_universe(n_frames=n_frames, n_atoms=n_atoms)

    labeler = FromExternalLabeler(
        dataset=dataset,
        universe=u,
        args={"keys": ["F"]},
    )

    names = labeler.get_label_names()
    expected_names = [
        "F_0_0", "F_0_1", "F_0_2",
        "F_1_0", "F_1_1", "F_1_2",
        "F_2_0", "F_2_1", "F_2_2",
    ]
    assert names == expected_names

    val = labeler.compute([1])
    assert np.allclose(val[0], dataset["F"][1].flatten())


def test_from_external_single_column_2d():
    n_frames = 5
    dataset = {
        "energy_2d": np.arange(n_frames, dtype=float).reshape(n_frames, 1)
    }
    u = create_dummy_universe(n_frames=n_frames, n_atoms=2)

    labeler = FromExternalLabeler(
        dataset=dataset,
        universe=u,
        args={"keys": ["energy_2d"]},
    )
    assert labeler.get_label_names() == ["energy_2d"]
    assert np.allclose(labeler.compute([3]), [[3.0]])


def test_resolver_lookup():
    cls1 = get_labeler("FROM_EXTERNAL")
    cls2 = get_labeler("from_external")
    assert cls1 is FromExternalLabeler
    assert cls2 is FromExternalLabeler


def test_error_handling():
    # Missing dataset
    with pytest.raises(Exception):
        FromExternalLabeler(args={"keys": ["E"]})

    # Missing keys in args
    with pytest.raises(Exception):
        FromExternalLabeler(dataset={"E": [1, 2]}, args={})

    # Unknown key
    with pytest.raises(Exception):
        FromExternalLabeler(dataset={"E": [1, 2]}, args={"keys": ["nonexistent"]})

    # Missing universe when compute() called without indices
    labeler = FromExternalLabeler(dataset={"E": [1, 2]}, universe=None, args={"keys": ["E"]})
    with pytest.raises(Exception):
        labeler.compute()


class MockExternalReader(ExternalDatasetReaderBase):
    _IDENTIFIER = "MOCK_EXTERNAL"

    def _validate_molecule(self):
        pass

    def _get_download_url(self):
        return "http://dummy/url/mock.npz"

    def _download_and_load(self):
        n_frames = 10
        n_atoms = 3
        self.raw_data = {
            "R": np.random.rand(n_frames, n_atoms, 3).astype(np.float32),
            "z": np.array([6, 1, 1], dtype=np.int32),
            "E": np.linspace(10.0, 20.0, n_frames).astype(np.float32),
            "F": np.ones((n_frames, n_atoms, 3), dtype=np.float32) * 5.0,
        }
        self.dataset = self.raw_data
        self.run_args["dataset"] = self.dataset
        self._construct_universe()


def test_integration_with_external_dataset_reader(tmp_path):
    reader = MockExternalReader(args={"molecule_name": "test_mol"}, run_dir=str(tmp_path))

    # Read trajectory with FROM_EXTERNAL labeler
    trajs, labels, failed = reader.read_trajectory(
        indices=[[0, 1, 2], [5, 6]],
        labeler_type="FROM_EXTERNAL",
        labeler_args={"keys": ["E", "F"]},
    )

    assert len(trajs) == 2
    assert len(trajs[0]) == 3
    assert len(trajs[1]) == 2

    # Check label names
    label_names = reader.get_label_names()
    assert label_names[0] == "E"
    assert "F_0_0" in label_names
    assert len(label_names) == 1 + 3 * 3  # 1 for E, 9 for F

    # Check values for split 0
    assert len(labels[0]) == 3
    for frame_idx, l in enumerate(labels[0]):
        l_flat = np.asarray(l).flatten()
        assert len(l_flat) == 10
        assert np.isclose(l_flat[0], reader.raw_data["E"][frame_idx])
        assert np.allclose(l_flat[1:], reader.raw_data["F"][frame_idx].flatten())
