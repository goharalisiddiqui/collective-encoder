import ase
import numpy as np
import pytest

from collective_encoder.datareaders.compressed_trajectory import CompressedTrajectoryReader
from collective_encoder.datareaders.plumed_output import PlumedOutputReader
from collective_encoder.datareaders.processors.chunk import ChunkProcessor
from collective_encoder.datareaders.processors.coarse_grain import CoarseGrainProcessor
from collective_encoder.datareaders.processors.filter import FilterProcessor
from collective_encoder.datareaders.processors.resolver import get_processor


@pytest.mark.unit
def test_compressed_trajectory_reader_seq_and_parallel(trajectory_files, metargs):
    pdb_path, xtc_path = trajectory_files

    # Sequential mode
    reader_seq = CompressedTrajectoryReader(
        args={
            "topology_file": pdb_path,
            "trajectory_file": xtc_path,
            "topology_format": "PDB",
            "selection": "all",
            "parallel": False,
        },
        **metargs,
    )
    assert reader_seq.get_total_frames() == 20
    assert len(reader_seq.get_atomic_numbers()) == 6
    assert len(reader_seq.get_element_symbols()) == 6
    assert len(reader_seq.get_atom_ids()) == 6
    assert len(reader_seq.get_bonds()) == 5

    splits = [[0, 1, 2, 3], [4, 5], [6, 7], []]
    trajs_seq, labels_seq, failed_seq = reader_seq.read_trajectory(
        indices=splits,
        labeler_type="DISTANCE",
        labeler_args={"selections": ["index 0 or index 1"]},
    )
    assert len(trajs_seq) == 4
    assert len(trajs_seq[0]) == 4
    assert len(trajs_seq[1]) == 2
    assert len(trajs_seq[3]) == 0
    assert failed_seq == ([], [], [], [])
    assert reader_seq.get_label_names() == ["distance_value_1"]

    # Parallel mode (>= 8 frames in a split triggers multiprocessing Pool)
    reader_par = CompressedTrajectoryReader(
        args={
            "topology_file": pdb_path,
            "trajectory_file": xtc_path,
            "topology_format": "PDB",
            "selection": "all",
            "parallel": True,
        },
        **metargs,
    )
    par_splits = [list(range(10)), [10, 11], [], []]
    trajs_par, labels_par, failed_par = reader_par.read_trajectory(
        indices=par_splits,
        labeler_type="DUMMY",
        labeler_args={},
    )
    assert len(trajs_par[0]) == 10
    assert failed_par == ([], [], [], [])
    for i in range(4):
        np.testing.assert_allclose(
            trajs_seq[0][i].get_positions(),
            trajs_par[0][i].get_positions(),
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.unit
def test_compressed_trajectory_reader_parallel_frame_labeler_reinitialize(trajectory_files, metargs):
    pdb_path, xtc_path = trajectory_files
    reader_seq = CompressedTrajectoryReader(
        args={
            "topology_file": pdb_path,
            "trajectory_file": xtc_path,
            "topology_format": "PDB",
            "parallel": False,
        },
        **metargs,
    )
    _, labels_seq, _ = reader_seq.read_trajectory(
        indices=[list(range(10)), [], [], []],
        labeler_type="DISTANCE",
        labeler_args={"selections": ["index 0 or index 1"]},
    )

    reader_par = CompressedTrajectoryReader(
        args={
            "topology_file": pdb_path,
            "trajectory_file": xtc_path,
            "topology_format": "PDB",
            "parallel": True,
        },
        **metargs,
    )
    _, labels_par, _ = reader_par.read_trajectory(
        indices=[list(range(10)), [], [], []],
        labeler_type="DISTANCE",
        labeler_args={"selections": ["index 0 or index 1"]},
    )
    np.testing.assert_allclose(labels_seq[0], labels_par[0], rtol=1e-5, atol=1e-5)


@pytest.mark.unit
def test_compressed_trajectory_reader_invalid_selection_and_missing_files(trajectory_files, metargs):
    pdb_path, xtc_path = trajectory_files

    with pytest.raises(FileNotFoundError):
        CompressedTrajectoryReader(
            args={
                "topology_file": "/nonexistent.pdb",
                "trajectory_file": xtc_path,
                "topology_format": "PDB",
            },
            **metargs,
        )

    with pytest.raises(ValueError, match="does not match any atoms"):
        CompressedTrajectoryReader(
            args={
                "topology_file": pdb_path,
                "trajectory_file": xtc_path,
                "topology_format": "PDB",
                "selection": "resname NONEXISTENT",
            },
            **metargs,
        )


@pytest.mark.unit
def test_plumed_output_reader_modes_and_trajectory_reading(plumed_colvar_file, metargs):
    # 1. Read all columns (columns=None, column_match=None)
    reader_all = PlumedOutputReader(
        args={"plumed_file": plumed_colvar_file},
        **metargs,
    )
    assert reader_all.get_total_frames() == 30
    assert set(reader_all.data.columns) >= {"phi", "psi", "ene"}

    # 2. Read specific columns
    reader_cols = PlumedOutputReader(
        args={
            "plumed_file": plumed_colvar_file,
            "columns": ["phi", "psi", "ene"],
        },
        **metargs,
    )
    trajs, labels, failed = reader_cols.read_trajectory(
        indices=[[0, 1, 2, 3], [4, 5], [], []],
        labeler_type="COLUMN_SELECTOR",
        labeler_args={"columns": ["ene"]},
    )
    assert len(trajs) == 4
    assert trajs[0].shape == (4, 2)  # 'ene' dropped from features -> ['phi', 'psi']
    assert labels[0].shape == (4, 1)
    assert trajs[2].shape == (0, 2)
    assert labels[2].shape == (0, 1)
    assert failed == ([], [], [], [])
    assert reader_cols.get_label_names() == ["ene"]

    # 3. Read with column_match
    reader_match = PlumedOutputReader(
        args={
            "plumed_file": plumed_colvar_file,
            "column_match": "p",
        },
        **metargs,
    )
    assert set(reader_match.data.columns) == {"phi", "psi"}

    # 4. Mutually exclusive columns + column_match raises error
    with pytest.raises(Exception):
        PlumedOutputReader(
            args={
                "plumed_file": plumed_colvar_file,
                "columns": ["phi"],
                "column_match": "p",
            },
            **metargs,
        )


@pytest.mark.unit
def test_trajectory_processors_unit():
    # Resolver
    assert get_processor("Chunk") is ChunkProcessor
    assert get_processor("CoarseGrain") is CoarseGrainProcessor
    assert get_processor("Filter") is FilterProcessor
    with pytest.raises(ValueError, match="Unknown processor"):
        get_processor("NONEXISTENT_PROC")

    # ChunkProcessor
    chunk_proc = ChunkProcessor(sequence_length=3)
    expanded = chunk_proc.prepare_seq([10, 20])
    assert expanded == [10, 11, 12, 20, 21, 22]
    assert chunk_proc.adjust_total_frames(30) == 27

    # CoarseGrainProcessor
    cg_proc = CoarseGrainProcessor(cg_window=2, sequence_length=1)
    cg_expanded = cg_proc.prepare_seq([0, 5])
    assert cg_expanded == [0, 1, 5, 6]
    assert cg_proc.adjust_total_frames(20) == 18

    s0 = ase.Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], cell=[10, 10, 10], pbc=True)
    s1 = ase.Atoms("H2", positions=[[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], cell=[10, 10, 10], pbc=True)
    cg_traj, cg_lbl = cg_proc.postprocess_seq(
        [s0, s1, s0, s1],
        [np.array([1.0]), np.array([3.0]), np.array([2.0]), np.array([4.0])],
    )
    assert len(cg_traj) == 2
    np.testing.assert_allclose(cg_traj[0].get_positions(), [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    np.testing.assert_allclose(cg_lbl[0], [2.0])

    # FilterProcessor
    filt_proc = FilterProcessor(label_index=0, min_val=0.0, max_val=2.0)
    f_traj, f_lbl = filt_proc.postprocess_seq(
        ["a", "b", "c"],
        [np.array([1.0]), np.array([5.0]), np.array([1.5])],
    )
    assert f_traj == ["a", "c"]
    assert len(f_lbl) == 2
