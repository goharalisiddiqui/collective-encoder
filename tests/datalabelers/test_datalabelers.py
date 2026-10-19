import numpy as np
import pytest

from collective_encoder.datalabelers import (
    DummyLabeler,
    DistanceValueLabeler,
    DihedralValueLabeler,
    CoordinationCountLabeler,
    StaticStructureFactorLabeler,
    DebyeStructureFactorLabeler,
    SteinhardtOrderParameterLabeler,
    ColumnSelectorLabeler,
    ConcatLabeler,
)


@pytest.mark.unit
def test_dummy_labeler(synthetic_universe, metargs):
    lbl = DummyLabeler(universe=synthetic_universe, args={}, **metargs)
    out_batch = lbl.compute([0, 1, 2])
    assert out_batch.shape == (3, 1)
    assert np.all(out_batch == 0.0)
    assert lbl.compute() == [0.0]
    assert lbl.get_label_names() == ["dummy"]


@pytest.mark.unit
def test_distance_value_labeler(synthetic_universe, metargs):
    lbl = DistanceValueLabeler(
        universe=synthetic_universe,
        args={"selections": ["index 0 or index 1", "index 1 or index 2"]},
        **metargs,
    )
    synthetic_universe.trajectory[0]
    out = lbl.compute()
    assert out.shape == (2,)
    assert np.all(out > 0.0)
    assert lbl.get_label_names() == ["distance_value_1", "distance_value_2"]

    # Selection that does not match exactly 2 atoms raises ValueError
    with pytest.raises(ValueError, match="2 atoms"):
        DistanceValueLabeler(
            universe=synthetic_universe,
            args={"selections": ["index 0"]},
            **metargs,
        )


@pytest.mark.unit
def test_dihedral_value_labeler(synthetic_peptide_universe, metargs):
    lbl = DihedralValueLabeler(
        universe=synthetic_peptide_universe,
        args={"label_dihedrals": ["phi_2", "psi_2"]},
        **metargs,
    )
    synthetic_peptide_universe.trajectory[0]
    out = lbl.compute()
    assert out.shape == (2,)
    assert lbl.get_label_names() == ["phi_2", "psi_2"]
    assert np.all(np.isfinite(out))

    # Invalid dihedral prefix raises ValueError
    with pytest.raises(ValueError, match="phi_"):
        DihedralValueLabeler(
            universe=synthetic_peptide_universe,
            args={"label_dihedrals": ["omega_2"]},
            **metargs,
        )


@pytest.mark.unit
def test_coordination_count_labeler(synthetic_universe, metargs):
    lbl = CoordinationCountLabeler(
        universe=synthetic_universe,
        args={
            "selection_centers": "index 0 or index 1",
            "selection_neighbors": "all",
            "cutoff_distance": 3.0,
            "bins": [1, 2],
        },
        **metargs,
    )
    synthetic_universe.trajectory[0]
    out = lbl.compute()
    assert out.shape == (2,)
    assert np.all(np.isfinite(out))
    assert lbl.get_label_names() == ["coordination_count_1", "coordination_count_2"]


@pytest.mark.unit
def test_structure_factor_and_debye_labelers(synthetic_universe, metargs):
    sfac = StaticStructureFactorLabeler(
        universe=synthetic_universe,
        args={"q_min": 0.5, "q_max": 1.5, "q_num": 3, "selection": "all"},
        **metargs,
    )
    synthetic_universe.trajectory[0]
    out_sfac = sfac.compute()
    assert out_sfac.shape == (3,)
    assert len(sfac.get_label_names()) == 3
    assert np.all(np.isfinite(out_sfac))

    debye = DebyeStructureFactorLabeler(
        universe=synthetic_universe,
        args={
            "q_min": 0.5,
            "q_max": 1.5,
            "q_num": 3,
            "selection": "all",
            "output_mode": "intensity",
        },
        **metargs,
    )
    out_debye = debye.compute()
    assert out_debye.shape == (3,)
    assert len(debye.get_label_names()) == 3
    assert all(name.startswith("debye_iq") for name in debye.get_label_names())
    assert np.all(np.isfinite(out_debye))


@pytest.mark.unit
def test_steinhardt_order_parameter_labeler(synthetic_universe, metargs):
    ql = SteinhardtOrderParameterLabeler(
        universe=synthetic_universe,
        args={
            "degrees": [4, 6],
            "cutoff_distance": 4.0,
            "average_type": "both",
            "selection_centers": "all",
            "selection_neighbors": "all",
        },
        **metargs,
    )
    synthetic_universe.trajectory[0]
    out = ql.compute()
    assert out.shape == (4,)
    assert ql.get_label_names() == [
        "q4_mean_local",
        "Q4_global",
        "q6_mean_local",
        "Q6_global",
    ]
    assert np.all(np.isfinite(out))


@pytest.mark.unit
def test_column_selector_labeler(plumed_dataframe, metargs):
    lbl = ColumnSelectorLabeler(
        dataframe=plumed_dataframe,
        args={"columns": ["phi", "psi"]},
        **metargs,
    )
    out = lbl.compute([0, 1, 2, 3])
    assert out.shape == (4, 2)
    assert lbl.get_label_names() == ["phi", "psi"]
    np.testing.assert_allclose(out, plumed_dataframe.loc[[0, 1, 2, 3], ["phi", "psi"]].values)

    # Missing column raises ValueError
    with pytest.raises(ValueError, match="not present"):
        ColumnSelectorLabeler(
            dataframe=plumed_dataframe,
            args={"columns": ["nonexistent_col"]},
            **metargs,
        )


@pytest.mark.unit
def test_concat_labeler(synthetic_universe, plumed_dataframe, metargs):
    # FrameLabeler mode (Universe)
    concat_frame = ConcatLabeler(
        universe=synthetic_universe,
        args={
            "labelers": [
                {
                    "type": "DISTANCE",
                    "args": {"selections": ["index 0 or index 1"]},
                },
                {
                    "type": "DISTANCE",
                    "args": {"selections": ["index 1 or index 2"]},
                },
            ]
        },
        **metargs,
    )
    synthetic_universe.trajectory[0]
    out_frame = concat_frame.compute()
    assert len(out_frame) == 2
    assert concat_frame.get_label_names() == ["distance_value_1", "distance_value_1"]

    # BatchLabeler mode (DataFrame)
    concat_batch = ConcatLabeler(
        dataframe=plumed_dataframe,
        args={
            "labelers": [
                {"type": "COLUMN_SELECTOR", "args": {"columns": ["phi"]}},
                {"type": "COLUMN_SELECTOR", "args": {"columns": ["psi"]}},
            ]
        },
        **metargs,
    )
    out_batch = concat_batch.compute([0, 1, 2])
    assert out_batch.shape == (3, 2)
    assert concat_batch.get_label_names() == ["phi", "psi"]
