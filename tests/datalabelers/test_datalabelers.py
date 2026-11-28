import numpy as np
import pytest

from collective_encoder.datalabelers import (
    DummyLabeler,
    DistanceValueLabeler,
    DihedralValueLabeler,
    TopologyLabeler,
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
    assert isinstance(out_batch, dict)
    assert list(out_batch.keys()) == ["dummylabeler"]
    assert out_batch["dummylabeler"].shape == (3, 1)
    assert np.all(out_batch["dummylabeler"] == 0.0)
    out_frame = lbl.compute()
    assert isinstance(out_frame, dict)
    np.testing.assert_allclose(out_frame["dummylabeler"], [0.0])
    assert lbl.get_label_names() == ["dummylabeler"]

    # Custom name parameter
    lbl_named = DummyLabeler(universe=synthetic_universe, args={"name": "my_dummy"}, **metargs)
    assert lbl_named.get_label_names() == ["my_dummy"]
    assert "my_dummy" in lbl_named.compute()


@pytest.mark.unit
def test_distance_value_labeler(synthetic_universe, metargs):
    lbl = DistanceValueLabeler(
        universe=synthetic_universe,
        args={"selections": ["index 0 or index 1", "index 1 or index 2"]},
        **metargs,
    )
    synthetic_universe.trajectory[0]
    out = lbl.compute()
    assert isinstance(out, dict)
    assert list(out.keys()) == ["distancevaluelabeler"]
    assert out["distancevaluelabeler"].shape == (2,)
    assert np.all(out["distancevaluelabeler"] > 0.0)
    assert lbl.get_label_names() == ["distancevaluelabeler"]

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
        args={
            "label_dihedrals": [
                {
                    "phi_2": "(resid 1 and name C) or (resid 2 and (name N or name CA or name C))"
                },
                {
                    "psi_2": "(resid 2 and (name N or name CA or name C)) or (resid 3 and name N)"
                },
            ]
        },
        **metargs,
    )
    synthetic_peptide_universe.trajectory[0]
    out = lbl.compute()
    assert isinstance(out, dict)
    assert list(out.keys()) == ["phi_2", "psi_2"]
    assert out["phi_2"].shape == (1,)
    assert out["psi_2"].shape == (1,)
    assert lbl.get_label_names() == ["phi_2", "psi_2"]
    assert np.all(np.isfinite(out["phi_2"])) and np.all(np.isfinite(out["psi_2"]))

    # Selection that does not match 4 atoms raises informative ValueError
    with pytest.raises(ValueError, match="must select exactly 4 atoms"):
        DihedralValueLabeler(
            universe=synthetic_peptide_universe,
            args={
                "label_dihedrals": [
                    {"bad_count": "resid 2 and (name N or name CA or name C)"}
                ]
            },
            **metargs,
        )

    # 4 atoms that are not bonded in a 3-bond dihedral chain raises informative ValueError
    with pytest.raises(ValueError, match="do not form a valid 4-atom dihedral chain"):
        DihedralValueLabeler(
            universe=synthetic_peptide_universe,
            args={
                "label_dihedrals": [
                    {
                        "disconnected": "(resid 1 and (name N or name CA)) or (resid 3 and (name CA or name C))"
                    }
                ]
            },
            **metargs,
        )

    # Non-dict entry raises informative ValueError
    with pytest.raises(ValueError, match="Expected a dictionary"):
        DihedralValueLabeler(
            universe=synthetic_peptide_universe,
            args={"label_dihedrals": ["phi_2"]},
            **metargs,
        )


@pytest.mark.unit
def test_topology_labeler(synthetic_peptide_universe, metargs):
    lbl = TopologyLabeler(
        universe=synthetic_peptide_universe,
        args={"selection": "all"},
        **metargs,
    )
    assert lbl.get_label_names() == ["bonds", "angles", "cos_torsions", "sin_torsions"]
    synthetic_peptide_universe.trajectory[0]
    out = lbl.compute()
    assert isinstance(out, dict)
    assert list(out.keys()) == ["bonds", "angles", "cos_torsions", "sin_torsions"]
    # 9-atom linear peptide chain: 8 bonds, 7*2=14 directed angles, 6*2=12 directed torsions
    assert out["bonds"].shape == (8,)
    assert out["angles"].shape == (14,)
    assert out["cos_torsions"].shape == (12,)
    assert out["sin_torsions"].shape == (12,)
    np.testing.assert_allclose(
        out["cos_torsions"] ** 2 + out["sin_torsions"] ** 2,
        np.ones(12),
        atol=1e-6,
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
    assert isinstance(out, dict)
    assert list(out.keys()) == ["coordinationcountlabeler"]
    assert out["coordinationcountlabeler"].shape == (2,)
    assert np.all(np.isfinite(out["coordinationcountlabeler"]))
    assert lbl.get_label_names() == ["coordinationcountlabeler"]


@pytest.mark.unit
def test_structure_factor_and_debye_labelers(synthetic_universe, metargs):
    sfac = StaticStructureFactorLabeler(
        universe=synthetic_universe,
        args={"q_min": 0.5, "q_max": 1.5, "q_num": 3, "selection": "all"},
        **metargs,
    )
    synthetic_universe.trajectory[0]
    out_sfac = sfac.compute()
    assert isinstance(out_sfac, dict)
    assert list(out_sfac.keys()) == ["staticstructurefactorlabeler"]
    assert out_sfac["staticstructurefactorlabeler"].shape == (3,)
    assert sfac.get_label_names() == ["staticstructurefactorlabeler"]
    assert np.all(np.isfinite(out_sfac["staticstructurefactorlabeler"]))

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
    assert isinstance(out_debye, dict)
    assert list(out_debye.keys()) == ["debyestructurefactorlabeler"]
    assert out_debye["debyestructurefactorlabeler"].shape == (3,)
    assert debye.get_label_names() == ["debyestructurefactorlabeler"]
    assert np.all(np.isfinite(out_debye["debyestructurefactorlabeler"]))


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
    assert isinstance(out, dict)
    assert list(out.keys()) == ["steinhardtorderparameterlabeler"]
    assert out["steinhardtorderparameterlabeler"].shape == (4,)
    assert ql.get_label_names() == ["steinhardtorderparameterlabeler"]
    assert np.all(np.isfinite(out["steinhardtorderparameterlabeler"]))


@pytest.mark.unit
def test_column_selector_labeler(plumed_dataframe, metargs):
    lbl = ColumnSelectorLabeler(
        dataframe=plumed_dataframe,
        args={"columns": ["phi", "psi"]},
        **metargs,
    )
    out = lbl.compute([0, 1, 2, 3])
    assert isinstance(out, dict)
    assert list(out.keys()) == ["columnselectorlabeler"]
    assert out["columnselectorlabeler"].shape == (4, 2)
    assert lbl.get_label_names() == ["columnselectorlabeler"]
    np.testing.assert_allclose(
        out["columnselectorlabeler"],
        plumed_dataframe.loc[[0, 1, 2, 3], ["phi", "psi"]].values,
    )

    # Missing column raises ValueError
    with pytest.raises(ValueError, match="not present"):
        ColumnSelectorLabeler(
            dataframe=plumed_dataframe,
            args={"columns": ["nonexistent_col"]},
            **metargs,
        )


@pytest.mark.unit
def test_concat_labeler(synthetic_peptide_universe, plumed_dataframe, metargs):
    # FrameLabeler mode: two DISTANCE labelers with default names merge into one key,
    # while a named DISTANCE labeler and TOPOLOGY labeler create new keys.
    concat_frame = ConcatLabeler(
        universe=synthetic_peptide_universe,
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
                {
                    "type": "DISTANCE",
                    "args": {"name": "custom_dist", "selections": ["index 2 or index 3"]},
                },
                {
                    "type": "TOPOLOGY",
                    "args": {},
                },
            ]
        },
        **metargs,
    )
    synthetic_peptide_universe.trajectory[0]
    out_frame = concat_frame.compute()
    assert isinstance(out_frame, dict)
    assert concat_frame.get_label_names() == [
        "distancevaluelabeler",
        "custom_dist",
        "bonds",
        "angles",
        "cos_torsions",
        "sin_torsions",
    ]
    assert out_frame["distancevaluelabeler"].shape == (2,)
    assert out_frame["custom_dist"].shape == (1,)
    assert out_frame["bonds"].shape == (8,)

    # BatchLabeler mode (DataFrame): two COLUMN_SELECTOR labelers merge under "columnselectorlabeler"
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
    assert isinstance(out_batch, dict)
    assert concat_batch.get_label_names() == ["columnselectorlabeler"]
    assert out_batch["columnselectorlabeler"].shape == (3, 2)
    np.testing.assert_allclose(
        out_batch["columnselectorlabeler"],
        plumed_dataframe.loc[[0, 1, 2], ["phi", "psi"]].values,
    )
