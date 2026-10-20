import os
import pytest
import numpy as np

from collective_encoder.common.module import CEModule
from collective_encoder.common.config_check import (
    validate_duplicate_keys,
    validate_required_fields,
)


class SampleModule(CEModule):
    _IDENTIFIER = "SAMPLE"
    _REQUIRED_ARGS = ["req_a", "req_b"]
    _OPTIONAL_ARGS = {"opt_x": 10, "opt_y": "default_y"}


@pytest.mark.unit
def test_cemodule_required_and_optional_args(metargs):
    # Missing required arg raises ValueError
    with pytest.raises(ValueError, match="Missing required args"):
        SampleModule(args={"req_a": 1}, **metargs)

    # Valid args populate attributes and defaults
    mod = SampleModule(args={"req_a": 1, "req_b": 2, "opt_x": 99}, **metargs)
    assert mod.req_a == 1
    assert mod.req_b == 2
    assert mod.opt_x == 99
    assert mod.opt_y == "default_y"
    assert mod.get_identifier() == "SAMPLE"
    assert mod.get_args()["opt_y"] == "default_y"
    assert mod.get_run_args()["run_dir"] == metargs["run_dir"]


@pytest.mark.unit
def test_cemodule_directory_and_result_logging(metargs):
    mod = SampleModule(args={"req_a": 1, "req_b": 2}, **metargs)
    sub_dir = os.path.join(metargs["run_dir"], "analysis_out")
    created_path = mod.safe_create_dir(sub_dir)
    assert os.path.isdir(created_path)

    # Calling safe_create_dir again on an existing path creates a numbered rundir
    second_path = mod.safe_create_dir(sub_dir)
    assert os.path.isdir(second_path)
    assert second_path != created_path

    # Result logging
    mod.log_result("test_string", label=" status")
    mod.log_result(3.1415, label="pi")
    mod.log_result({"a": 1}, label="dict_res")
    assert os.path.isfile(mod.results_file)


@pytest.mark.unit
def test_config_check_duplicate_keys(tmp_path):
    valid_yaml = tmp_path / "valid.yaml"
    valid_yaml.write_text("a: 1\nb:\n  c: 2\n  d: 3\n")
    assert validate_duplicate_keys(str(valid_yaml)) is True

    dup_yaml = tmp_path / "dup.yaml"
    dup_yaml.write_text("a: 1\nb: 2\na: 3\n")
    with pytest.raises(ValueError, match="Duplicate keys"):
        validate_duplicate_keys(str(dup_yaml))


@pytest.mark.unit
def test_config_check_required_fields():
    assert validate_required_fields({"a": 1}, fields=[]) is True

    with pytest.raises(ValueError, match="Config is None"):
        validate_required_fields(None, fields=["a"])

    with pytest.raises(ValueError, match="Config is empty"):
        validate_required_fields({}, fields=["a"])

    missing = validate_required_fields({"a": 1}, fields=["a", "b", "c"])
    assert missing == ["b", "c"]
    assert validate_required_fields({"a": 1, "b": 2}, fields=["a", "b"]) is True


@pytest.mark.unit
def test_cemodule_save_npy_without_prior_results_dir(metargs):
    mod = SampleModule(args={"req_a": 1, "req_b": 2}, **metargs)
    mod.save_npy(np.array([1.0, 2.0, 3.0]), "arr_test")
    assert os.path.isfile(os.path.join(mod.results_dir, "arr_test.npy"))


@pytest.mark.unit
def test_cemodule_save_npy_after_create_results_dir(metargs):
    mod = SampleModule(args={"req_a": 1, "req_b": 2}, **metargs)
    mod.create_results_dir()
    mod.save_npy(np.array([1.0, 2.0, 3.0]), "arr_test")
    assert os.path.isfile(os.path.join(mod.results_dir, "arr_test.npy"))


@pytest.mark.unit
def test_common_transforms_lv2std():
    from collective_encoder.common.transforms import add_transformed, transform_lv2std

    vals = {"logvar_z": np.array([0.0, 2.0], dtype=np.float64)}
    out = add_transformed({"std_z": "lv2std:logvar_z"}, vals.copy())
    assert "std_z" in out
    np.testing.assert_allclose(out["std_z"], np.array([1.0, np.e]))

    # Missing or invalid arguments return None without crashing
    assert transform_lv2std([], vals) is None
    assert transform_lv2std(["missing_key"], vals) is None
    assert add_transformed(None, vals) is vals

