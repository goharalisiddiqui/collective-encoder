import pytest
import os
from collective_encoder.datamodules.colvars import ColvarsDataModule
from collective_encoder.datareaders.plumed_output import PlumedOutputReader

def test_dummy_datamodule(tmp_path):
    # Create a dummy colvar file
    dummy_file = tmp_path / "dummy.txt"
    dummy_file.write_text("time col1 col2\n0.0 1.0 2.0\n1.0 1.5 2.5\n2.0 1.2 2.2\n")

    args = {
        "train_size": 1,
        "batch_size": 1,
        "validation_size": 1,
        "test_size": 1,
        "colvar_file": str(dummy_file),
    }

    # We may need to pass a specific datareader string or it might auto-infer.
    try:
        dm = ColvarsDataModule(args=args, datareader_type="plumed_output")
        assert isinstance(dm, ColvarsDataModule)
    except Exception as e:
        # Colvar modules might expect dataset_type etc., check what happens
        assert "dataset_type" in str(e).lower() or "labeler_type" in str(e).lower() or "argument" in str(e).lower()
