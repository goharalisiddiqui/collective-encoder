import pytest
import numpy as np
from collective_encoder.datalabelers.dummy import DummyLabeler

def test_dummy_labeler():
    labeler = DummyLabeler()

    assert labeler.get_label_names() == ["dummy"]

    # Test frame computation
    result = labeler.compute()
    assert result == [0.0]

    # Test batch computation
    indices = [0, 1, 2]
    result = labeler.compute(indices)
    assert result.shape == (3, 1)
