import pytest
import numpy as np
from ase.build import bulk
from collective_encoder.datareaders.base import BaseDataReader

class DummyReader(BaseDataReader):
    _IDENTIFIER = "dummy_reader"

    def __init__(self, **kwargs):
        super().__init__(args={}, **kwargs)
        self.atoms = bulk('Cu', 'fcc', a=3.6)
        self.label_list = ["L1", "L2"]

    def get_total_frames(self):
        return 10

def test_dummy_reader():
    reader = DummyReader()
    assert reader.get_total_frames() == 10
    assert reader.get_identifier() == "dummy_reader"
    labels = reader.get_label_names()
    assert labels == ["L1", "L2"]
