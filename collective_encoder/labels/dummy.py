import MDAnalysis as mda
from typing import Dict, List, Union

from .base import BaseLabeler

class DummyLabeler(BaseLabeler):
    def __init__(self, 
                 universe: mda.Universe,
                 args: Dict[str, Union[str, List[str]]]
                 ) -> None:
        pass
    def get_names(self) -> List[str]:
        return ['None']

    def compute(self) -> List[float]:
        return [0.0]
