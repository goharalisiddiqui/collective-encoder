import glob
from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    System,
)

class MetatomicCV(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        pass

        # outputs = self.model()