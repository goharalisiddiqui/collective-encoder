from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

class MetatomicCV(torch.nn.Module):
    def __init__(self, 
                 dataprocessor: torch.nn.Module,
                 model: torch.nn.Module):
        super().__init__()
        self.dataprocessor = dataprocessor
        self.model = model
        if hasattr(model, 'metaD'):
            self.model.metaD = True
        self.num_outputs = 2

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if list(outputs.keys()) != ["cv"]:
            raise ValueError(
                "this model can only compute 'cv', but `outputs` contains other "
                f"keys: {', '.join(outputs.keys())}"
            )

        # we don't want to worry about selected_atoms yet
        if selected_atoms is not None:
            raise NotImplementedError("selected_atoms is not implemented")

        if outputs["cv"].per_atom:
            raise NotImplementedError("per atom cv is not implemented")
        
        if len(systems[0].positions) == 0:
            # PLUMED will first call the model with 0 atoms to get the size of the
            # output, so we need to handle this case first
            keys = Labels("_", torch.tensor([[0]]))
            block = TensorBlock(
                torch.zeros((0, self.num_outputs), dtype=torch.float64),
                samples=Labels("structure", torch.zeros((0, 1), dtype=torch.int32)),
                components=[],
                properties=Labels([f"cv-{a}" for a in range(self.num_outputs)], 
                                  torch.arange(self.num_outputs).reshape(-1, self.num_outputs)),
            )
            return {"cv": TensorMap(keys, [block])}

        model_input = self.dataprocessor(systems, outputs, selected_atoms)
        cv = self.model(model_input)
    
        # add metadata to the output
        block = TensorBlock(
            values=cv,
            samples=Labels("system", torch.arange(len(systems)).reshape(-1, 1)),
            components=[],
            properties=Labels([f"cv-{a}" for a in range(self.num_outputs)], 
                              torch.arange(self.num_outputs).reshape(-1, self.num_outputs)),
        )
        return {
            "cv": TensorMap(keys=Labels("_", torch.tensor([[0]])), blocks=[block])
        }
