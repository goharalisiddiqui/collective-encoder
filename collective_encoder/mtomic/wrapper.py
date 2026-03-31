from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System, NeighborListOptions

class MetatomicCV(torch.nn.Module):
    def __init__(self, 
                 dataprocessor: torch.nn.Module,
                 model: torch.nn.Module):
        super().__init__()
        self.dataprocessor = dataprocessor
        self.model = model.eval()
        if hasattr(model, 'metaD'):
            self.model.metaD = True
        self.num_outputs = 2

        # self._nl_options = NeighborListOptions(
        #     cutoff=5.0, full_list=True, strict=True
        # )
    
    # def requested_neighbor_lists(self) -> List[NeighborListOptions]:
    #     # request a neighbor list to be computed and stored in the system passed to
    #     # `forward`.
    #     return [self._nl_options]


    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if list(outputs.keys()) != ["features"]:
            raise ValueError(
                "this model can only compute 'cv', but `outputs` contains other "
                f"keys: {', '.join(outputs.keys())}"
            )

        # we don't want to worry about selected_atoms yet
        # if selected_atoms is not None:
        #     raise NotImplementedError("selected_atoms is not implemented")

        if outputs["features"].per_atom:
            raise NotImplementedError("per atom cv is not implemented")
        
        if len(systems[0].positions) == 0:
            # PLUMED will first call the model with 0 atoms to get the size of the
            # output, so we need to handle this case first
            block = TensorBlock(
                torch.ones((0, self.num_outputs), dtype=torch.float64) * 42.0,
                samples=Labels("structure", torch.zeros((0, 1), dtype=torch.int32)),
                components=[],
                properties=Labels("cv", torch.arange(self.num_outputs).reshape(self.num_outputs, 1)),
            )
            return {"features": TensorMap(keys=Labels("_", torch.tensor([[0]])), blocks=[block])}
        
        model_input = self.dataprocessor(systems, outputs, selected_atoms)
        if len(model_input.shape) == 1:
            model_input = model_input.view(1, -1) # ensure batch dimension
        cv = self.model(model_input)

        block = TensorBlock(
            values=cv,
            samples=Labels("system", torch.arange(len(systems)).reshape(-1, 1)),
            components=[],
            properties=Labels("cv", torch.arange(self.num_outputs).reshape(self.num_outputs, 1)),
        )
        return {
            "features": TensorMap(keys=Labels("_", torch.tensor([[0]])), blocks=[block])
        }
