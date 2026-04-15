
from typing import Dict, List, Tuple, Any, Union, Union

import numpy as np
import ase

import torch

from embeddings.resolver import get_encdec
from .bondgraph import BondGraphDataset

class BondGraphLatentDataset(BondGraphDataset):
    _IDENTIFIER = "GRAPH_LATENT"
    _REQUIRED_ARGS = ["encoder_name", "encoder_ckpt"]
    
    def __init__(
        self,
        structures: List[ase.Atoms],
        labels: List[List[float]],
        dataset_args: Dict[str, Union[float, int, str]] = None,
        **kwargs,
    ):
        encoder_name = dataset_args.pop("encoder_name")
        encoder_ckpt = dataset_args.pop("encoder_ckpt")
        self.encoder = get_encdec(encoder_name).load_from_checkpoint(
            encoder_ckpt, strict=False, map_location=torch.device('cpu')
        )
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        super().__init__(structures=structures, 
                         labels=labels, 
                         dataset_args=dataset_args,
                         **kwargs)
        
        # Encode the graphs
        self.log_msg("Encoding graphs with provided encoder...")
        with torch.no_grad():
            self.encoded = [self.encoder.encode(self[i]).flatten() for i in range(len(self))]
        self.log_msg("Encoding complete.")
        self.encoder = None  # Free up memory
        

    def __len__(self):
        return len(self.structures)

    def get(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, "encoded"):
            return super().get(index)  # Return Graph if not encoded
        return self.encoded[index], self.labels[index]
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([d.numpy() for d in self.encoded]), np.array([l.numpy() for l in self.labels])
    
    def get_norm_data(self) -> np.ndarray:
        return np.vstack(self.encoded)
    
    def get_datapoint_shape(self) -> tuple:
        return tuple(self.encoded[0].shape)
