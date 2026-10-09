
from typing import Dict, List, Tuple, Union, Union

import numpy as np
import ase

import torch

from embeddings.resolver import get_encdec
from .bondgraph import BondGraphDataset

class BondGraphLatentDataset(BondGraphDataset):
    """
    BondGraphDataset variant that computes and stores latent graph embeddings.

    Loads a pre-trained encoder model, evaluates all structures to get their
    latent representations, and serves those representations directly, replacing
    the raw graph data.

    Parameters
    ----------
    structures : list of ase.Atoms
        A list of molecular structures.
    labels : list of list of float
        A list of target labels corresponding to each structure.
    dataset_args : dict, optional
        Configuration dictionary. Must include:
        - ``encoder_name`` (str): Name of the encoder model to instantiate.
        - ``encoder_ckpt`` (str): Path to the pre-trained checkpoint.
        In addition to all `BondGraphDataset` required args.
    kwargs : dict
        Additional keyword arguments (e.g. ``tag``).
    """
    _IDENTIFIER = "BONDGRAPH_LATENT"
    _REQUIRED_ARGS = BondGraphDataset._REQUIRED_ARGS + ["encoder_name", "encoder_ckpt"]
    
    def __init__(
        self,
        structures: List[ase.Atoms],
        labels: List[List[float]],
        dataset_args: Dict[str, Union[float, int, str]] = None,
        **kwargs,
    ):
        super().__init__(structures=structures, 
                         labels=labels, 
                         dataset_args=dataset_args,
                         **kwargs)
        
        self.encoder = get_encdec(self.encoder_name).load_from_checkpoint(
            self.encoder_ckpt, strict=False, map_location=torch.device('cpu')
        )
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Encode the graphs
        self.log_msg("Encoding graphs with provided encoder...")
        with torch.no_grad():
            self.encoded = [self.encoder._encode(self[i])[0].flatten() for i in range(len(self))]
        self.log_msg("Encoding complete.")
        self.encoder = None  # Free up memory
        

    def __len__(self):
        return len(self.structures)

    def get(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, "encoded"):
            return super().get(index)  # Return Graph if not encoded
        return self.encoded[index], self.labels[index]
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all encoded latent vectors and labels as NumPy arrays.

        Returns
        -------
        tuple
            Tuple of (latents_array, labels_array).
        """
        return np.array([d.numpy() for d in self.encoded]), np.array([l.numpy() for l in self.labels])
    
    def get_norm_data(self) -> np.ndarray:
        """
        Return array of data to be normalized.

        Returns
        -------
        numpy.ndarray
            Stacked latent vectors.
        """
        return np.vstack(self.encoded)
    
    def get_datapoint_shape(self) -> tuple:
        """
        Return the shape of a single data point's features.

        Returns
        -------
        tuple
            The shape of the extracted latent features.
        """
        return tuple(self.encoded[0].shape)
