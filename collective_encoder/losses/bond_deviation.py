from typing import Any, Dict, Tuple

from ase.data import covalent_radii

import torch
import torch.nn.functional as F

from .base import CELossBase

EPSILON = 1e-7

class CELossBondDeviation(CELossBase):
    _IDENTIFIER = "CELossBondDeviation"
    _REQUIRED_ARGS = ['atomic_numbers', 'bond_indices']
    
    def __init__(self, 
                args: Dict[str, Any] = None, 
                **kwargs) -> None:
        super().__init__(self, args, **kwargs)
        
        cov_radii = [covalent_radii[el] for el in self.atomic_numbers]
        cov_radii = torch.tensor(cov_radii).float()
        cov_radii = cov_radii.unsqueeze(0)
        cd_t = cov_radii.transpose(0, 1)
        cov_mat = cov_radii.unsqueeze(0) + cd_t.unsqueeze(1)
        self.cov_mat = cov_mat.squeeze(1)

    def forward(self, 
                inp: torch.Tensor, 
                latent: torch.Tensor, 
                output: torch.Tensor, 
                labels: torch.Tensor, 
                meta: Dict[str, torch.Tensor],
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        bonded_indices = self.bond_indices
        coordinates = inp.view(inp.shape[0], -1, 3)
        n_atoms = coordinates.shape[-2]
        flattened_instances = coordinates.reshape(-1, n_atoms, 3)
        n_instances = flattened_instances.shape[0]
        flattened_coordinates = flattened_instances.reshape(-1, 3)

        for bond in bonded_indices:
            if bond[0] >= n_atoms or bond[1] >= n_atoms:
                self.raise_error(f"Invalid bond indices: {bond} for {n_atoms} atoms")

        mask1 = torch.zeros(len(bonded_indices), device=inp.device)
        mask2 = torch.zeros(len(bonded_indices), device=inp.device)
        cov_distances = torch.zeros(len(bonded_indices), device=inp.device)

        for ind, (i, j) in enumerate(bonded_indices):
            mask1[ind] = i
            mask2[ind] = j
            cov_distances[ind] = self.cov_mat[i, j]

        mask1 = mask1.repeat(n_instances)
        mask2 = mask2.repeat(n_instances)
        cov_distances = cov_distances.repeat(n_instances)
        set1 = flattened_coordinates[mask1.long()]
        set2 = flattened_coordinates[mask2.long()]

        dist = F.pairwise_distance(set1, set2)
        deviation = (dist - cov_distances) ** 2
        return deviation.mean(), {}
