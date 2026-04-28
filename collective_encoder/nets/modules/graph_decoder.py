from typing import List, Optional

import numpy as np

import torch
from torch import nn, Tensor

from .mp_modules import ScalarFeatureEmbedding, AttentionMP

class BondGraphDecoder(nn.Module):
    """Bond-based message passing GNN decoder with multi-head attention.

    Steps:
      1. During initialization, a precomputed graph template is provided
      2. The second feature of the edge attributes first embedded using radial basis functions
      3. Then all the node and edge features are embedded using a individial MLPs like the encoder
      4. L attention message passing layers (each with residual, BatchNorm, ELU).
      5. The steps uptill now are only done once during initialization
      6. During forward pass, we get the latent vector for the batch only.
      7. We concatenate this latent vector and the node features from selected nodes after 
         the message passing layers to get 4 vectors of interest
      8. The first vector denotes the bond distance and is calculated between any two 
         adjacent nodes. We concatenate the final node features of the two nodes and the 
         latent vector to get the input
      9. The second vector denotes the angle and is calculated between any three adjacent nodes
      10. The third vector denotes the cosine of the dihedral angle and is calculated between any four 
          adjacent nodes
      11. The fourth vector denotes the sine of the dihedral angle and is calculated between any four 
          adjacent nodes
      12. Each of these vectors is then passed through a two-layer MLP with ELU activations 
          and dropout to map to a single scalar output

    Args:
        template_data: PyG Data object (atom-level or bond-level template graph)
        latent_dim: Output latent embedding size.
        hidden_dim: Hidden embedding size.
        num_layers: Number of message passing layers (L).
        heads: Attention heads.
        dropout: Dropout applied to attention coefficients.
        rbf_dim: Number of radial basis functions for distance embedding.
        rbf_min: Minimum distance for RBFs.
        rbf_max: Maximum distance for RBFs.
        rbf_gamma: Width parameter for RBFs.
        precompute: Whether to precompute the structural embedding of the template graph.
    """
    def __init__(
        self,
        template_data,  # PyG Data object (atom-level or bond-level template graph)
        latent_dim: int,
        label_indices: Optional[tuple],  # (bond_index, angle_index, torsion_index)
        node_embed_dim: int = 10,
        edge_embed_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 4,
        heads: int = 4,
        dropout_mp: float = 0.0,
        dropout_mlp: float = 0.0,
        final_mlp_layers: int = 3,
        rbf_dim: int = 16,
        rbf_min: float = 0.0,
        rbf_max: float = 4.0,
        rbf_gamma: float = 10.0,
        precompute: bool = True,
        out_labels: List[str] = ['bond_dist', 'angle', 'dihedral_cos', 'dihedral_sin'],
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.bond_index = np.array(label_indices[0])
        self.angle_index = np.array(label_indices[1])
        self.torsion_index = np.array(label_indices[2])
        self.node_embed_dim = node_embed_dim
        self.edge_embed_dim = edge_embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout_mp = dropout_mp
        self.dropout_mlp = dropout_mlp
        self.final_mlp_layers = final_mlp_layers
        self.rbf_dim = rbf_dim
        self.register_buffer('rbf_centers', torch.linspace(rbf_min, rbf_max, rbf_dim))
        self.rbf_gamma = rbf_gamma
        self.precompute = precompute
        self.out_labels = out_labels

        # ---------- Template graph processing ----------
        # Expect edge_attr: [bond_type_one_hot(5), bond_length]
        assert hasattr(template_data, 'edge_attr'), "template_data must have edge_attr"
        edge_attr_raw: Tensor = template_data.edge_attr.clone().detach()
        assert edge_attr_raw.size(1) >= 6, "edge_attr must contain 5 bond-type one-hot + length"
        # Ensure consistent dtype (float32 default) for features
        desired_dtype = torch.get_default_dtype()
        bond_types = edge_attr_raw[:, :5].to(desired_dtype)
        bond_len = edge_attr_raw[:, 5].to(desired_dtype)
        # RBF embedding of bond lengths
        bond_rbf = self._rbf_embed(bond_len)
        edge_features = torch.cat([bond_types, bond_rbf], dim=-1)  # (E, 5+rbf_dim)

        # Per-feature MLP embedding for nodes and edges
        self.node_embed = ScalarFeatureEmbedding(in_dim=template_data.x.size(1), out_dim=node_embed_dim)
        self.edge_embed = ScalarFeatureEmbedding(in_dim=edge_features.size(1), out_dim=edge_embed_dim)

        self.node_dim = node_embed_dim #* template_data.x.size(1)
        self.edge_dim = edge_embed_dim #* edge_features.size(1)
        # Attention message passing layers (edge_dim = hidden_dim after embedding)
        self.mp_layers = nn.ModuleList([
            AttentionMP(node_feat_dim=self.node_dim if i == 0 else hidden_dim, 
                        edge_feat_dim=self.edge_dim, 
                        hidden_dim=hidden_dim, 
                        heads=heads, 
                        dropout=dropout_mp)
            for i in range(num_layers)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.elu = nn.ELU()

        # Precompute structural embedding (optional)
        if precompute:
            with torch.no_grad():
                h = self.node_embed(template_data.x)
                e = self.edge_embed(edge_features)
                # print("Precomputing template node representations with shape:", h.shape)
                # print("Precomputing template edge representations with shape:", e.shape)
                # print(self.node_dim, self.edge_dim)
                # print(template_data.x.size(1), edge_features.size(1))
                # print(template_data.x.size(0), edge_features.size(0))
                # exit()
                for mp, bn in zip(self.mp_layers, self.bns):
                    h = mp(h, template_data.edge_index, e)
                    h = bn(h)
                    h = self.elu(h)
                self.register_buffer('template_node_repr', h)
        else:
            self.template_node_repr = None  # type: ignore
        # Always store processed edge features for reuse
        self.register_buffer('template_edge_features', edge_features)
        self.register_buffer('template_edge_index', template_data.edge_index.clone())

        # ---------- Build combinatorial sets (bonds, angles, dihedrals) ----------
        # self._build_topology_sets(template_data)

        # ---------- Prediction heads ----------
        bond_in = 2 * hidden_dim + latent_dim
        angle_in = 3 * hidden_dim + latent_dim
        dihedral_in = 4 * hidden_dim + latent_dim

        def head(in_dim, n_layers=2):
            layers = []
            for _ in range(n_layers):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ELU())
                layers.append(nn.Dropout(dropout_mlp))
                in_dim = hidden_dim
            layers.append(nn.Linear(hidden_dim, 1))
            return nn.Sequential(*layers)

        self.bond_head = head(bond_in, final_mlp_layers)
        self.angle_head = head(angle_in, final_mlp_layers)
        self.dih_cos_head = head(dihedral_in, final_mlp_layers)
        self.dih_sin_head = head(dihedral_in, final_mlp_layers)

    # ------------------------------------------------------------------
    def _rbf_embed(self, distances: Tensor) -> Tensor:
        # distances: (E,)
        diff = distances.unsqueeze(-1) - self.rbf_centers  # (E, rbf_dim)
        return torch.exp(-self.rbf_gamma * diff * diff)

    # ------------------------------------------------------------------
    def _compute_template_repr(self, device):
        # Compute (or retrieve) node representations for template graph
        if self.template_node_repr is not None:
            return self.template_node_repr.to(device)
        # If not precomputed, run embedding graph dynamically (keeps gradients)
        x = self.node_embed(self.template_edge_index.new_tensor([]))  # placeholder (won't be used)
        raise RuntimeError("Dynamic (non-precomputed) template representation not implemented.")

    # ------------------------------------------------------------------
    def forward(self, latent: Tensor):
        """Decode latent vector(s) into structural predictions.

        Args:
            latent: (B, latent_dim) latent embeddings.

        Returns dict with keys: bond_dist, angle, dihedral_cos, dihedral_sin
        Shapes: (B, num_bonds / num_angles / num_dihedrals)
        """
        B = latent.size(0)
        H = self.hidden_dim
        device = latent.device
        template_h = self._compute_template_repr(device)  # (N, H)

        # Bonds
        bond_idx = self.bond_index
        angle_idx = self.angle_index
        dih_idx = self.torsion_index

        preds = {}
        if bond_idx.size > 0:
            h_bi = template_h[bond_idx[:,0]]
            h_bj = template_h[bond_idx[:,1]]
            bond_feat = torch.cat([h_bi, h_bj], dim=-1)  # (Nb, 2H)
            bond_feat = bond_feat.unsqueeze(0).expand(B, -1, -1)
            latent_exp = latent.unsqueeze(1).expand(-1, bond_feat.size(1), -1)
            bond_in = torch.cat([bond_feat, latent_exp], dim=-1)
            bond_out = self.bond_head(bond_in.view(-1, bond_in.size(-1))).view(B, -1)
        else:
            bond_out = latent.new_empty((B, 0))
        preds[self.out_labels[0]] = bond_out

        # Angles
        if angle_idx.size > 0:
            h_i = template_h[angle_idx[:,0]]
            h_j = template_h[angle_idx[:,1]]
            h_k = template_h[angle_idx[:,2]]
            ang_feat = torch.cat([h_i, h_j, h_k], dim=-1)  # (Na, 3H)
            ang_feat = ang_feat.unsqueeze(0).expand(B, -1, -1)
            latent_exp = latent.unsqueeze(1).expand(-1, ang_feat.size(1), -1)
            ang_in = torch.cat([ang_feat, latent_exp], dim=-1)
            angle_out = self.angle_head(ang_in.view(-1, ang_in.size(-1))).view(B, -1)
        else:
            angle_out = latent.new_empty((B, 0))
        preds[self.out_labels[1]] = angle_out

        # Dihedrals
        if dih_idx.size > 0:
            h_i = template_h[dih_idx[:,0]]
            h_j = template_h[dih_idx[:,1]]
            h_k = template_h[dih_idx[:,2]]
            h_l = template_h[dih_idx[:,3]]
            dih_feat = torch.cat([h_i, h_j, h_k, h_l], dim=-1)  # (Nd, 4H)
            dih_feat = dih_feat.unsqueeze(0).expand(B, -1, -1)
            latent_exp = latent.unsqueeze(1).expand(-1, dih_feat.size(1), -1)
            dih_in = torch.cat([dih_feat, latent_exp], dim=-1)
            cos_out = self.dih_cos_head(dih_in.view(-1, dih_in.size(-1))).view(B, -1)
            sin_out = self.dih_sin_head(dih_in.view(-1, dih_in.size(-1))).view(B, -1)
        else:
            cos_out = latent.new_empty((B, 0))
            sin_out = latent.new_empty((B, 0))
        preds[self.out_labels[2]] = cos_out
        preds[self.out_labels[3]] = sin_out

        return preds
