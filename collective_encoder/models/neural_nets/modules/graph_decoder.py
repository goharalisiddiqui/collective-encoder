from typing import Any, Dict, List, Optional

import numpy as np

import torch
from torch import nn, Tensor

from collective_encoder import TENSOR_PRECISION
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
        dtype=TENSOR_PRECISION,
        species_templates: Optional[Dict[int, Any]] = None,
        species_label_indices: Optional[Dict[int, tuple]] = None,
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
        self.dtype = dtype

        # ---------- Template graph processing ----------
        edge_features = self._extract_template_edge_features(template_data)

        # Per-feature MLP embedding for nodes and edges
        self.node_embed = ScalarFeatureEmbedding(in_dim=template_data.x.size(1), 
                                                 out_dim=node_embed_dim, 
                                                 dtype=dtype)
        self.edge_embed = ScalarFeatureEmbedding(in_dim=edge_features.size(1), 
                                                 out_dim=edge_embed_dim, 
                                                 dtype=dtype)

        self.node_dim = node_embed_dim #* template_data.x.size(1)
        self.edge_dim = edge_embed_dim #* edge_features.size(1)
        # Attention message passing layers (edge_dim = hidden_dim after embedding)
        self.mp_layers = nn.ModuleList([
            AttentionMP(node_feat_dim=self.node_dim if i == 0 else hidden_dim, 
                        edge_feat_dim=self.edge_dim, 
                        hidden_dim=hidden_dim, 
                        heads=heads, 
                        dropout=dropout_mp,
                        dtype=dtype
                        )
            for i in range(num_layers)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim, dtype=dtype) for _ in range(num_layers)])
        self.elu = nn.ELU()

        # Precompute structural embedding (optional)
        if precompute:
            h = self._encode_template_graph(template_data, edge_features)
            self.register_buffer('template_node_repr', h)
        else:
            self.template_node_repr = None  # type: ignore
        # Always store processed edge features for reuse
        self.register_buffer('template_edge_features', edge_features)
        self.register_buffer('template_edge_index', template_data.edge_index.clone())

        # ---------- Multi-species template setup ----------
        self.species_bond_index: Dict[int, np.ndarray] = {}
        self.species_angle_index: Dict[int, np.ndarray] = {}
        self.species_torsion_index: Dict[int, np.ndarray] = {}
        self.is_multispecies = bool(
            species_templates is not None
            and species_label_indices is not None
            and len(species_templates) > 1
        )
        if species_templates and species_label_indices:
            for sid, s_tmpl in species_templates.items():
                sid_int = int(sid)
                s_lbl = species_label_indices[sid]
                self.species_bond_index[sid_int] = np.array(s_lbl[0])
                self.species_angle_index[sid_int] = np.array(s_lbl[1])
                self.species_torsion_index[sid_int] = np.array(s_lbl[2])
                s_edge_feats = self._extract_template_edge_features(s_tmpl)
                s_h = self._encode_template_graph(s_tmpl, s_edge_feats)
                self.register_buffer(f"template_node_repr_sp_{sid_int}", s_h)

            self.max_bonds = max(len(v) for v in self.species_bond_index.values())
            self.max_angles = max(len(v) for v in self.species_angle_index.values())
            self.max_torsions = max(len(v) for v in self.species_torsion_index.values())
            self._largest_species_id = max(
                self.species_bond_index.keys(),
                key=lambda k: len(self.species_bond_index[k]),
            )
        else:
            self.max_bonds = len(self.bond_index)
            self.max_angles = len(self.angle_index)
            self.max_torsions = len(self.torsion_index)
            self._largest_species_id = None

        # ---------- Prediction heads ----------
        bond_in = 2 * hidden_dim + latent_dim
        angle_in = 3 * hidden_dim + latent_dim
        dihedral_in = 4 * hidden_dim + latent_dim

        def head(in_dim, n_layers=2):
            layers = []
            for _ in range(n_layers):
                layers.append(nn.Linear(in_dim, hidden_dim, dtype=dtype))
                layers.append(nn.ELU())
                layers.append(nn.Dropout(dropout_mlp))
                in_dim = hidden_dim
            layers.append(nn.Linear(hidden_dim, 1, dtype=dtype))
            return nn.Sequential(*layers)

        self.bond_head = head(bond_in, final_mlp_layers)
        self.angle_head = head(angle_in, final_mlp_layers)
        self.dih_cos_head = head(dihedral_in, final_mlp_layers)
        self.dih_sin_head = head(dihedral_in, final_mlp_layers)

    def _extract_template_edge_features(self, template_data) -> Tensor:
        """Extract 5 one-hot bond-distance hop features + RBF distance features."""
        assert hasattr(template_data, 'edge_attr'), "template_data must have edge_attr"
        edge_attr_raw: Tensor = template_data.edge_attr.clone().detach()
        assert edge_attr_raw.size(1) >= 6, "edge_attr must contain 5 bond-type one-hot + length"
        bond_types = edge_attr_raw[:, :5].to(self.dtype)
        bond_len = edge_attr_raw[:, 5].to(self.dtype)
        bond_rbf = self._rbf_embed(bond_len)
        return torch.cat([bond_types, bond_rbf], dim=-1)

    def _encode_template_graph(self, template_data, edge_features: Tensor) -> Tensor:
        """Compute static atom node embeddings for a template graph."""
        with torch.no_grad():
            h = self.node_embed(template_data.x)
            e = self.edge_embed(edge_features)
            for mp, bn in zip(self.mp_layers, self.bns):
                h = mp(h, template_data.edge_index, e)
                h = bn(h)
                h = self.elu(h)
        return h

    # ------------------------------------------------------------------
    def _rbf_embed(self, distances: Tensor) -> Tensor:
        # distances: (E,)
        diff = distances.unsqueeze(-1) - self.rbf_centers  # (E, rbf_dim)
        return torch.exp(-self.rbf_gamma * diff * diff)

    # ------------------------------------------------------------------
    def _compute_template_repr(self, device, species_id: Optional[int] = None):
        if species_id is not None and hasattr(self, f"template_node_repr_sp_{int(species_id)}"):
            return getattr(self, f"template_node_repr_sp_{int(species_id)}").to(device)
        if self.template_node_repr is not None:
            return self.template_node_repr.to(device)
        x = self.node_embed(self.template_edge_index.new_tensor([]))
        raise RuntimeError("Dynamic (non-precomputed) template representation not implemented.")

    def _decode_with_template(
        self,
        latent: Tensor,
        template_h: Tensor,
        bond_idx: np.ndarray,
        angle_idx: np.ndarray,
        dih_idx: np.ndarray,
    ) -> Dict[str, Tensor]:
        """Decode a homogeneous batch of latent vectors using a single template."""
        B = latent.size(0)
        preds: Dict[str, Tensor] = {}

        if bond_idx.size > 0:
            h_bi = template_h[bond_idx[:, 0]]
            h_bj = template_h[bond_idx[:, 1]]
            bond_feat = torch.cat([h_bi, h_bj], dim=-1).unsqueeze(0).expand(B, -1, -1)
            latent_exp = latent.unsqueeze(1).expand(-1, bond_feat.size(1), -1)
            bond_in = torch.cat([bond_feat, latent_exp], dim=-1)
            bond_out = self.bond_head(bond_in.view(-1, bond_in.size(-1))).view(B, -1)
        else:
            bond_out = latent.new_empty((B, 0))
        preds[self.out_labels[0]] = bond_out

        if angle_idx.size > 0:
            h_i = template_h[angle_idx[:, 0]]
            h_j = template_h[angle_idx[:, 1]]
            h_k = template_h[angle_idx[:, 2]]
            ang_feat = torch.cat([h_i, h_j, h_k], dim=-1).unsqueeze(0).expand(B, -1, -1)
            latent_exp = latent.unsqueeze(1).expand(-1, ang_feat.size(1), -1)
            ang_in = torch.cat([ang_feat, latent_exp], dim=-1)
            angle_out = self.angle_head(ang_in.view(-1, ang_in.size(-1))).view(B, -1)
        else:
            angle_out = latent.new_empty((B, 0))
        preds[self.out_labels[1]] = angle_out

        if dih_idx.size > 0:
            h_i = template_h[dih_idx[:, 0]]
            h_j = template_h[dih_idx[:, 1]]
            h_k = template_h[dih_idx[:, 2]]
            h_l = template_h[dih_idx[:, 3]]
            dih_feat = torch.cat([h_i, h_j, h_k, h_l], dim=-1).unsqueeze(0).expand(B, -1, -1)
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

    # ------------------------------------------------------------------
    def forward(
        self,
        latent: Tensor,
        species_id: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Decode latent vector(s) into structural predictions.

        Parameters
        ----------
        latent : torch.Tensor
            Latent embeddings of shape ``(B, latent_dim)``.
        species_id : torch.Tensor, optional
            Per-sample species identifiers of shape ``(B,)`` or ``(B, 1)``.
            When provided on a multi-species decoder, decodes each sample using its
            species template and pads outputs to ``(B, max_dim)`` with ``NaN``.

        Returns
        -------
        dict of str to torch.Tensor
            Predicted ``bond_dist``, ``angle``, ``dihedral_cos``, and ``dihedral_sin``.
        """
        device = latent.device
        B = latent.size(0)

        if not self.is_multispecies:
            template_h = self._compute_template_repr(device)
            return self._decode_with_template(
                latent,
                template_h,
                self.bond_index,
                self.angle_index,
                self.torsion_index,
            )

        if species_id is None:
            sid = self._largest_species_id
            template_h = self._compute_template_repr(device, species_id=sid)
            return self._decode_with_template(
                latent,
                template_h,
                self.species_bond_index[sid],
                self.species_angle_index[sid],
                self.species_torsion_index[sid],
            )

        sids = species_id.view(-1).to(dtype=torch.long, device=device)
        if sids.numel() == 1 and B > 1:
            sids = sids.expand(B)

        preds = {
            self.out_labels[0]: torch.full((B, self.max_bonds), float("nan"), dtype=latent.dtype, device=device),
            self.out_labels[1]: torch.full((B, self.max_angles), float("nan"), dtype=latent.dtype, device=device),
            self.out_labels[2]: torch.full((B, self.max_torsions), float("nan"), dtype=latent.dtype, device=device),
            self.out_labels[3]: torch.full((B, self.max_torsions), float("nan"), dtype=latent.dtype, device=device),
        }

        for uniq_sid in torch.unique(sids):
            sid_int = int(uniq_sid.item())
            mask = sids == uniq_sid
            if sid_int in self.species_bond_index:
                t_h = self._compute_template_repr(device, species_id=sid_int)
                b_idx = self.species_bond_index[sid_int]
                a_idx = self.species_angle_index[sid_int]
                d_idx = self.species_torsion_index[sid_int]
            else:
                t_h = self._compute_template_repr(device)
                b_idx, a_idx, d_idx = self.bond_index, self.angle_index, self.torsion_index

            sub_preds = self._decode_with_template(latent[mask], t_h, b_idx, a_idx, d_idx)
            for k, v in sub_preds.items():
                if v.size(1) > 0:
                    preds[k][mask, : v.size(1)] = v

        return preds
