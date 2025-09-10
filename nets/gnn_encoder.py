import argparse
import math
from typing import List, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, Set2Set
from torch_geometric.utils import softmax
import pytorch_lightning as pl

class ScalarFeatureEmbedding(nn.Module):
    """Applies one independent MLP per scalar feature dimension and sums outputs.

    Given input x of shape (N, F), we create F small MLPs each processing x[:, f:f+1].
    Each MLP outputs (N, hidden_dim); final embedding h is the (optionally scaled) sum.
    """
    def __init__(self, in_features: int, hidden_dim: int, mlp_hidden: Optional[int] = None, activation=nn.ELU()):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        mlp_hidden = mlp_hidden or hidden_dim
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, mlp_hidden),
                activation,
                nn.Linear(mlp_hidden, hidden_dim),
            )
            for _ in range(in_features)
        ])

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, F)
        outs = []
        for f, mlp in enumerate(self.mlps):
            outs.append(mlp(x[:, f:f+1]))
        h = torch.stack(outs, dim=0).sum(dim=0)  # (N, hidden_dim)
        h = h / math.sqrt(self.in_features)  # scale
        return h


class AttentionMP(MessagePassing):
    """Multi-head dot-product attention message passing layer with edge features.

    Attention score per head: a_ij^h = (q_i^h · k_j^h)/sqrt(d) + b_e^h
    where b_e^h is a learned scalar bias from transformed edge features.
    Message: m_ij^h = a_ij^h * (v_j^h + e_msg_ij^h)
    Aggregation: sum over j -> i
    Output: residual + linear projection + optional norm & activation handled externally.
    """
    def __init__(self, hidden_dim: int, heads: int = 4, edge_dim: int = 3, dropout: float = 0.0):
        super().__init__(aggr='add', node_dim=0)
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.d_head = hidden_dim // heads
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Edge encoders
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_attn = nn.Linear(hidden_dim, heads, bias=False)
        self.edge_msg = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor]) -> Tensor:
        if edge_index.numel() == 0:
            return x  # no edges
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor, index: Tensor) -> Tensor:
        # x_i, x_j: (E, hidden_dim)
        q = self.q_proj(x_i).view(-1, self.heads, self.d_head)
        k = self.k_proj(x_j).view(-1, self.heads, self.d_head)
        v = self.v_proj(x_j).view(-1, self.heads, self.d_head)

        e = self.edge_mlp(edge_attr)  # (E, hidden_dim)
        e_attn = self.edge_attn(e)  # (E, heads)
        e_msg = self.edge_msg(e).view(-1, self.heads, self.d_head)  # (E, heads, d_head)

        logits = (q * k).sum(dim=-1) / math.sqrt(self.d_head)  # (E, heads)
        logits = logits + e_attn  # add edge bias

        alpha = softmax(logits, index)  # softmax over incoming edges per target node
        alpha = self.attn_drop(alpha)
        alpha = alpha.unsqueeze(-1)  # (E, heads, 1)

        msg = alpha * (v + e_msg)  # (E, heads, d_head)
        return msg

    def aggregate(self, inputs: Tensor, index: Tensor, ptr=None, dim_size=None):  # type: ignore
        # inputs: (E, heads, d_head)
        out = torch.zeros(dim_size, self.heads, self.d_head, device=inputs.device, dtype=inputs.dtype)
        out.index_add_(0, index, inputs)
        return out

    def update(self, aggr_out: Tensor) -> Tensor:
        # aggr_out: (N, heads, d_head)
        out = aggr_out.view(-1, self.hidden_dim)
        out = self.out_proj(out)
        return out


def bgne_parse_args():
    desc = "GRAPH-ENCODER Arguments"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--in_features', type=int, default=3, help='Number of scalar node features (default 3 for bond nodes)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden embedding size')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of message passing layers (L)')
    parser.add_argument('--heads', type=int, default=4, help='Attention heads')
    parser.add_argument('--edge_dim', type=int, default=3, help='Edge feature dimension (default 3: angle one-hot + value)')
    parser.add_argument('--set2set_steps', type=int, default=3, help='T processing steps for Set2Set')
    parser.add_argument('--latent_dim', type=int, default=256, help='Output latent embedding size')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout applied to attention coefficients')


    args, _ = parser.parse_known_args()

    return args


BGNE_args = bgne_parse_args

class BondGraphNetEncoder(nn.Module):
    """Bond-based message passing GNN Encoder with multi-head attention and Set2Set pooling.

    Steps:
      1. Per-scalar feature MLP embeddings summed to initial node embedding h.
      2. L attention message passing layers (each with residual, BatchNorm, ELU).
      3. Set2Set pooling over nodes for T processing steps.
      4. Final linear layer maps pooled embedding to latent_dim.

    Args:
        in_features: Number of scalar node features (default 3 for bond nodes).
        hidden_dim: Hidden embedding size.
        num_layers: Number of message passing layers (L).
        heads: Attention heads.
        edge_dim: Edge feature dimension (default 3: angle one-hot + value).
        set2set_steps: T processing steps for Set2Set.
        latent_dim: Output latent embedding size.
        dropout: Dropout applied to attention coefficients.
    """
    def __init__(
        self,
        in_features: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        heads: int = 4,
        edge_dim: int = 3,
        set2set_steps: int = 3,
        latent_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.feature_embed = ScalarFeatureEmbedding(in_features, hidden_dim)
        self.layers = nn.ModuleList([
            AttentionMP(hidden_dim, heads=heads, edge_dim=edge_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.elu = nn.ELU()
        self.set2set = Set2Set(hidden_dim, processing_steps=set2set_steps, num_layers=1)
        self.readout = nn.Linear(2 * hidden_dim, latent_dim)

    def forward(self, data) -> Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, getattr(data, 'edge_attr', None)
        if edge_attr is None:
            # create zero edge features if missing
            edge_attr = torch.zeros(edge_index.size(1), 3, device=x.device, dtype=x.dtype)
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        h = self.feature_embed(x)
        for mp, bn in zip(self.layers, self.bns):
            h_new = mp(h, edge_index, edge_attr)
            h = h + h_new  # residual
            h = bn(h)
            h = self.elu(h)

        pooled = self.set2set(h, batch)  # (batch_size, 2*hidden_dim)
        latent = self.readout(pooled)
        return latent

def bgnd_parse_args():
    desc = "GRAPH-DECODER Arguments"
    parser = argparse.ArgumentParser(description=desc)


    args, _ = parser.parse_known_args()

    return args


BGND_args = bgnd_parse_args

class BondGraphNetDecoder(nn.Module):
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
        hidden_dim: int = 128,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.0,
        rbf_dim: int = 16,
        rbf_min: float = 0.0,
        rbf_max: float = 4.0,
        rbf_gamma: float = 10.0,
        precompute: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.rbf_dim = rbf_dim
        self.register_buffer('rbf_centers', torch.linspace(rbf_min, rbf_max, rbf_dim))
        self.rbf_gamma = rbf_gamma
        self.precompute = precompute

        # ---------- Template graph processing ----------
        # Expect edge_attr: [bond_type_one_hot(5), bond_length]
        assert hasattr(template_data, 'edge_attr'), "template_data must have edge_attr"
        edge_attr_raw: Tensor = template_data.edge_attr.clone().detach()
        assert edge_attr_raw.size(1) >= 6, "edge_attr must contain 5 bond-type one-hot + length"
        bond_types = edge_attr_raw[:, :5]
        bond_len = edge_attr_raw[:, 5]
        # RBF embedding of bond lengths
        bond_rbf = self._rbf_embed(bond_len)
        edge_features = torch.cat([bond_types, bond_rbf], dim=-1)  # (E, 5+rbf_dim)

        # Per-feature MLP embedding for nodes and edges
        self.node_embed = ScalarFeatureEmbedding(in_features=template_data.x.size(1), hidden_dim=hidden_dim)
        self.edge_embed = ScalarFeatureEmbedding(in_features=edge_features.size(1), hidden_dim=hidden_dim)

        # Attention message passing layers (edge_dim = hidden_dim after embedding)
        self.mp_layers = nn.ModuleList([
            AttentionMP(hidden_dim, heads=heads, edge_dim=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.elu = nn.ELU()

        # Precompute structural embedding (optional)
        if precompute:
            with torch.no_grad():
                h_nodes = self.node_embed(template_data.x)
                h_edges = self.edge_embed(edge_features)
                h = h_nodes
                for mp, bn in zip(self.mp_layers, self.bns):
                    h_new = mp(h, template_data.edge_index, h_edges)
                    h = h + h_new
                    h = bn(h)
                    h = self.elu(h)
                self.register_buffer('template_node_repr', h)
        else:
            self.template_node_repr = None  # type: ignore
        # Always store processed edge features for reuse
        self.register_buffer('template_edge_features', edge_features)
        self.register_buffer('template_edge_index', template_data.edge_index.clone())

        # ---------- Build combinatorial sets (bonds, angles, dihedrals) ----------
        self._build_topology_sets(template_data)

        # ---------- Prediction heads ----------
        bond_in = 2 * hidden_dim + latent_dim
        angle_in = 3 * hidden_dim + latent_dim
        dihedral_in = 4 * hidden_dim + latent_dim

        def head(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

        self.bond_head = head(bond_in)
        self.angle_head = head(angle_in)
        self.dih_cos_head = head(dihedral_in)
        self.dih_sin_head = head(dihedral_in)

    # ------------------------------------------------------------------
    def _rbf_embed(self, distances: Tensor) -> Tensor:
        # distances: (E,)
        diff = distances.unsqueeze(-1) - self.rbf_centers  # (E, rbf_dim)
        return torch.exp(-self.rbf_gamma * diff * diff)

    def _build_topology_sets(self, template_data):
        edge_index = template_data.edge_index
        edge_attr = template_data.edge_attr
        # Identify real bonds (exclude virtual). virtual index assumed last of 5 one-hot positions
        real_mask = edge_attr[:, 4] < 0.5  # (E,)
        bonds_set = set()
        for e in range(edge_index.size(1)):
            if not real_mask[e]:
                continue
            i = int(edge_index[0, e])
            j = int(edge_index[1, e])
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            bonds_set.add((a, b))
        bonds_list = sorted(list(bonds_set))
        self.register_buffer('bond_index_pairs', torch.tensor(bonds_list, dtype=torch.long) if bonds_list else torch.empty((0, 2), dtype=torch.long))

        # Build adjacency from real bonds
        adj = {i: set() for i in range(template_data.x.size(0))}
        for (a, b) in bonds_list:
            adj[a].add(b)
            adj[b].add(a)

        # Angles: i-j-k with bonds (i,j) and (j,k), i<k to avoid dup
        angles = []
        for j in range(template_data.x.size(0)):
            neigh = sorted(list(adj[j]))
            ln = len(neigh)
            for u in range(ln):
                for v in range(u + 1, ln):
                    i = neigh[u]; k = neigh[v]
                    if i == k:
                        continue
                    if i < k:
                        angles.append((i, j, k))
                    else:
                        angles.append((k, j, i))
        angles = sorted(list(set(angles)))
        self.register_buffer('angle_index_triples', torch.tensor(angles, dtype=torch.long) if angles else torch.empty((0,3), dtype=torch.long))

        # Dihedrals: i-j-k-l with bonds (i,j),(j,k),(k,l), ensure uniqueness (i<l)
        dihedrals = set()
        for (j, k) in bonds_list:
            # consider j<k already ensured by bonds_list
            lefts = [i for i in adj[j] if i != k]
            rights = [l for l in adj[k] if l != j]
            for i in lefts:
                for l in rights:
                    if len({i, j, k, l}) < 4:
                        continue
                    if i < l:
                        dihedrals.add((i, j, k, l))
                    else:
                        dihedrals.add((l, k, j, i))  # canonical reverse
        dih_list = sorted(list(dihedrals))
        self.register_buffer('dihedral_index_quads', torch.tensor(dih_list, dtype=torch.long) if dih_list else torch.empty((0,4), dtype=torch.long))

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
        bond_idx = self.bond_index_pairs.to(device)
        angle_idx = self.angle_index_triples.to(device)
        dih_idx = self.dihedral_index_quads.to(device)

        preds = {}
        if bond_idx.numel() > 0:
            h_bi = template_h[bond_idx[:,0]]
            h_bj = template_h[bond_idx[:,1]]
            bond_feat = torch.cat([h_bi, h_bj], dim=-1)  # (Nb, 2H)
            bond_feat = bond_feat.unsqueeze(0).expand(B, -1, -1)
            latent_exp = latent.unsqueeze(1).expand(-1, bond_feat.size(1), -1)
            bond_in = torch.cat([bond_feat, latent_exp], dim=-1)
            bond_out = self.bond_head(bond_in.view(-1, bond_in.size(-1))).view(B, -1)
        else:
            bond_out = latent.new_empty((B, 0))
        preds['bond_dist'] = bond_out

        # Angles
        if angle_idx.numel() > 0:
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
        preds['angle'] = angle_out

        # Dihedrals
        if dih_idx.numel() > 0:
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
        preds['dihedral_cos'] = cos_out
        preds['dihedral_sin'] = sin_out

        return preds

class BondGraphNetEncoderDecoder(pl.LightningModule):
    """LightningModule wrapper for BondGraphNet.

    Args:
        gnn_enc_kwargs: keyword arguments passed to BondGraphNetEncoder.
        gnn_dec_kwargs: keyword arguments passed to BondGraphNetDecoder.
        target_dim: output dimension of prediction head.
        lr: learning rate.
        weight_decay: weight decay for AdamW.
        scheduler_gamma: optional exponential LR scheduler gamma (set None to disable).
        loss: loss function (defaults to MSE for regression).
    """
    def __init__(
        self,
        gnn_enc_kwargs: Optional[dict] = None,
        gnn_dec_kwargs: Optional[dict] = None,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        scheduler_gamma: Optional[float] = None,
        loss: Optional[nn.Module] = None,
    ):
        super().__init__()
        gnn_enc_kwargs = gnn_enc_kwargs or {}
        gnn_dec_kwargs = gnn_dec_kwargs or {}
        self.save_hyperparameters(ignore=["loss"])
        self.gnn_enc = BondGraphNetEncoder(**gnn_enc_kwargs)
        self.gnn_dec = BondGraphNetDecoder(**gnn_dec_kwargs)
        self.loss_fn = loss if loss is not None else nn.MSELoss()

    def forward(self, data):  # inference
        latent = self.gnn_enc(data)
        pred = self.gnn_dec(latent)
        return pred

    def step(self, batch, stage: str):
        y = batch.y.view(batch.y.size(0), -1) if hasattr(batch, 'y') else None
        pred = self.forward(batch)
        loss = self.loss_fn(pred, y) if y is not None else torch.tensor(0.0, device=pred.device)
        with torch.no_grad():
            mae = torch.mean(torch.abs(pred - y)) if y is not None else torch.tensor(0.0, device=pred.device)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=(stage=="train"), on_epoch=True)
        self.log(f"{stage}_mae", mae, prog_bar=(stage!="train"), on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward(batch)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.scheduler_gamma is not None:
            sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.hparams.scheduler_gamma)
            return {"optimizer": opt, "lr_scheduler": sched, "monitor": "val_loss"}
        return opt

    def get_latent(self, data):
        return self.gnn(data)

__all__ = ["BondGraphNetEncoder", "BondGraphNetEncoderLit"]

