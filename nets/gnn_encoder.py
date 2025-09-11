import argparse
import math
from typing import List, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, Set2Set
from torch_geometric.utils import softmax
import pytorch_lightning as pl

torch.set_default_dtype(torch.float64)

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
        
        self.node_msg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_msg = nn.Linear(edge_dim, hidden_dim, bias=False)

        self.beta = nn.Sequential(
            nn.Linear(hidden_dim*3, 1, bias=False), 
            nn.Sigmoid()
        )

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

        e_msg = self.edge_msg(edge_attr).view(-1, self.heads, self.d_head)  # (E, heads, d_head)

        k_e = k + e_msg  # (E, heads, d_head)

        logits = (q * k_e).sum(dim=-1) / math.sqrt(self.d_head)  # (E, heads)

        alpha = softmax(logits, index)  # softmax over incoming edges per target node
        alpha = self.attn_drop(alpha)
        alpha = alpha.unsqueeze(-1)  # (E, heads, 1)

        msg = alpha * (v + e_msg)  # (E, heads, d_head)

        msg = msg.view(-1, self.hidden_dim)  # (E, hidden_dim)
              
        return msg

    def update(self, aggr_out: Tensor, x_i: Tensor, index: Tensor) -> Tensor:
        # aggr_out: (N, heads, d_head)
        n_msg_all = self.node_msg(x_i)
        n_msg = torch.zeros_like(aggr_out)
        n_msg.index_add_(0, index, n_msg_all)

        beta = self.beta(torch.cat([n_msg, aggr_out, n_msg - aggr_out], dim=-1))  # (N, heads, 1)
        out = beta * n_msg + (1 - beta) * aggr_out  # (N, heads, d_head)
        out = out.view(-1, self.hidden_dim)  # (N, hidden_dim)
        return out

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
        dropout_mp: float = 0.0,
        dropout_mlp: float = 0.0,
        rbf_dim: int = 16,
        rbf_min: float = 0.0,
        rbf_max: float = 4.0,
        rbf_gamma: float = 10.0,
        precompute: bool = True,
        out_labels: List[str] = ['bond_dist', 'angle', 'dihedral_cos', 'dihedral_sin'],
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout_mp = dropout_mp
        self.dropout_mlp = dropout_mlp
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
        self.node_embed = ScalarFeatureEmbedding(in_features=template_data.x.size(1), hidden_dim=hidden_dim)
        self.edge_embed = ScalarFeatureEmbedding(in_features=edge_features.size(1), hidden_dim=hidden_dim)

        # Attention message passing layers (edge_dim = hidden_dim after embedding)
        self.mp_layers = nn.ModuleList([
            AttentionMP(hidden_dim, heads=heads, edge_dim=hidden_dim, dropout=dropout_mp)
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
                nn.Dropout(dropout_mlp),
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
        preds[self.out_labels[0]] = bond_out

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
        preds[self.out_labels[1]] = angle_out

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
        preds[self.out_labels[2]] = cos_out
        preds[self.out_labels[3]] = sin_out

        return preds

def bgne_parse_args():
    desc = "Bond-GNN-ENCODER-DECODER Arguments"
    parser = argparse.ArgumentParser(description=desc)

    # Encoder args
    parser.add_argument('--in_features', type=int, default=3, help='Number of scalar node features (default 3 for bond nodes)')
    parser.add_argument('--edge_dim', type=int, default=3, help='Edge feature dimension (default 3: angle one-hot + value)')
    parser.add_argument('--enc_hidden_dim', type=int, default=128, help='Hidden embedding size')
    parser.add_argument('--enc_num_layers', type=int, default=4, help='Number of message passing layers (L)')
    parser.add_argument('--enc_heads', type=int, default=4, help='Attention heads')
    parser.add_argument('--set2set_steps', type=int, default=3, help='T processing steps for Set2Set')
    parser.add_argument('--latent_dim', type=int, default=256, help='Output latent embedding size')
    parser.add_argument('--enc_dropout', type=float, default=0.0, help='Dropout applied to attention coefficients')

    # Decoder args
    parser.add_argument('--dec_hidden_dim', type=int, default=128, help='Hidden embedding size for decoder')
    parser.add_argument('--dec_num_layers', type=int, default=4, help='Number of message passing layers (L) for decoder')
    parser.add_argument('--dec_heads', type=int, default=4, help='Attention heads for decoder')
    parser.add_argument('--dec_dropout_mp', type=float, default=0.0, help='Dropout applied to attention coefficients in decoder')
    parser.add_argument('--dec_dropout_mlp', type=float, default=0.0, help='Dropout applied in MLP heads of decoder')

    parser.add_argument('--rbf_dim', type=int, default=16, help='Number of radial basis functions for distance embedding')
    parser.add_argument('--rbf_min', type=float, default=0.0, help='Minimum distance for RBFs')
    parser.add_argument('--rbf_max', type=float, default=4.0, help='Maximum distance for RBFs')
    parser.add_argument('--rbf_gamma', type=float, default=10.0, help='Width parameter for RBFs')

    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the training')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weights regularization for the training')
    parser.add_argument('--normalize_inputs', dest='normIn', action='store_true', help='Whether to normalize the input features')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1, help='Learning rate scheduler gamma')

    args, _ = parser.parse_known_args()

    return args


BGNE_args = bgne_parse_args

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
        in_features: int = 3, ### ENCODER ARGS
        edge_dim: int = 3,
        enc_hidden_dim: int = 128,
        enc_num_layers: int = 4,
        enc_heads: int = 4,
        set2set_steps: int = 3,
        latent_dim: int = 256,
        enc_dropout: float = 0.0,
        template_data = None,  ### DECODER ARGS
        dec_hidden_dim: int = 128,
        dec_num_layers: int = 4,
        dec_heads: int = 4,
        dec_dropout_mp: float = 0.0,
        dec_dropout_mlp: float = 0.0,
        rbf_dim: int = 16,
        rbf_min: float = 0.0,
        rbf_max: float = 4.0,
        rbf_gamma: float = 10.0,
        precompute: bool = True,
        lr: float = 1e-4,       ### OPTIMIZER ARGS
        weight_decay: float = 0.0,
        normIn: bool = False,
        scheduler_gamma: Optional[float] = None,
        loss: Optional[nn.Module] = None,
        loss_weights: Optional[List[float]] = None,
        out_labels: List[str] = ['bond_dist', 'angle', 'dihedral_cos', 'dihedral_sin'],
    ):
        super().__init__()
        assert template_data is not None, "template_data must be provided"
        assert len(out_labels) == 4, "out_labels must be a list of 4 strings"
        assert loss_weights is None or len(loss_weights) == 4, "loss_weights must be None or a list of 4 floats"
        if loss_weights is None:
            loss_weights = [1.0, 1.0, 1.0, 1.0]
        self.loss_weights = loss_weights
        gnn_enc_kwargs = {
            "in_features": in_features,
            "edge_dim": edge_dim,
            "hidden_dim": enc_hidden_dim,
            "num_layers": enc_num_layers,
            "heads": enc_heads,
            "set2set_steps": set2set_steps,
            "latent_dim": latent_dim,
            "dropout": enc_dropout,
        }
        gnn_dec_kwargs = {
            "template_data": template_data,
            "latent_dim": latent_dim,
            "hidden_dim": dec_hidden_dim,
            "num_layers": dec_num_layers,
            "heads": dec_heads,
            "dropout_mp": dec_dropout_mp,
            "dropout_mlp": dec_dropout_mlp,
            "rbf_dim": rbf_dim,
            "rbf_min": rbf_min,
            "rbf_max": rbf_max,
            "rbf_gamma": rbf_gamma,
            "precompute": precompute,
            "out_labels": out_labels,
        }
        self.save_hyperparameters(ignore=["loss"])
        self.gnn_enc = BondGraphNetEncoder(**gnn_enc_kwargs)
        self.gnn_dec = BondGraphNetDecoder(**gnn_dec_kwargs)
        self.loss_fn = loss if loss is not None else nn.MSELoss()
        # Normalization flag & statistics (avoid name clash with method normalize())
        self.normIn = normIn
        self.normSet = False
        # Register buffers for feature-wise mean/range; sized by encoder input features
        self.register_buffer('Mean', torch.zeros(in_features + edge_dim))
        self.register_buffer('Range', torch.ones(in_features + edge_dim))

    def set_norm(self):
        if not self.trainer.datamodule:
            raise RuntimeError("Trainer datamodule not found; cannot compute normalization.")
        with torch.no_grad():
            Mean = torch.tensor(self.trainer.datamodule.get_scaler_mean(), device=self.device)
            Range = torch.tensor(self.trainer.datamodule.get_scaler_scale(), device=self.device)
            assert Mean.size(0) == self.hparams.in_features + self.hparams.edge_dim, \
                f"Mean size {Mean.size(0)} does not match expected {(self.hparams.in_features + self.hparams.edge_dim)}"
            assert Range.size(0) == self.hparams.in_features + self.hparams.edge_dim, \
                f"Range size {Range.size(0)} does not match expected {(self.hparams.in_features + self.hparams.edge_dim)}"
            Range = Range.clone()
            Range[Range == 0.0] = 1.0
            print(f"[{type(self).__name__}] Setting normalization for inputs.")
            self.Mean = Mean
            self.Range = Range
            self.normSet = True
    
    def normalize(self, data):
        """Normalize a PyG Data object's node & edge attributes in-place.

        Expects stored Mean/Range concatenated as [node_feats, edge_feats].
        If called multiple times, skips when already normalized (flag _normalized).
        Falls back to tensor behavior if a plain tensor is passed.
        """
        if not self.normIn:
            return data
        if not self.normSet:
            self.set_norm()

        # Graph Data object normalization
        if getattr(data, '_normalized', False):
            return data

        node_dim = self.hparams.in_features
        edge_dim = self.hparams.edge_dim
        mean_node = self.Mean[:node_dim]
        range_node = self.Range[:node_dim]
        mean_edge = self.Mean[node_dim:node_dim+edge_dim]
        range_edge = self.Range[node_dim:node_dim+edge_dim]

        # Normalize node features
        if hasattr(data, 'x') and data.x is not None:
            if data.x.size(-1) != node_dim:
                raise ValueError(f"Node feature dim mismatch: data.x={data.x.size(-1)} expected={node_dim}")
            if data.x.dtype != mean_node.dtype:
                mean_node = mean_node.to(data.x.dtype)
                range_node = range_node.to(data.x.dtype)
            data.x = (data.x - mean_node.view(1, -1)) / range_node.view(1, -1)

        # Normalize edge features (only first edge_dim columns if extra exist)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            if data.edge_attr.size(-1) < edge_dim:
                raise ValueError(f"Edge feature dim mismatch: edge_attr={data.edge_attr.size(-1)} expected>={edge_dim}")
            ea = data.edge_attr
            if ea.dtype != mean_edge.dtype:
                mean_edge = mean_edge.to(ea.dtype)
                range_edge = range_edge.to(ea.dtype)
            head = (ea[:, :edge_dim] - mean_edge.view(1, -1)) / range_edge.view(1, -1)
            if ea.size(-1) > edge_dim:
                data.edge_attr = torch.cat([head, ea[:, edge_dim:]], dim=-1)
            else:
                data.edge_attr = head

        setattr(data, '_normalized', True)
        return data
    
    def denormalize(self, data):
        """Inverse of normalize for a Data object or tensor.

        Only reverses if object was previously normalized (or is a tensor).
        """
        if not self.normIn:
            return data
        if not self.normSet:
            self.set_norm()

        if not getattr(data, '_normalized', False):
            return data

        node_dim = self.hparams.in_features
        edge_dim = self.hparams.edge_dim
        mean_node = self.Mean[:node_dim]
        range_node = self.Range[:node_dim]
        mean_edge = self.Mean[node_dim:node_dim+edge_dim]
        range_edge = self.Range[node_dim:node_dim+edge_dim]

        if hasattr(data, 'x') and data.x is not None:
            if data.x.dtype != mean_node.dtype:
                mean_node = mean_node.to(data.x.dtype)
                range_node = range_node.to(data.x.dtype)
            data.x = data.x * range_node.view(1, -1) + mean_node.view(1, -1)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            ea = data.edge_attr
            if ea.size(-1) >= edge_dim:
                if ea.dtype != mean_edge.dtype:
                    mean_edge = mean_edge.to(ea.dtype)
                    range_edge = range_edge.to(ea.dtype)
                head = ea[:, :edge_dim] * range_edge.view(1, -1) + mean_edge.view(1, -1)
                if ea.size(-1) > edge_dim:
                    data.edge_attr = torch.cat([head, ea[:, edge_dim:]], dim=-1)
                else:
                    data.edge_attr = head
        setattr(data, '_normalized', False)
        return data

    def forward(self, data):  # inference
        data = self.normalize(data)
        latent = self.gnn_enc(data)
        pred = self.gnn_dec(latent)
        return pred

    def extract_labels(self, batch):
        """Compute target geometric labels from batched graph coordinates.

        Expects batch.pos (N_total, 3) containing 3D coordinates for all graphs
        concatenated, and batch.batch (N_total,) with graph indices.
        Uses the template index buffers (bond/angle/dihedral) defined in decoder;
        assumes every graph in batch shares the same topology as template.
        Returns dict mapping each out_label to a (B, count) tensor.
        Missing topology yields empty tensors with correct batch dimension.
        """
        if not hasattr(batch, 'pos'):
            raise AttributeError("Batch must have 'pos' attribute with node coordinates.")
        pos = batch.pos  # (N_total, 3)
        if pos.dim() != 2 or pos.size(-1) != 3:
            raise ValueError("batch.pos must have shape (N_total, 3)")
        graph_idx = getattr(batch, 'batch', None)
        if graph_idx is None:
            # Single graph - fabricate batch indices
            graph_idx = pos.new_zeros(pos.size(0), dtype=torch.long)
        B = int(graph_idx.max().item()) + 1

        dec = self.gnn_dec
        device = pos.device
        # Retrieve index buffers
        bond_idx = dec.bond_index_pairs  # (Nb, 2)
        angle_idx = dec.angle_index_triples  # (Na, 3)
        dih_idx = dec.dihedral_index_quads  # (Nd, 4)

        def gather(node_indices: torch.Tensor) -> torch.Tensor:
            # node_indices: (K, m) referencing per-graph node indices
            # We assume node ordering per graph matches template ordering.
            # Build an expanded (B, K, m) tensor of global indices.
            if node_indices.numel() == 0:
                return pos.new_empty((B, node_indices.size(0), 0, 3))
            # For each graph g, offset = index where graph_idx == g starts.
            # Faster: precompute per-graph node list assuming equal node count.
            counts = torch.bincount(graph_idx, minlength=B)
            if counts.unique().numel() != 1:
                raise ValueError("All graphs must have identical node count to use shared template indices.")
            n_per = counts[0].item()
            # global index = g*n_per + local_index
            g = torch.arange(B, device=device).view(B, 1, 1)
            ni = node_indices.to(device).view(1, *node_indices.shape).expand(B, -1, -1)
            global_idx = g * n_per + ni  # (B, K, m)
            return pos[global_idx]  # (B, K, m, 3)

        labels = {}

        # Bonds: distances
        if bond_idx.numel() > 0:
            pts = gather(bond_idx)  # (B, Nb, 2, 3)
            diffs = pts[:, :, 0, :] - pts[:, :, 1, :]
            bond_dist = diffs.norm(dim=-1)  # (B, Nb)
        else:
            bond_dist = pos.new_empty((B, 0))
        labels[self.hparams['out_labels'][0]] = bond_dist

        # Angles: angle between vectors (j->i) and (j->k)
        if angle_idx.numel() > 0:
            pts = gather(angle_idx)  # (B, Na, 3, 3) order (i,j,k)
            v1 = pts[:, :, 0] - pts[:, :, 1]  # (B, Na, 3)
            v2 = pts[:, :, 2] - pts[:, :, 1]
            v1_n = F.normalize(v1, dim=-1)
            v2_n = F.normalize(v2, dim=-1)
            cos_ang = (v1_n * v2_n).sum(dim=-1).clamp(-1.0, 1.0)
            angle = torch.acos(cos_ang)
        else:
            angle = pos.new_empty((B, 0))
        labels[self.hparams['out_labels'][1]] = angle

        # Dihedrals: compute torsion angle i-j-k-l; output cos and sin
        if dih_idx.numel() > 0:
            pts = gather(dih_idx)  # (B, Nd, 4, 3) order (i,j,k,l)
            p0 = pts[:, :, 0]
            p1 = pts[:, :, 1]
            p2 = pts[:, :, 2]
            p3 = pts[:, :, 3]
            b0 = p1 - p0
            b1 = p2 - p1
            b2 = p3 - p2
            # Normalize b1 for stability
            b1n = F.normalize(b1, dim=-1)
            # Build normals
            n0 = torch.cross(b0, b1, dim=-1)
            n1 = torch.cross(b1, b2, dim=-1)
            n0 = F.normalize(n0, dim=-1)
            n1 = F.normalize(n1, dim=-1)
            m1 = torch.cross(n0, b1n, dim=-1)
            x = (n0 * n1).sum(dim=-1)
            y = (m1 * n1).sum(dim=-1)
            dih = torch.atan2(y, x)
            dih_cos = torch.cos(dih)
            dih_sin = torch.sin(dih)
        else:
            dih_cos = pos.new_empty((B, 0))
            dih_sin = pos.new_empty((B, 0))
        labels[self.hparams['out_labels'][2]] = dih_cos
        labels[self.hparams['out_labels'][3]] = dih_sin

        return labels


    def step(self, batch, stage: str):
        pred = self.forward(batch)
        labels = self.extract_labels(batch)

        loss = self.loss_fn(pred[self.hparams['out_labels'][0]], labels[self.hparams['out_labels'][0]]) * self.loss_weights[0], \
               self.loss_fn(pred[self.hparams['out_labels'][1]], labels[self.hparams['out_labels'][1]]) * self.loss_weights[1], \
                self.loss_fn(pred[self.hparams['out_labels'][2]], labels[self.hparams['out_labels'][2]]) * self.loss_weights[2], \
                self.loss_fn(pred[self.hparams['out_labels'][3]], labels[self.hparams['out_labels'][3]]) * self.loss_weights[3]

        with torch.no_grad():
            mae = (torch.abs(pred[self.hparams['out_labels'][0]] - labels[self.hparams['out_labels'][0]]).mean() if labels[self.hparams['out_labels'][0]].numel() > 0 else torch.tensor(0.0, device=loss.device)), \
                  (torch.abs(pred[self.hparams['out_labels'][1]] - labels[self.hparams['out_labels'][1]]).mean() if labels[self.hparams['out_labels'][1]].numel() > 0 else torch.tensor(0.0, device=loss.device)), \
                  (torch.abs(pred[self.hparams['out_labels'][2]] - labels[self.hparams['out_labels'][2]]).mean() if labels[self.hparams['out_labels'][2]].numel() > 0 else torch.tensor(0.0, device=loss.device)), \
                  (torch.abs(pred[self.hparams['out_labels'][3]] - labels[self.hparams['out_labels'][3]]).mean() if labels[self.hparams['out_labels'][3]].numel() > 0 else torch.tensor(0.0, device=loss.device))

        for i in range(len(self.hparams['out_labels'])):
            self.log(f"{stage}_{self.hparams['out_labels'][i]}_mae", mae[i], prog_bar=False, on_epoch=True)  
            self.log(f"{stage}_{self.hparams['out_labels'][i]}_loss", loss[i], prog_bar=False, on_step=(stage=="train"), on_epoch=True)
        mae = sum(mae) / len(mae)
        self.log(f"{stage}_mae", mae, prog_bar=(stage!="train"), on_epoch=True)
        loss = sum(loss)
        self.log(f"{stage}_loss", loss, prog_bar=(stage=="train"), on_step=(stage=="train"), on_epoch=True)
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
        return self.gnn_enc(data)

__all__ = ["BondGraphNetEncoderDecoder"]

