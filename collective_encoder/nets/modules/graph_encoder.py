import torch
from torch import nn, Tensor
from torch_geometric.nn import Set2Set, MetaLayer

from .mp_modules import ScalarFeatureEmbedding, AttentionMP, EdgeModel

class BondGraphEncoder(nn.Module):
    """Bond-based message passing GNN Encoder with multi-head attention and Set2Set pooling.

    Steps:
      1. Per-scalar feature MLP embeddings summed to initial node embedding h.
      2. L attention message passing layers (each with residual, BatchNorm, ELU).
      3. Set2Set pooling over nodes for T processing steps.
      4. Final linear layer maps pooled embedding to latent_dim.

    Args:
        in_features: Number of scalar node features (default 3 for bond nodes).
        edge_dim: Edge feature dimension (default 3: angle one-hot + value).
        hidden_dim: Hidden embedding size.
        num_layers: Number of message passing layers (L).
        heads: Attention heads.
        set2set_steps: T processing steps for Set2Set.
        latent_dim: Output latent embedding size.
        dropout: Dropout applied to attention coefficients.
    """
    def __init__(
        self,
        node_feat: int,
        edge_feat: int,
        latent_dim: int,
        node_embed_dim: int = 10,
        edge_embed_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 4,
        heads: int = 4,
        set2set_steps: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_embed_dim = node_embed_dim
        self.edge_embed_dim = edge_embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        
        self.node_embed = ScalarFeatureEmbedding(node_feat, node_embed_dim)
        self.edge_embed = ScalarFeatureEmbedding(edge_feat, edge_embed_dim)

        self.node_dim = node_embed_dim #* node_feat
        self.edge_dim = edge_embed_dim #* edge_feat

        self.layers = nn.ModuleList([
            AttentionMP(node_feat_dim=self.node_dim if i == 0 else hidden_dim, 
                    edge_feat_dim=self.edge_dim, 
                    hidden_dim=hidden_dim, 
                    heads=heads, 
                    dropout=dropout)
            for i in range(num_layers)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.elu = nn.ELU()
        self.set2set = Set2Set(hidden_dim, processing_steps=set2set_steps, num_layers=1)
        self.readout = nn.Linear(2 * hidden_dim, latent_dim)

    def forward(self, data) -> Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, getattr(data, 'edge_attr', None)
        if edge_attr is None:
            # create zero edge features if missing
            edge_attr = torch.zeros(edge_index.size(1), self.edge_dim, device=x.device, dtype=x.dtype)
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        h = self.node_embed(x)
        e = self.edge_embed(edge_attr)
        for mp, bn in zip(self.layers, self.bns):
            h = mp(h, edge_index, e)
            h = bn(h)
            h = self.elu(h)

        pooled = self.set2set(h, batch)  # (batch_size, 2*hidden_dim)
        latent = self.readout(pooled)
        return latent

class BondGraphEncoderV2(nn.Module):
    """Bond-based message passing GNN Encoder with multi-head attention and Set2Set pooling.

    Steps:
      1. Per-scalar feature MLP embeddings summed to initial node embedding h.
      2. L attention message passing layers (each with residual, BatchNorm, ELU).
      3. Set2Set pooling over nodes for T processing steps.
      4. Final linear layer maps pooled embedding to latent_dim.

    Args:
        in_features: Number of scalar node features (default 3 for bond nodes).
        edge_dim: Edge feature dimension (default 3: angle one-hot + value).
        hidden_dim: Hidden embedding size.
        num_layers: Number of message passing layers (L).
        heads: Attention heads.
        set2set_steps: T processing steps for Set2Set.
        latent_dim: Output latent embedding size.
        dropout: Dropout applied to attention coefficients.
    """
    def __init__(
        self,
        node_feat: int = 3,
        edge_feat: int = 3,
        node_embed_dim: int = 10,
        edge_embed_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 4,
        heads: int = 4,
        set2set_steps: int = 3,
        latent_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_embed_dim = node_embed_dim
        self.edge_embed_dim = edge_embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        
        self.node_embed = ScalarFeatureEmbedding(node_feat, node_embed_dim)
        self.edge_embed = ScalarFeatureEmbedding(edge_feat, edge_embed_dim)

        self.node_dim = node_embed_dim #* node_feat
        self.edge_dim = edge_embed_dim #* edge_feat

        self.layers = nn.ModuleList([
            MetaLayer(
                EdgeModel(self.node_dim if i == 0 else hidden_dim, self.edge_dim),
                AttentionMP(node_feat_dim=self.node_dim if i == 0 else hidden_dim, 
                        edge_feat_dim=self.edge_dim, 
                        hidden_dim=hidden_dim, 
                        heads=heads, 
                        dropout=dropout),
                None,
            )
            for i in range(num_layers)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.elu = nn.ELU()
        self.set2set = Set2Set(hidden_dim, processing_steps=set2set_steps, num_layers=1)
        self.readout = nn.Linear(2 * hidden_dim, latent_dim)

    def forward(self, data) -> Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, getattr(data, 'edge_attr', None)
        if edge_attr is None:
            # create zero edge features if missing
            edge_attr = torch.zeros(edge_index.size(1), self.edge_dim, device=x.device, dtype=x.dtype)
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        h = self.node_embed(x)
        e = self.edge_embed(edge_attr)
        for mp, bn in zip(self.layers, self.bns):
            h, e, _ = mp(h, edge_index, e, None, batch)
            h = bn(h)
            h = self.elu(h)

        pooled = self.set2set(h, batch)  # (batch_size, 2*hidden_dim)
        latent = self.readout(pooled)
        return latent