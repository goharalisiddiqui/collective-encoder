import math
from typing import Optional, Union


import torch
from torch import nn, Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class ScalarFeatureEmbedding(nn.Module):
    """Applies one independent MLP per scalar feature dimension and sums outputs.

    Given input x of shape (N, F), we create F small MLPs each processing x[:, f:f+1].
    Each MLP outputs (N, hidden_dim); final embedding h is the (optionally scaled) sum.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128, activation=nn.ELU()):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                activation,
                nn.Linear(hidden_dim, out_dim),
            )
            for _ in range(in_dim)
        ])

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, F)
        outs = []
        for f, mlp in enumerate(self.mlps):
            outs.append(mlp(x[:, f:f+1]))
        h = torch.stack(outs, dim=0).sum(dim=0)  # (N, hidden_dim)
        h = h / math.sqrt(self.in_dim)  # scale
        return h


class AttentionMP(MessagePassing):
    """Multi-head dot-product attention message passing layer with edge features.

    Attention score per head: a_ij^h = (q_i^h · k_j^h)/sqrt(d) + b_e^h
    where b_e^h is a learned scalar bias from transformed edge features.
    Message: m_ij^h = a_ij^h * (v_j^h + e_msg_ij^h)
    Aggregation: sum over j -> i
    Output: residual + linear projection + optional norm & activation handled externally.
    """
    def __init__(self, node_feat_dim: int, edge_feat_dim: int, hidden_dim: int = 128, heads: int = 4, dropout: float = 0.0):
        super().__init__(aggr='add', node_dim=0)
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.d_head = hidden_dim // heads
        self.dropout = dropout

        self.q_proj = nn.Linear(node_feat_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(node_feat_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(node_feat_dim, hidden_dim, bias=False)

        self.node_msg = nn.Linear(node_feat_dim, hidden_dim, bias=False)
        self.edge_msg = nn.Linear(edge_feat_dim, hidden_dim, bias=False)

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

class EdgeModel(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim):
        super(EdgeModel, self).__init__()
        # Define MLP for edge feature updates
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_in_dim + edge_in_dim, edge_in_dim),
            nn.Tanh(),
            nn.Linear(edge_in_dim, edge_in_dim),
        )

    def forward(self, src, dest, edge_attr, u, batch):
        # Concatenate source and destination node features and edge attributes
        out = torch.cat([src, dest, edge_attr], dim=1)
        return self.edge_mlp(out)