"""
models.py
=========
All neural networks compared in the paper (Sections 2.6-2.9), written in plain
PyTorch (no PyTorch Geometric needed).

A trial graph has only N = 6 nodes, so the adjacency is kept as a small dense
6 x 6 matrix and every graph layer works on all T windows at the same time.

Tensor shapes used everywhere (see graph_data.py):
    x                 [N, T, F]        hand-crafted node features per window
    force_raw         [N, T, 3, L]     down-sampled raw force windows
    acceleration_raw  [N, T, 3, L]     down-sampled raw acceleration windows
    adj               [2, N, N]        edge weights: [0] kinematic, [1] thermal
      N = zones (6), T = windows per zone, F = features, L = samples per window

Every model returns a tuple of 3 things:
    instability_logits [N]   (apply sigmoid to get a probability of chatter)
    ra_prediction      [N]   (standardised Ra; train_evaluate.py converts it back to um)
    embedding          [N, H]

PyTorch reminder
----------------
* A model is a class that inherits from nn.Module.
* __init__ creates the layers (and their random starting weights).
* forward(...) describes how the input flows through the layers. Calling
  model(graph) runs forward(graph).
* IMPORTANT for reproducibility: layers draw their random starting weights in
  the order they are created in __init__. Do not reorder those lines, or the
  results will change slightly even with the same seed.
"""
import torch
from torch import nn
import torch.nn.functional as F


# ===========================================================================
# Building blocks
# ===========================================================================
class ModalityEncoder(nn.Module):
    """Small 1D-CNN that turns ONE raw window [3, L] into a 16-number summary (Section 2.7)."""

    def __init__(self, channels: int = 3, out_dim: int = 16):
        super().__init__()                       # always call the parent __init__ first
        self.net = nn.Sequential(                # layers applied one after the other
            nn.Conv1d(channels, 16, kernel_size=7, stride=2, padding=3),   # 16 filters, halves the length
            nn.BatchNorm1d(16),                  # keeps activations at a stable scale
            nn.ReLU(),                           # max(0, x)
            nn.MaxPool1d(4),                     # keep the max of every 4 samples
            nn.Conv1d(16, out_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))             # average over time -> one value per filter

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        n_zones, n_windows = raw.shape[:2]
        # Conv1d expects [batch, channels, length], so treat every (zone, window)
        # pair as one item of a batch ...
        batch = raw.reshape(n_zones * n_windows, *raw.shape[2:])          # [N*T, 3, L]
        encoded = self.net(batch)                                          # [N*T, 16, 1]
        # ... and put the zone and window axes back.
        return encoded.reshape(n_zones, n_windows, -1)                     # [N, T, 16]


class GatedTemporalConv(nn.Module):
    """Eq. (12): gated 1D convolution along the window axis, for each zone, with a residual."""

    def __init__(self, dim: int, kernel: int = 3):
        super().__init__()
        self.value = nn.Conv1d(dim, dim, kernel, padding=kernel // 2)   # what to add
        self.gate = nn.Conv1d(dim, dim, kernel, padding=kernel // 2)    # how much of it to let through
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:          # h: [N, T, H]
        y = h.transpose(1, 2)                                     # Conv1d wants [N, H, T]
        update = self.value(y) * torch.sigmoid(self.gate(y))      # gate is between 0 and 1
        return self.norm(h + update.transpose(1, 2))              # residual: keep the old h too


class GCNLayer(nn.Module):
    """Eq. (10): graph convolution  H' = D^-1/2 (A + I) D^-1/2  H W.

    A = kinematic weights + thermal weights, I = self-loops, D = node degrees.
    Each zone's new vector is a weighted average of itself and its neighbours.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)                         # the weight matrix W

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        n = adj.shape[-1]
        # A + I : add both edge channels together and add self-loops.
        a = adj.sum(dim=0) + torch.eye(n, device=h.device)
        # D^-1/2 : one over the square root of each node's total edge weight.
        d_inv_sqrt = a.sum(dim=1).rsqrt()
        # Symmetric normalisation: a_norm[i, j] = a[i, j] / sqrt(d_i * d_j).
        a_norm = d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]
        # Mix the zones: out[i, t, :] = sum over j of a_norm[i, j] * (h W)[j, t, :].
        return torch.einsum("ij,jth->ith", a_norm, self.linear(h))


class GATLayer(nn.Module):
    """Eq. (11): graph attention whose scores also see the physical edge weights.

    e_ij = LeakyReLU( a_l . W h_i  +  a_r . W h_j  +  b . w_ij )
    alpha_ij = softmax over the neighbours j of i (self-loop included).
    b = 0 would give the standard GAT.
    """

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        assert dim % heads == 0, "hidden_dim must be divisible by attention_heads"
        self.heads = heads
        self.head_dim = dim // heads
        self.linear = nn.Linear(dim, dim, bias=False)                          # W
        self.att_src = nn.Parameter(torch.randn(heads, self.head_dim) * 0.1)   # a_r (neighbour part)
        self.att_dst = nn.Parameter(torch.randn(heads, self.head_dim) * 0.1)   # a_l (receiving node)
        self.edge_bias = nn.Linear(2, heads, bias=False)                       # b: uses the 2 edge weights
        self.dropout = nn.Dropout(dropout)
        self.last_attention = None      # [heads, N, N] from the last forward pass, for Figure 5

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        n, t, _ = h.shape

        # 1) Project every zone vector and split it into `heads` parts.
        wh = self.linear(h).reshape(n, t, self.heads, self.head_dim)          # [N, T, heads, d]

        # 2) One score term for the receiving node i and one for the neighbour j.
        score_i = (wh * self.att_dst).sum(-1)                                  # [N, T, heads]
        score_j = (wh * self.att_src).sum(-1)                                  # [N, T, heads]

        # 3) Raw score for every pair (i, j): score_i + score_j  -> [T, heads, N, N].
        score_i = score_i.permute(1, 2, 0)[..., :, None]                       # [T, heads, N, 1]
        score_j = score_j.permute(1, 2, 0)[..., None, :]                       # [T, heads, 1, N]
        scores = score_i + score_j                                             # broadcast -> [T, heads, N, N]

        # 4) Add the physical edge information b . w_ij.
        edge_features = adj.permute(1, 2, 0)                                   # [N, N, 2]
        edge_term = self.edge_bias(edge_features).permute(2, 0, 1)             # [heads, N, N]
        scores = scores + edge_term[None]                                      # same for every window

        # 5) Only real neighbours (and the node itself) may receive attention.
        is_neighbour = (adj.sum(0) > 0) | torch.eye(n, dtype=torch.bool, device=h.device)
        scores = F.leaky_relu(scores, 0.2).masked_fill(~is_neighbour, float("-inf"))

        # 6) Softmax over j turns the scores into weights that sum to 1.
        alpha = torch.softmax(scores, dim=-1)                                  # [T, heads, N, N]
        self.last_attention = alpha.detach().mean(dim=0)                       # average over windows

        # 7) New vector of zone i = sum over j of alpha_ij * W h_j (for every head and window).
        out = torch.einsum("thij,jthd->ithd", self.dropout(alpha), wh)         # [N, T, heads, d]
        return out.reshape(n, t, -1)                                           # glue the heads back


class MultiTaskHeads(nn.Module):
    """Two linear outputs (Eq. 13): chatter logit and standardised Ra."""

    def __init__(self, dim: int):
        super().__init__()
        self.instability = nn.Linear(dim, 1)
        self.roughness = nn.Linear(dim, 1)

    def forward(self, z: torch.Tensor):
        logits = self.instability(z).squeeze(-1)       # squeeze removes the last axis of size 1
        ra = self.roughness(z).squeeze(-1)
        return logits, ra, z


# ===========================================================================
# The models compared in Tables 2-4
# ===========================================================================
class SpatioTemporalGNN(nn.Module):
    """ST-GCN / ST-GAT.

    raw windows -> CNN encoders -+
                                 +-> linear projection -> 2 x [spatial graph layer + temporal conv]
    hand-crafted features -------+     -> average over windows -> two output heads
    """

    def __init__(self, feature_dim: int, hidden: int = 64, kind: str = "gat", heads: int = 4,
                 dropout: float = 0.25, kernel: int = 3, temporal: bool = True,
                 use_force_raw: bool = True, use_accel_raw: bool = True):
        super().__init__()
        self.kind = kind                          # "gcn" or "gat"
        self.temporal_enabled = temporal          # False for the "without temporal module" ablation
        self.use_force_raw = use_force_raw
        self.use_accel_raw = use_accel_raw

        # Encoders for the raw windows (16 numbers each -> 32 together).
        self.force_encoder = ModalityEncoder()
        self.accel_encoder = ModalityEncoder()
        self.project = nn.Linear(feature_dim + 32, hidden)

        # Two spatial layers (created one after the other: first, then second).
        spatial_layers = []
        for _ in range(2):
            if kind == "gcn":
                spatial_layers.append(GCNLayer(hidden))
            else:
                spatial_layers.append(GATLayer(hidden, heads, dropout))
        self.spatial = nn.ModuleList(spatial_layers)

        self.temporal = nn.ModuleList([GatedTemporalConv(hidden, kernel), GatedTemporalConv(hidden, kernel)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden), nn.LayerNorm(hidden)])
        self.dropout = nn.Dropout(dropout)
        self.heads = MultiTaskHeads(hidden)

    def encode(self, g: dict) -> torch.Tensor:
        """Combine hand-crafted features and CNN summaries into one vector per zone and window."""
        f = self.force_encoder(g["force_raw"])                        # [N, T, 16]
        a = self.accel_encoder(g["acceleration_raw"])                 # [N, T, 16]
        if not self.use_force_raw:                                    # ablation: hide the force CNN
            f = torch.zeros_like(f)
        if not self.use_accel_raw:                                    # ablation: hide the accel CNN
            a = torch.zeros_like(a)
        combined = torch.cat([g["x"], f, a], dim=-1)                  # [N, T, F + 32]
        return self.project(combined)                                 # [N, T, hidden]

    def forward(self, g: dict):
        h = self.encode(g)
        for block in range(2):
            # Spatial step: zones exchange information along the edges.
            spatial = torch.relu(self.spatial[block](h, g["adj"]))
            h = self.norms[block](h + self.dropout(spatial))          # residual + normalisation
            # Temporal step: each zone looks at neighbouring windows.
            if self.temporal_enabled:
                h = self.temporal[block](h)
        zone_vectors = h.mean(dim=1)                                  # average over windows -> [N, hidden]
        return self.heads(zone_vectors)

    def attention_matrix(self) -> torch.Tensor:
        """GAT attention [N, N] of the last forward pass, averaged over heads and the two layers.

        Row i = how much zone i listens to every zone j (each row sums to 1).
        """
        per_layer = [layer.last_attention.mean(0) for layer in self.spatial]
        return torch.stack(per_layer).mean(0)

    def attention_by_edge_type(self, adj: torch.Tensor) -> tuple[float, float]:
        """Mean GAT attention on kinematic edges and on thermal edges (last forward pass)."""
        if self.kind != "gat":
            return float("nan"), float("nan")
        alpha = self.attention_matrix()
        result = []
        for channel in range(2):                                      # 0 = kinematic, 1 = thermal
            w = adj[channel]
            if w.sum() > 0:
                result.append(float((alpha * w).sum() / w.sum()))     # edge-weighted mean
            else:
                result.append(float("nan"))
        return tuple(result)


class FusionMLP(nn.Module):
    """Same inputs as the GNN, but every zone is processed ALONE (no graph, no edges)."""

    def __init__(self, feature_dim: int, hidden: int = 64, dropout: float = 0.25):
        super().__init__()
        self.force_encoder = ModalityEncoder()
        self.accel_encoder = ModalityEncoder()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + 32, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.heads = MultiTaskHeads(hidden)

    def forward(self, g: dict):
        f = self.force_encoder(g["force_raw"])
        a = self.accel_encoder(g["acceleration_raw"])
        z = torch.cat([g["x"], f, a], dim=-1)                         # [N, T, F + 32]
        zone_vectors = self.mlp(z).mean(dim=1)                        # [N, hidden]
        return self.heads(zone_vectors)


class BiLSTMBaseline(nn.Module):
    """Bidirectional LSTM over the sequence of hand-crafted feature vectors of each zone."""

    def __init__(self, feature_dim: int, hidden: int = 64, dropout: float = 0.25):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.heads = MultiTaskHeads(2 * hidden)                       # forward + backward states

    def forward(self, g: dict):
        # Zones act as the batch, windows as the time steps.
        _, (h, _) = self.lstm(g["x"])                                 # h: [2 directions, N, hidden]
        last_forward, last_backward = h[-2], h[-1]
        zone_vectors = torch.cat([last_forward, last_backward], dim=-1)
        return self.heads(self.dropout(zone_vectors))


class RawCNNBaseline(nn.Module):
    """1D-CNN on the raw force and acceleration windows only (no hand-crafted features)."""

    def __init__(self, hidden: int = 64, dropout: float = 0.25):
        super().__init__()
        self.force_encoder = ModalityEncoder()
        self.accel_encoder = ModalityEncoder()
        self.mlp = nn.Sequential(nn.Linear(32, hidden), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden))
        self.heads = MultiTaskHeads(hidden)

    def forward(self, g: dict):
        f = self.force_encoder(g["force_raw"])
        a = self.accel_encoder(g["acceleration_raw"])
        z = torch.cat([f, a], dim=-1)                                 # [N, T, 32]
        return self.heads(self.mlp(z.mean(dim=1)))


# ===========================================================================
# Factory: build a model from its short name
# ===========================================================================
class ModelFactory:
    """Creates the right model object from a name such as "gat" or "bilstm"."""

    NAMES = ["cnn1d", "bilstm", "fusion_mlp", "gcn", "gat"]

    @staticmethod
    def build(kind: str, feature_dim: int, config, variant: str = "full") -> nn.Module:
        t = config.train
        if kind in ("gcn", "gat"):
            return SpatioTemporalGNN(feature_dim, t.hidden_dim, kind, t.attention_heads, t.dropout,
                                     t.temporal_kernel,
                                     temporal=variant != "without_temporal_module",
                                     use_force_raw=variant != "without_force_features",
                                     use_accel_raw=variant != "without_acceleration_features")
        if kind == "fusion_mlp":
            return FusionMLP(feature_dim, t.hidden_dim, t.dropout)
        if kind == "bilstm":
            return BiLSTMBaseline(feature_dim, t.hidden_dim, t.dropout)
        if kind == "cnn1d":
            return RawCNNBaseline(t.hidden_dim, t.dropout)
        raise ValueError(f"unknown model {kind!r}")
