"""Models of Section 2.6-2.9, written in plain PyTorch (no torch_geometric needed).

A trial graph has only N = 6 nodes, so the adjacency is kept as a dense 6 x 6
matrix and every graph layer is applied to all T windows at once with einsum.

Tensor shapes used throughout:
    x                 [N, T, F]        hand-crafted node features per window
    force_raw         [N, T, 3, L]     decimated raw force windows
    acceleration_raw  [N, T, 3, L]     decimated raw acceleration windows
    adj               [2, N, N]        edge weights: [0] kinematic, [1] thermal
Every model returns (instability_logits [N], ra_prediction [N], embedding [N, H]).
"""
import torch
from torch import nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
class ModalityEncoder(nn.Module):
    """Two-layer 1D-CNN over the raw samples of one window (Section 2.7)."""

    def __init__(self, channels: int = 3, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 16, kernel_size=7, stride=2, padding=3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, out_dim, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        n, t = raw.shape[:2]
        return self.net(raw.reshape(n * t, *raw.shape[2:])).reshape(n, t, -1)


class GatedTemporalConv(nn.Module):
    """Eq. (12): gated 1D convolution along the window index, per zone, with residual."""

    def __init__(self, dim: int, kernel: int = 3):
        super().__init__()
        self.value = nn.Conv1d(dim, dim, kernel, padding=kernel // 2)
        self.gate = nn.Conv1d(dim, dim, kernel, padding=kernel // 2)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:          # h [N, T, H]
        y = h.transpose(1, 2)
        update = self.value(y) * torch.sigmoid(self.gate(y))
        return self.norm(h + update.transpose(1, 2))


class GCNLayer(nn.Module):
    """Eq. (10): H' = sigma( D^-1/2 (A + I) D^-1/2  H W ), A = kinematic + thermal weights."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        a = adj.sum(dim=0) + torch.eye(adj.shape[-1], device=h.device)
        d = a.sum(dim=1).rsqrt()
        a_norm = d[:, None] * a * d[None, :]
        return torch.einsum("ij,jth->ith", a_norm, self.linear(h))


class GATLayer(nn.Module):
    """Eq. (11): graph attention whose score is biased by the physical edge weights.

    e_ij = LeakyReLU( a_l^T W h_i + a_r^T W h_j + b^T w_ij ),  softmax over the
    neighbours j of i (self-loop included). b = 0 recovers the standard GAT.
    """

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        assert dim % heads == 0, "hidden_dim must be divisible by attention_heads"
        self.heads, self.head_dim = heads, dim // heads
        self.linear = nn.Linear(dim, dim, bias=False)
        self.att_src = nn.Parameter(torch.randn(heads, self.head_dim) * 0.1)
        self.att_dst = nn.Parameter(torch.randn(heads, self.head_dim) * 0.1)
        self.edge_bias = nn.Linear(2, heads, bias=False)      # b in Eq. (11)
        self.dropout = nn.Dropout(dropout)
        self.last_attention = None                            # [heads, N, N], averaged over windows

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        n, t, _ = h.shape
        wh = self.linear(h).reshape(n, t, self.heads, self.head_dim)
        src = (wh * self.att_dst).sum(-1)                     # [N, T, heads], term for node i
        dst = (wh * self.att_src).sum(-1)                     # [N, T, heads], term for node j
        scores = src.permute(1, 2, 0)[..., :, None] + dst.permute(1, 2, 0)[..., None, :]   # [T, heads, i, j]
        edge_feat = adj.permute(1, 2, 0)                      # [i, j, 2]
        scores = scores + self.edge_bias(edge_feat).permute(2, 0, 1)[None]
        mask = (adj.sum(0) > 0) | torch.eye(n, dtype=torch.bool, device=h.device)
        scores = F.leaky_relu(scores, 0.2).masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(scores, dim=-1)
        self.last_attention = alpha.detach().mean(dim=0)
        out = torch.einsum("thij,jthd->ithd", self.dropout(alpha), wh)
        return out.reshape(n, t, -1)


class MultiTaskHeads(nn.Module):
    """Two linear heads (Eq. 13): instability logit and standardised Ra."""

    def __init__(self, dim: int):
        super().__init__()
        self.instability = nn.Linear(dim, 1)
        self.roughness = nn.Linear(dim, 1)

    def forward(self, z: torch.Tensor):
        return self.instability(z).squeeze(-1), self.roughness(z).squeeze(-1), z


# --------------------------------------------------------------------------
# Models compared in Tables 2-4
# --------------------------------------------------------------------------
class SpatioTemporalGNN(nn.Module):
    """ST-GCN / ST-GAT: modality encoders -> 2 x (spatial layer + gated temporal conv) -> heads."""

    def __init__(self, feature_dim: int, hidden: int = 64, kind: str = "gat", heads: int = 4,
                 dropout: float = 0.25, kernel: int = 3, temporal: bool = True,
                 use_force_raw: bool = True, use_accel_raw: bool = True):
        super().__init__()
        self.kind, self.temporal_enabled = kind, temporal
        self.use_force_raw, self.use_accel_raw = use_force_raw, use_accel_raw
        self.force_encoder = ModalityEncoder()
        self.accel_encoder = ModalityEncoder()
        self.project = nn.Linear(feature_dim + 32, hidden)
        layer = (lambda: GCNLayer(hidden)) if kind == "gcn" else (lambda: GATLayer(hidden, heads, dropout))
        self.spatial = nn.ModuleList([layer(), layer()])
        self.temporal = nn.ModuleList([GatedTemporalConv(hidden, kernel), GatedTemporalConv(hidden, kernel)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden), nn.LayerNorm(hidden)])
        self.dropout = nn.Dropout(dropout)
        self.heads = MultiTaskHeads(hidden)

    def encode(self, g: dict) -> torch.Tensor:
        f = self.force_encoder(g["force_raw"])
        a = self.accel_encoder(g["acceleration_raw"])
        f = f if self.use_force_raw else torch.zeros_like(f)
        a = a if self.use_accel_raw else torch.zeros_like(a)
        return self.project(torch.cat([g["x"], f, a], dim=-1))

    def forward(self, g: dict):
        h = self.encode(g)
        for block in range(2):
            spatial = torch.relu(self.spatial[block](h, g["adj"]))
            h = self.norms[block](h + self.dropout(spatial))
            if self.temporal_enabled:
                h = self.temporal[block](h)
        return self.heads(h.mean(dim=1))

    def attention_by_edge_type(self, adj: torch.Tensor) -> tuple[float, float]:
        """Mean GAT coefficient on kinematic and on thermal edges (last forward pass)."""
        if self.kind != "gat":
            return float("nan"), float("nan")
        alpha = torch.stack([layer.last_attention.mean(0) for layer in self.spatial]).mean(0)
        result = []
        for channel in range(2):
            w = adj[channel]
            result.append(float((alpha * w).sum() / w.sum()) if w.sum() > 0 else float("nan"))
        return tuple(result)


class FusionMLP(nn.Module):
    """Same encoders and features as the GNN, but every zone is processed alone (no graph)."""

    def __init__(self, feature_dim: int, hidden: int = 64, dropout: float = 0.25):
        super().__init__()
        self.force_encoder = ModalityEncoder()
        self.accel_encoder = ModalityEncoder()
        self.mlp = nn.Sequential(nn.Linear(feature_dim + 32, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.heads = MultiTaskHeads(hidden)

    def forward(self, g: dict):
        z = torch.cat([g["x"], self.force_encoder(g["force_raw"]), self.accel_encoder(g["acceleration_raw"])], -1)
        return self.heads(self.mlp(z).mean(dim=1))


class BiLSTMBaseline(nn.Module):
    """BiLSTM over the sequence of hand-crafted feature vectors of each zone."""

    def __init__(self, feature_dim: int, hidden: int = 64, dropout: float = 0.25):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.heads = MultiTaskHeads(2 * hidden)

    def forward(self, g: dict):
        _, (h, _) = self.lstm(g["x"])
        return self.heads(self.dropout(torch.cat([h[-2], h[-1]], dim=-1)))


class RawCNNBaseline(nn.Module):
    """1D-CNN on raw force and acceleration windows only (no hand-crafted features)."""

    def __init__(self, hidden: int = 64, dropout: float = 0.25):
        super().__init__()
        self.force_encoder = ModalityEncoder()
        self.accel_encoder = ModalityEncoder()
        self.mlp = nn.Sequential(nn.Linear(32, hidden), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden))
        self.heads = MultiTaskHeads(hidden)

    def forward(self, g: dict):
        z = torch.cat([self.force_encoder(g["force_raw"]), self.accel_encoder(g["acceleration_raw"])], -1)
        return self.heads(self.mlp(z.mean(dim=1)))


def build_model(kind: str, feature_dim: int, cfg, variant: str = "full") -> nn.Module:
    """Factory used by train_evaluate.py."""
    t = cfg.train
    if kind in ("gcn", "gat"):
        return SpatioTemporalGNN(feature_dim, t.hidden_dim, kind, t.attention_heads, t.dropout, t.temporal_kernel,
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
