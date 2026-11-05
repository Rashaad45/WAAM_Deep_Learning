"""
models.py - neural networks of the benchmark (plain PyTorch). Every model returns ONE chatter logit
per zone, shape [N]. Window -> zone aggregation is the SAME for every model: the mean over the
T windows of the last per-window representation (AUDIT item 4). No Ra head (ZONE_RA = no).

Shapes: x [N, T, F]; force_raw / acceleration_raw [N, T, 3, L]; adj [N, N] (kinematic edges only).
"""
import torch
from torch import nn
import torch.nn.functional as F


def window_mean(h: torch.Tensor) -> torch.Tensor:
    """The one aggregation rule: [N, T, H] -> [N, H]."""
    return h.mean(dim=1)


class ModalityEncoder(nn.Module):
    """1D-CNN: one raw window [3, L] -> 16 numbers."""

    def __init__(self, channels: int = 3, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 16, 7, stride=2, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, out_dim, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))

    def forward(self, raw):
        n, t = raw.shape[:2]
        return self.net(raw.reshape(n * t, *raw.shape[2:])).reshape(n, t, -1)


def normalised_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """D^-1/2 (A + I) D^-1/2 (Kipf & Welling)."""
    a = adj + torch.eye(adj.shape[-1], device=adj.device)
    d = a.sum(1).rsqrt()
    return d[:, None] * a * d[None, :]


class GCNLayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, h, adj):                     # h [N, T, H]
        return torch.einsum("ij,jth->ith", normalised_adjacency(adj), self.linear(h))


class GATLayer(nn.Module):
    """Graph attention with a learned bias b * a_ij from the kinematic edge weight."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.hd = heads, dim // heads
        self.linear = nn.Linear(dim, dim, bias=False)
        self.att_src = nn.Parameter(torch.randn(heads, self.hd) * 0.1)
        self.att_dst = nn.Parameter(torch.randn(heads, self.hd) * 0.1)
        self.edge_bias = nn.Parameter(torch.zeros(heads))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        n, t, _ = h.shape
        wh = self.linear(h).reshape(n, t, self.heads, self.hd)
        si = (wh * self.att_dst).sum(-1).permute(1, 2, 0)[..., :, None]   # [T, heads, N, 1]
        sj = (wh * self.att_src).sum(-1).permute(1, 2, 0)[..., None, :]   # [T, heads, 1, N]
        scores = si + sj + (self.edge_bias[:, None, None] * adj[None])[None]
        mask = (adj > 0) | torch.eye(n, dtype=torch.bool, device=h.device)
        alpha = torch.softmax(F.leaky_relu(scores, 0.2).masked_fill(~mask, float("-inf")), dim=-1)
        return torch.einsum("thij,jthd->ithd", self.dropout(alpha), wh).reshape(n, t, -1)


class GatedTemporalConv(nn.Module):
    def __init__(self, dim: int, kernel: int = 3):
        super().__init__()
        self.value = nn.Conv1d(dim, dim, kernel, padding=kernel // 2)
        self.gate = nn.Conv1d(dim, dim, kernel, padding=kernel // 2)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h):
        y = h.transpose(1, 2)
        return self.norm(h + (self.value(y) * torch.sigmoid(self.gate(y))).transpose(1, 2))


class SpatioTemporalGNN(nn.Module):
    """ST-GCN / ST-GAT: CNN encoders + hand-crafted features -> 2 x (graph layer + gated temporal conv)."""

    def __init__(self, feature_dim, hidden=64, kind="gat", heads=4, dropout=0.25, kernel=3):
        super().__init__()
        self.kind = kind
        self.force_encoder, self.accel_encoder = ModalityEncoder(), ModalityEncoder()
        self.project = nn.Linear(feature_dim + 32, hidden)
        self.spatial = nn.ModuleList([GCNLayer(hidden, hidden) if kind == "gcn" else GATLayer(hidden, heads, dropout)
                                      for _ in range(2)])
        self.temporal = nn.ModuleList([GatedTemporalConv(hidden, kernel) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, g):
        h = self.project(torch.cat([g["x"], self.force_encoder(g["force_raw"]),
                                    self.accel_encoder(g["acceleration_raw"])], -1))
        for k in range(2):
            h = self.norms[k](h + self.dropout(torch.relu(self.spatial[k](h, g["adj"]))))
            h = self.temporal[k](h)
        return self.head(window_mean(h)).squeeze(-1)


class SmallGNN(nn.Module):
    """<= 5k parameters: hand-crafted features only, per-window projection, 2 GCN layers, linear head."""

    def __init__(self, feature_dim, hidden=16, dropout=0.25):
        super().__init__()
        self.project = nn.Linear(feature_dim, hidden)
        self.gcn1, self.gcn2 = GCNLayer(hidden, hidden), GCNLayer(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, g):
        h = torch.relu(self.project(g["x"]))
        h = self.dropout(torch.relu(self.gcn1(h, g["adj"])))
        h = torch.relu(self.gcn2(h, g["adj"]))
        return self.head(window_mean(h)).squeeze(-1)


class FusionMLP(nn.Module):
    """Same inputs as the ST-GNN, every zone processed alone (no edges)."""

    def __init__(self, feature_dim, hidden=64, dropout=0.25):
        super().__init__()
        self.force_encoder, self.accel_encoder = ModalityEncoder(), ModalityEncoder()
        self.mlp = nn.Sequential(nn.Linear(feature_dim + 32, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.head = nn.Linear(hidden, 1)

    def forward(self, g):
        z = torch.cat([g["x"], self.force_encoder(g["force_raw"]), self.accel_encoder(g["acceleration_raw"])], -1)
        return self.head(window_mean(self.mlp(z))).squeeze(-1)


class BiLSTMBaseline(nn.Module):
    """BiLSTM over the window sequence of hand-crafted features; outputs averaged over windows."""

    def __init__(self, feature_dim, hidden=64, dropout=0.25):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, g):
        out, _ = self.lstm(g["x"])                     # [N, T, 2H]
        return self.head(self.dropout(window_mean(out))).squeeze(-1)


class RawCNNBaseline(nn.Module):
    """1D-CNN on raw force and acceleration windows only."""

    def __init__(self, hidden=64, dropout=0.25):
        super().__init__()
        self.force_encoder, self.accel_encoder = ModalityEncoder(), ModalityEncoder()
        self.mlp = nn.Sequential(nn.Linear(32, hidden), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden))
        self.head = nn.Linear(hidden, 1)

    def forward(self, g):
        z = torch.cat([self.force_encoder(g["force_raw"]), self.accel_encoder(g["acceleration_raw"])], -1)
        return self.head(window_mean(self.mlp(z))).squeeze(-1)


DEEP_MODELS = ["cnn1d", "bilstm", "fusion_mlp", "gcn", "gat", "small_gnn"]


def build_model(kind: str, feature_dim: int, cfg) -> nn.Module:
    t = cfg.train
    if kind in ("gcn", "gat"):
        return SpatioTemporalGNN(feature_dim, t.hidden_dim, kind, t.attention_heads, t.dropout, t.temporal_kernel)
    if kind == "small_gnn":
        return SmallGNN(feature_dim, t.small_hidden_dim, t.dropout)
    if kind == "fusion_mlp":
        return FusionMLP(feature_dim, t.hidden_dim, t.dropout)
    if kind == "bilstm":
        return BiLSTMBaseline(feature_dim, t.hidden_dim, t.dropout)
    if kind == "cnn1d":
        return RawCNNBaseline(t.hidden_dim, t.dropout)
    raise ValueError(kind)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
