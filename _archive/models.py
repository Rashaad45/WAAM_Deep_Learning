"""Spatial GCN/GAT and non-graph baselines with zone-level multitask heads."""
import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv


class ModalityEncoder(nn.Module):
    def __init__(self, channels: int, output_dim: int = 16):
        super().__init__()
        self.network = nn.Sequential(nn.Conv1d(channels, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
                                     nn.Conv1d(16, output_dim, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class ThermalEncoder(nn.Module):
    def __init__(self, output_dim: int = 16):
        super().__init__()
        self.network = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
                                     nn.Conv2d(8, output_dim, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).flatten(1)


class GatedTemporalConv(nn.Module):
    """Gated 1D convolution over ordered windows independently for each zone."""
    def __init__(self, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.value = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.gate = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        update = self.value(y) * torch.sigmoid(self.gate(y))
        return self.norm(x + update.transpose(1, 2))


class MultiTaskHeads(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.instability = nn.Linear(hidden_dim, 1)
        self.roughness = nn.Linear(hidden_dim, 1)

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.instability(embedding).squeeze(-1), self.roughness(embedding).squeeze(-1), embedding


class SpatioTemporalGraphNet(nn.Module):
    """Two spatial message-passing blocks, each followed by gated temporal conv."""
    def __init__(self, feature_dim: int, force_channels: int, acceleration_channels: int,
                 hidden_dim: int = 64, heads: int = 4, dropout: float = 0.25, kind: str = "gat",
                 temporal_enabled: bool = True, use_thermal: bool = True):
        super().__init__()
        self.kind = kind.lower()
        self.temporal_enabled = temporal_enabled
        self.use_thermal = use_thermal
        if self.kind not in {"gcn", "gat"}:
            raise ValueError("kind must be 'gcn' or 'gat'")
        self.force_encoder = ModalityEncoder(force_channels)
        self.acceleration_encoder = ModalityEncoder(acceleration_channels)
        self.thermal_encoder = ThermalEncoder() if use_thermal else None
        self.input_projection = nn.Linear(feature_dim + (48 if use_thermal else 32), hidden_dim)
        if self.kind == "gcn":
            self.spatial = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(2)])
        else:
            self.spatial = nn.ModuleList([GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True, edge_dim=2, dropout=dropout),
                                          GATConv(hidden_dim, hidden_dim, heads=1, concat=False, edge_dim=2, dropout=dropout)])
        self.temporal = nn.ModuleList([GatedTemporalConv(hidden_dim), GatedTemporalConv(hidden_dim)])
        self.last_edge_attention = []
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)])
        self.dropout = nn.Dropout(dropout)
        self.heads = MultiTaskHeads(hidden_dim)

    def _encode_modalities(self, data) -> torch.Tensor:
        n, t = data.x.shape[:2]
        force = self.force_encoder(data.force_raw.reshape(n * t, *data.force_raw.shape[2:]))
        acceleration = self.acceleration_encoder(data.acceleration_raw.reshape(n * t, *data.acceleration_raw.shape[2:]))
        parts = [data.x.reshape(n * t, -1), force, acceleration]
        if self.use_thermal:
            thermal = self.thermal_encoder(data.thermal_raw.reshape(n * t, *data.thermal_raw.shape[2:]))
            parts.append(thermal)
        z = torch.cat(parts, dim=-1)
        return self.input_projection(z).reshape(n, t, -1)

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self._encode_modalities(data)
        self.last_edge_attention = []
        weights = data.edge_attr.sum(dim=1).clamp_min(1e-8) if data.edge_attr.numel() else None
        for block in range(2):
            time_steps = []
            for time in range(x.shape[1]):
                current = x[:, time]
                if self.kind == "gcn":
                    updated = self.spatial[block](current, data.edge_index, edge_weight=weights)
                else:
                    updated, attention = self.spatial[block](current, data.edge_index, data.edge_attr,
                                                             return_attention_weights=True)
                    self.last_edge_attention.append((attention[0].detach(), attention[1].detach()))
                time_steps.append(torch.relu(updated))
            spatial = torch.stack(time_steps, dim=1)
            x = self.norms[block](x + self.dropout(spatial))
            if self.temporal_enabled:
                x = self.temporal[block](x)
        embedding = x.mean(dim=1)
        return self.heads(embedding)


class MultimodalFusionMLP(nn.Module):
    """Same encoded modalities as ST-GAT, but nodes are classified independently."""
    def __init__(self, feature_dim: int, force_channels: int, acceleration_channels: int,
                 hidden_dim: int = 64, dropout: float = 0.25, use_thermal: bool = True):
        super().__init__()
        self.use_thermal = use_thermal
        self.force_encoder = ModalityEncoder(force_channels)
        self.acceleration_encoder = ModalityEncoder(acceleration_channels)
        self.thermal_encoder = ThermalEncoder() if use_thermal else None
        self.network = nn.Sequential(nn.Linear(feature_dim + (48 if use_thermal else 32), hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.heads = MultiTaskHeads(hidden_dim)

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, t = data.x.shape[:2]
        f = self.force_encoder(data.force_raw.reshape(n * t, *data.force_raw.shape[2:]))
        a = self.acceleration_encoder(data.acceleration_raw.reshape(n * t, *data.acceleration_raw.shape[2:]))
        parts = [data.x.reshape(n * t, -1), f, a]
        if self.use_thermal:
            parts.append(self.thermal_encoder(data.thermal_raw.reshape(n * t, *data.thermal_raw.shape[2:])))
        merged = torch.cat(parts, dim=-1).reshape(n, t, -1).mean(dim=1)
        return self.heads(self.network(merged))


class BiLSTMFeatureBaseline(nn.Module):
    """BiLSTM on hand-crafted multimodal feature sequences, without graph edges."""
    def __init__(self, feature_dim: int, hidden_dim: int = 64, dropout: float = 0.25):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.heads = MultiTaskHeads(hidden_dim * 2)

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, (hidden, _) = self.lstm(data.x)
        return self.heads(self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=-1)))


class RawCNNBaseline(nn.Module):
    """1D-CNN benchmark using raw force and acceleration windows only."""
    def __init__(self, force_channels: int, acceleration_channels: int, hidden_dim: int = 64, dropout: float = 0.25):
        super().__init__()
        self.force_encoder = ModalityEncoder(force_channels)
        self.acceleration_encoder = ModalityEncoder(acceleration_channels)
        self.projection = nn.Sequential(nn.Linear(32, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_dim))
        self.heads = MultiTaskHeads(hidden_dim)

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zones, windows = data.x.shape[:2]
        force = self.force_encoder(data.force_raw.reshape(zones * windows, *data.force_raw.shape[2:]))
        acceleration = self.acceleration_encoder(data.acceleration_raw.reshape(zones * windows, *data.acceleration_raw.shape[2:]))
        temporal = torch.cat((force, acceleration), dim=-1).reshape(zones, windows, 32).mean(dim=1)
        return self.heads(self.projection(temporal))
