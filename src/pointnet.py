"""PointNet-style global point-cloud encoder."""
import torch
import torch.nn as nn

class PointNetEncoder(nn.Module):
    """1x1 Conv1d point features + global max/mean pooling."""
    def __init__(self, out_dim=128):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.ReLU(),
        )
        self.projection = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, out_dim), nn.ReLU(),
        )

    def forward(self, points):
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("Expected (batch, n_points, 3)")
        x = self.point_mlp(points.transpose(1, 2))
        x = torch.cat([x.max(dim=2).values, x.mean(dim=2)], dim=1)
        return self.projection(x)
