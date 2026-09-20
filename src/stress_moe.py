"""Three-expert multimodal MoE for automotive hood stress prediction."""
import torch
import torch.nn as nn
try:
    from .pointnet import PointNetEncoder
except ImportError:
    from pointnet import PointNetEncoder

class ParameterEncoder(nn.Module):
    def __init__(self, in_dim=54):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class GeometryEncoder(nn.Module):
    def __init__(self, in_dim=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,64), nn.ReLU(),
            nn.Linear(64,64), nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class StressExpert(nn.Module):
    def __init__(self, in_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,64), nn.ReLU(),
            nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32,1)
        )
    def forward(self, x):
        return self.net(x)

class Gate(nn.Module):
    def __init__(self, in_dim=128, n_experts=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,64), nn.ReLU(),
            nn.Linear(64,n_experts)
        )
    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)

class StressMoE(nn.Module):
    """Documented final architecture: points + 54 parameters + 24 descriptors."""
    def __init__(self, n_params=54, n_geo=24, n_experts=3):
        super().__init__()
        self.point_encoder = PointNetEncoder(128)
        self.parameter_encoder = ParameterEncoder(n_params)
        self.geometry_encoder = GeometryEncoder(n_geo)
        self.fusion = nn.Sequential(
            nn.Linear(320,256), nn.ReLU(),
            nn.Linear(256,128), nn.ReLU()
        )
        self.experts = nn.ModuleList([StressExpert(128) for _ in range(n_experts)])
        self.gate = Gate(128, n_experts)

    def forward(self, points, params, geometry, return_details=False):
        p = self.point_encoder(points)
        d = self.parameter_encoder(params)
        g = self.geometry_encoder(geometry)
        shared = self.fusion(torch.cat([p,d,g], dim=1))
        expert_outputs = torch.cat([e(shared) for e in self.experts], dim=1)
        weights = self.gate(shared)
        prediction = (weights * expert_outputs).sum(dim=1)
        if return_details:
            return prediction, weights, expert_outputs
        return prediction

def gate_balance_loss(gate_weights):
    n = gate_weights.shape[1]
    target = torch.full((n,), 1.0/n, device=gate_weights.device,
                        dtype=gate_weights.dtype)
    return ((gate_weights.mean(dim=0) - target)**2).mean()
