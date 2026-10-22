"""
© 2026. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001
for Los Alamos National Laboratory (LANL), which is operated by Triad
National Security, LLC for the U.S. Department of Energy/National Nuclear
Security Administration. All rights in the program are reserved by Triad
National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others
acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
in this material to reproduce, prepare derivative works, distribute copies
to the public, perform publicly and display publicly, and to permit others
to do so.

==============================================================================================================

@author: Andre Ruybalid 
andreruybalid@gmail.com
-----------------------
"""

import torch
import torch.nn as nn


# ============================================================
# 1. Basic MLP (no global context)
# ============================================================
class MLPRegressor(nn.Module):
    """Simple MLP without global max-pooling."""
    def __init__(self, input_dim, hidden=128, num_layers=3, output_dim=1):
        super().__init__()
        layers = []
        prev = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.LayerNorm(hidden))   # Changed from BatchNorm1d
            layers.append(nn.ReLU())
            prev = hidden
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3:
            B, N, D = x.shape
            x = x.view(B * N, D)
            out = self.net(x)
            return out.view(B, N, -1)
        else:
            # Tabular input [B, D]
            return self.net(x)

# ============================================================
# 2. PointNet-style model with global max-pooling
# ============================================================
class PointNetRegressor(nn.Module):
    """PointNet with global max-pooling (local + global features)."""
    def __init__(self, input_dim, hidden=128, num_layers=3, output_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden

        # Local feature extractor
        layers = []
        prev = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.LayerNorm(hidden))   # Changed from BatchNorm1d
            layers.append(nn.ReLU())
            prev = hidden
        self.local_mlp = nn.Sequential(*layers)

        # Global feature
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),                 # Changed from BatchNorm1d
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2)
        )

        # Final predictor
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden + hidden // 2, hidden),
            nn.LayerNorm(hidden),                 # Changed from BatchNorm1d
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        if x.dim() == 2:
            # Tabular mode — compute both local + global so final_mlp input width is consistent
            local = self.local_mlp(x)                    # [B, hidden]
            global_feat = self.global_mlp(local)         # [B, hidden//2]
            combined = torch.cat([local, global_feat], dim=-1)  # [B, hidden + hidden//2]
            out = self.final_mlp(combined)
            return out
        else:
            # Point cloud mode
            B, N, D = x.shape
            x_flat = x.view(B * N, D)
            local = self.local_mlp(x_flat).view(B, N, -1)

            global_feat, _ = torch.max(local, dim=1)
            global_feat = self.global_mlp(global_feat)
            global_feat = global_feat.unsqueeze(1).expand(-1, N, -1)

            combined = torch.cat([local, global_feat], dim=-1)
            out = self.final_mlp(combined.view(B * N, -1))
            return out.view(B, N, -1)

# ============================================================
# 3. Model Factory
# ============================================================
MODEL_REGISTRY = {
    "mlp": MLPRegressor,
    "pointnet": PointNetRegressor,
}

def get_model(model_name, input_dim, hidden=128, output_dim=1):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name](input_dim=input_dim, hidden=hidden, output_dim=output_dim)