# utils/mlp_utils.py
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

import torch, copy, json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path


# Export torch models for use in MOOSE 
def export_torchscript(model, n_inputs, path, x_check=None, double=True):
    """Export a trained MLP/PointNet (tabular mode) for libtorch."""
    m = copy.deepcopy(model).cpu().eval()
    dtype = torch.float64 if double else torch.float32
    m = m.to(dtype)
    with torch.no_grad():
        ts = torch.jit.trace(m, torch.zeros(1, n_inputs, dtype=dtype))
    ts.save(path)

    if x_check is not None:
        x = torch.tensor(x_check[:5], dtype=dtype)
        with torch.no_grad():
            y = ts(x)
        with open(path + ".check.json", "w") as f:
            json.dump({"x_scaled": x.tolist(), "y_scaled": y.tolist()}, f, indent=2)
    return path

class PointDataset(Dataset):
    """
    Generic point dataset for LAROMance concatenated data.
    Each sample is one node (point) with input features and target outputs.
    """
    def __init__(self, X, y, augment=False):
        """
        X : np.ndarray shape (N, n_inputs)
        y : np.ndarray shape (N, n_outputs)
        """
        self.X = torch.tensor(X, dtype=torch.float64)
        self.y = torch.tensor(y, dtype=torch.float64)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.augment:
            # Light jitter (optional, can be extended)
            x = x + torch.randn_like(x) * 0.01

        return x, y
    

def collate_fn(batch):
    """Simple collate for fixed-size point data (no padding needed)."""
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)


def compute_gate_activity(model, X_scaled):
    """Compute per-point expert gating activity for a Mixture-of-Experts model.

    Parameters
    ----------
    model : torch.nn.Module
        A trained model. Only models exposing a ``gating`` method (i.e.
        ``MoERegressor``) produce output; anything else returns ``None``.
    X_scaled : np.ndarray, shape (N, n_inputs)
        The SCALED inputs (same scaling used for inference), one row per point.

    Returns
    -------
    dict or None
        ``{"weights": (N, E) float array, "argmax": (N,) int array,
           "num_experts": E, "top_k": k}`` for an MoE model, else ``None``.
    """
    if not hasattr(model, "gating"):
        return None
    try:
        model_eval = model.eval()
        with torch.no_grad():
            x = torch.tensor(np.asarray(X_scaled, dtype=np.float64), dtype=torch.float64)
            # Match the model's working dtype/device
            try:
                p = next(model_eval.parameters())
                x = x.to(dtype=p.dtype, device=p.device)
            except StopIteration:
                pass
            weights = model_eval.gating(x).cpu().numpy()
    except Exception as exc:  # never break plotting because of gating
        print(f"[compute_gate_activity] WARNING: gating failed: {exc}")
        return None

    weights = np.asarray(weights)
    if weights.ndim != 2:
        weights = weights.reshape(-1, weights.shape[-1])
    return {
        "weights": weights,
        "argmax": weights.argmax(axis=1).astype(int),
        "num_experts": int(getattr(model, "num_experts", weights.shape[1])),
        "top_k": int(getattr(model, "top_k", 1)),
    }


# # Helper function for gating networks (MoE)
# def load_balance_loss(gate_weights):
#     # Encourage uniform expert usage
#     expert_usage = gate_weights.mean(dim=0)
#     return (expert_usage * torch.log(expert_usage + 1e-8)).sum() 

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def train_mlp(X, y, model, epochs=150, batch_size=1024, lr=1e-3,
              device=None, output_dir="./output", drop_last=True):
    """
    High-level training function for MLP models on LAROMance point data.

    X : np.ndarray (N, n_inputs)
    y : np.ndarray (N, n_outputs)
    model : torch.nn.Module
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PointDataset(X, y, augment=True)
    loader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        shuffle=True,
                        collate_fn=collate_fn,
                        drop_last=drop_last)

    model = model.to(device).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    Path(output_dir).mkdir(exist_ok=True)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1:03d} | train_loss = {train_loss:.6f}")

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
            }, f"{output_dir}/best_mlp_model.pt")

    print(f"\nTraining finished. Best loss: {best_loss:.6f}")

    ckpt = torch.load(f"{output_dir}/best_mlp_model.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Restored best weights from epoch {ckpt['epoch']+1}")

    return model.cpu()