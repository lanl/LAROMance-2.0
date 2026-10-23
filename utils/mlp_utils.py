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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path


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
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
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

    model = model.to(device)
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
    return model