#!/usr/bin/env python3
"""
Dataset-parametric deep BNN training (PyTorch).
Supports: mnist, fashion_mnist, emnist (balanced), cifar10.

Architecture: {in_features} -> hidden^3 -> {n_classes}
- Binary weights and activations ({-1,+1}) via straight-through estimator
- BatchNorm before each binarization
- Cosine-annealed Adam

Usage:
    python3 train_bnn_dataset.py --dataset fashion_mnist [--hidden 2048] [--epochs 200]
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from torchvision import datasets, transforms
except ImportError:
    datasets = None


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def dataset_config(name):
    """Per-dataset input binarization policy.
    - mnist / fashion_mnist: black background (0), ink > 0
    - emnist: stored inverted (white background) -> invert, then > 0.5
    - cifar10: dense RGB -> threshold at > 0.5 (sign() on [0,1] is too lossy)
    - pcb: pre-binarized patches (Otsu), stored already {0,1} -> no-op
    """
    if name in ("mnist", "fashion_mnist"):
        return {"bin_threshold": 0.0, "invert": False}
    if name == "emnist":
        return {"bin_threshold": 0.5, "invert": True}
    if name == "cifar10":
        return {"bin_threshold": 0.5, "invert": False}
    if name == "pcb":
        return {"bin_threshold": 0.0, "invert": False}
    raise ValueError(f"Unknown dataset {name}")


def load_dataset(name):
    if name == "pcb":
        # Pre-binarized patch data produced by prepare_pcb.py (already {0,1})
        p = "/tmp/bnn_datasets/pcb"
        trn_x = np.load(f"{p}/train_x.npy").astype(np.float32)
        trn_y = np.load(f"{p}/train_y.npy").astype(np.int64)
        tst_x = np.load(f"{p}/test_x.npy").astype(np.float32)
        tst_y = np.load(f"{p}/test_y.npy").astype(np.int64)
        return trn_x, trn_y, tst_x, tst_y
    if datasets is None:
        raise RuntimeError("torchvision required for dataset loading")
    root = "/tmp/bnn_datasets"
    os.makedirs(root, exist_ok=True)
    tf = transforms.Compose([transforms.ToTensor()])
    if name == "mnist":
        trn = datasets.MNIST(root, train=True, download=True, transform=tf)
        tst = datasets.MNIST(root, train=False, download=True, transform=tf)
    elif name == "fashion_mnist":
        trn = datasets.FashionMNIST(root, train=True, download=True, transform=tf)
        tst = datasets.FashionMNIST(root, train=False, download=True, transform=tf)
    elif name == "emnist":
        trn = datasets.EMNIST(root, split="balanced", train=True, download=True, transform=tf)
        tst = datasets.EMNIST(root, split="balanced", train=False, download=True, transform=tf)
    elif name == "cifar10":
        trn = datasets.CIFAR10(root, train=True, download=True, transform=tf)
        tst = datasets.CIFAR10(root, train=False, download=True, transform=tf)
    else:
        raise ValueError(f"Unknown dataset {name}")

    cfg = dataset_config(name)

    def to_np(ds):
        xs, ys = [], []
        for x, y in ds:
            xs.append(x.numpy().flatten())
            ys.append(y)
        arr = np.stack(xs).astype(np.float32)
        if cfg["invert"]:
            arr = 1.0 - arr
        # Hard-binarize inputs {0,1} up front so train/inference/hardware
        # all see identical patterns (model binarizes again with sign()).
        arr = (arr > cfg["bin_threshold"]).astype(np.float32)
        return arr, np.array(ys, dtype=np.int64)

    trn_x, trn_y = to_np(trn)
    tst_x, tst_y = to_np(tst)
    return trn_x, trn_y, tst_x, tst_y


# ------------------------------------------------------------------
# BNN layers
# ------------------------------------------------------------------

class BinAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        gi = grad_output.clone()
        gi[x.ge(1)] = 0
        gi[x.le(-1)] = 0
        return gi


class BinW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


binarize_a = BinAct.apply
binarize_w = BinW.apply


class BNN(nn.Module):
    """{in_features} -> hidden^3 -> n_classes with binary hidden layers."""

    def __init__(self, in_features, hidden, n_classes):
        super().__init__()
        self.in_features = in_features
        self.hidden = hidden
        self.n_classes = n_classes
        self.fc1 = nn.Linear(in_features, hidden, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc3 = nn.Linear(hidden, hidden, bias=False)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.fc4 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = x.view(-1, self.in_features)
        z = self.bn1(F.linear(binarize_a(x), binarize_w(self.fc1.weight)))
        x = F.hardtanh(z)
        z = self.bn2(F.linear(binarize_a(x), binarize_w(self.fc2.weight)))
        x = F.hardtanh(z)
        z = self.bn3(F.linear(binarize_a(x), binarize_w(self.fc3.weight)))
        x = F.hardtanh(z)
        return self.fc4(x)

    def binarized_forward(self, x):
        return self.forward(x)


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = total = correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
        correct += (out.argmax(1) == y).sum().item()
        total += len(x)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model.binarized_forward(x).argmax(1) == y).sum().item()
        total += len(x)
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fashion_mnist",
                    choices=["mnist", "fashion_mnist", "emnist", "cifar10", "pcb"])
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = "cpu"
    print(f"Loading {args.dataset}...")
    trn_x, trn_y, tst_x, tst_y = load_dataset(args.dataset)
    print(f"Train: {trn_x.shape}, Test: {tst_x.shape}, classes: {len(np.unique(trn_y))}")

    n_in = trn_x.shape[1]
    n_cls = len(np.unique(trn_y))
    model = BNN(n_in, args.hidden, n_cls).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_in}->{args.hidden}x3->{n_cls} ({n_params:,} params)")

    trn_ds = TensorDataset(torch.tensor(trn_x), torch.tensor(trn_y))
    tst_ds = TensorDataset(torch.tensor(tst_x), torch.tensor(tst_y))
    trn_loader = DataLoader(trn_ds, batch_size=args.batch, shuffle=True)
    tst_loader = DataLoader(tst_ds, batch_size=1024, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best = 0.0
    tag = f"_{args.tag}" if args.tag else ""
    ckpt = os.path.join(os.path.dirname(__file__), f"best_bnn_{args.dataset}{tag}.pt")
    cfg = {"in_features": n_in, "hidden": args.hidden, "n_classes": n_cls,
           "bin_threshold": dataset_config(args.dataset)["bin_threshold"],
           "invert": dataset_config(args.dataset)["invert"]}
    with open(os.path.join(os.path.dirname(__file__), f"{args.dataset}{tag}_config.json"), "w") as f:
        json.dump(cfg, f)

    print("Training...")
    for epoch in range(args.epochs):
        loss, acc = train_epoch(model, trn_loader, optimizer, device)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            test_acc = evaluate(model, tst_loader, device)
            if test_acc > best:
                best = test_acc
                torch.save(model.state_dict(), ckpt)
            print(f"Epoch {epoch+1:3d}/{args.epochs} loss={loss:.4f} "
                  f"train={acc:.4f} test={test_acc:.4f} best={best:.4f}")
        scheduler.step()

    print(f"\nBest: {best:.4f} -> {ckpt}")


if __name__ == "__main__":
    main()