#!/usr/bin/env python3
"""
Deep BNN for MNIST - 784-2048-2048-2048-10
Targets >95% accuracy
"""

import os
import struct
import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from urllib.request import urlretrieve


class BinAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # Gradient through where |x| < 1
        grad_input = grad_output.clone()
        grad_input[x.ge(1)] = 0
        grad_input[x.le(-1)] = 0
        return grad_input


class BinW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.sign()
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


binarize_a = BinAct.apply
binarize_w = BinW.apply


def download_mnist():
    os.makedirs("/tmp/mnist", exist_ok=True)
    base = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    for f in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
              "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
        p = f"/tmp/mnist/{f}"
        if not os.path.exists(p):
            urlretrieve(base + f, p)
    def imgs(path):
        with gzip.open(path, 'rb') as fp:
            _, n, r, c = struct.unpack('>IIII', fp.read(16))
            return np.frombuffer(fp.read(), np.uint8).reshape(n, r*c).astype(np.float32) / 255.0
    def lbls(path):
        with gzip.open(path, 'rb') as fp:
            fp.read(8)
            return np.frombuffer(fp.read(), np.uint8)
    return (imgs("/tmp/mnist/train-images-idx3-ubyte.gz"),
            lbls("/tmp/mnist/train-labels-idx1-ubyte.gz"),
            imgs("/tmp/mnist/t10k-images-idx3-ubyte.gz"),
            lbls("/tmp/mnist/t10k-labels-idx1-ubyte.gz"))


class BNNDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 2048, bias=False)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 2048, bias=False)
        self.bn2 = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 2048, bias=False)
        self.bn3 = nn.BatchNorm1d(2048)
        self.fc4 = nn.Linear(2048, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        # Layer 1
        x_bin = binarize_a(x)
        w_bin = binarize_w(self.fc1.weight)
        x = F.linear(x_bin, w_bin)
        x = self.bn1(x)
        x = F.hardtanh(x)
        # Layer 2
        x_bin = binarize_a(x)
        w_bin = binarize_w(self.fc2.weight)
        x = F.linear(x_bin, w_bin)
        x = self.bn2(x)
        x = F.hardtanh(x)
        # Layer 3
        x_bin = binarize_a(x)
        w_bin = binarize_w(self.fc3.weight)
        x = F.linear(x_bin, w_bin)
        x = self.bn3(x)
        x = F.hardtanh(x)
        # Output
        return self.fc4(x)
    
    def binarized_forward(self, x):
        return self.forward(x)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0; correct = 0; total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*len(x)
        correct += (out.argmax(1)==y).sum().item()
        total += len(x)
    return total_loss/total, correct/total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0; total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model.binarized_forward(x)
        correct += (out.argmax(1)==y).sum().item()
        total += len(x)
    return correct/total


def main():
    device = 'cpu'
    print('Loading MNIST...')
    trn_x, trn_y, tst_x, tst_y = download_mnist()
    trn_ds = TensorDataset(torch.tensor(trn_x), torch.tensor(trn_y))
    tst_ds = TensorDataset(torch.tensor(tst_x), torch.tensor(tst_y))
    trn_loader = DataLoader(trn_ds, batch_size=256, shuffle=True)
    tst_loader = DataLoader(tst_ds, batch_size=1024, shuffle=False)
    
    model = BNNDeep().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    best = 0
    print('Training...')
    for epoch in range(200):
        loss, acc = train_epoch(model, trn_loader, optimizer, device)
        if (epoch+1) % 10 == 0:
            test_acc = evaluate(model, tst_loader, device)
            if test_acc > best:
                best = test_acc
                torch.save(model.state_dict(), 'best_bnn_deep.pt')
            print(f'Epoch {epoch+1:3d} loss={loss:.4f} acc={acc:.4f} test={test_acc:.4f} best={best:.4f}')
        scheduler.step()
    print(f'Best: {best:.4f}')


if __name__ == '__main__':
    main()
