#!/usr/bin/env python3
"""
Dataset-parametric BNN export for Verilog PE hardware.

Exports trained weights, calibrates per-neuron thresholds in the HARDWARE
popcount domain (full 64-bit words including zero padding), and writes
.mem files + model_info.txt consumed by the C++ Verilator testbench.

Usage:
    python3 export_bnn_dataset.py --dataset fashion_mnist
    python3 export_bnn_dataset.py --dataset emnist
    python3 export_bnn_dataset.py --dataset cifar10
"""

import argparse
import json
import os
import sys
import shutil
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from train_bnn_dataset import BNN, load_dataset, binarize_a, binarize_w


def compute_xnor_popcount_padded(x_np, W_np, n_words):
    """XNOR-popcount matching PE hardware (full 64-bit words incl. zero padding).
    Zero padding always matches (0 XNOR 0 = 1), exactly like the PE."""
    n_neurons = W_np.shape[0]
    result = np.zeros((x_np.shape[0], n_neurons), dtype=np.float64)
    for w in range(n_words):
        s = w * 64
        e_x = min(s + 64, x_np.shape[1])
        e_w = min(s + 64, W_np.shape[1])
        xw = x_np[:, s:e_x] @ W_np[:, s:e_w].T
        x_sum = x_np[:, s:e_x].sum(axis=1, keepdims=True)
        w_sum = W_np[:, s:e_w].sum(axis=1)
        # Full-word match count: 64 - x_sum - w_sum + 2*xw (each pad bit matches)
        result += 64 - x_sum - w_sum + 2 * xw
    return result


def calibrate(z_all, a_golden):
    """Per-neuron midpoint between mean popcount when golden-active vs inactive."""
    n_h = z_all.shape[1]
    thr = np.zeros(n_h, dtype=np.float32)
    for h in range(n_h):
        m = a_golden[:, h] > 0.5
        if m.sum() > 0 and (~m).sum() > 0:
            thr[h] = (z_all[m, h].mean() + z_all[~m, h].mean()) / 2.0
        else:
            thr[h] = np.median(z_all[:, h])
    return thr


def emem_bin(W, path, label):
    rows, cols = W.shape
    nw = (cols + 63) // 64
    with open(path, 'w') as f:
        f.write(f"// {label}: {rows}x{cols}, {nw} words/neuron\n")
        for n in range(rows):
            for w in range(nw):
                s = w * 64
                e = min(s + 64, cols)
                val = 0
                for j in range(e - s):
                    if W[n, s + j]:
                        val |= (1 << j)
                f.write(f"{val:016X}\n")


def emem_float(W, path, label):
    """Q4.12, exactly 4 weights of 16 bits per 64-bit word."""
    rows, cols = W.shape
    nw = (cols + 3) // 4
    with open(path, 'w') as f:
        f.write(f"// {label}: {rows}x{cols}, {nw} words/neuron, Q4.12\n")
        for n in range(rows):
            for k in range(nw):
                val = 0
                for j in range(4):
                    idx = k * 4 + j
                    if idx >= cols:
                        break
                    v = int(np.clip(W[n, idx], -8, 7.999) * 4096) & 0xFFFF
                    val |= (v << (j * 16))
                f.write(f"{val:016X}\n")


def ebias(b, path, label):
    with open(path, 'w') as f:
        f.write(f"// {label}: {len(b)} values, Q4.12\n")
        for v in b:
            q = int(np.clip(v, -8, 7.999) * 4096) & 0xFFFF
            f.write(f"{q:04X}\n")


def ethr(t, path, label):
    with open(path, 'w') as f:
        f.write(f"// {label}: {len(t)} values\n")
        for v in t:
            f.write(f"{int(np.clip(v, 0, 65535)) & 0xFFFF:04X}\n")


def eimg(x, path, n_show=1000):
    n = min(n_show, x.shape[0])
    d = x.shape[1]
    nw = (d + 63) // 64
    with open(path, 'w') as f:
        f.write(f"// Test images: {n}x{d}\n")
        for i in range(n):
            b = (x[i] > 0).astype(np.uint8)  # load_dataset already binarized
            for w in range(nw):
                s = w * 64
                e = min(s + 64, d)
                val = 0
                for j in range(e - s):
                    if b[s + j]:
                        val |= (1 << j)
                f.write(f"{val:016X}\n")


def elbl(y, path, n_show=1000):
    n = min(n_show, len(y))
    with open(path, 'w') as f:
        f.write(f"// Labels: {n}\n")
        for v in y[:n]:
            f.write(f"{v:X}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist",
                    choices=["mnist", "fashion_mnist", "emnist", "cifar10", "pcb"])
    ap.add_argument("--cal-n", type=int, default=5000)
    ap.add_argument("--n-show", type=int, default=1000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    here = os.path.dirname(__file__)
    tag = f"_{args.tag}" if args.tag else ""
    cfg_path = os.path.join(here, f"{args.dataset}{tag}_config.json")
    if not os.path.exists(cfg_path):
        sys.exit(f"Missing {cfg_path} - run train_bnn_dataset.py first")
    cfg = json.load(open(cfg_path))
    n_in = cfg["in_features"]
    hidden = cfg["hidden"]
    n_cls = cfg["n_classes"]

    model = BNN(n_in, hidden, n_cls)
    model.load_state_dict(torch.load(
        os.path.join(here, f"best_bnn_{args.dataset}{tag}.pt"), weights_only=True))
    model.eval()

    print(f"Loading {args.dataset}...")
    trn_x, trn_y, tst_x, tst_y = load_dataset(args.dataset)
    print(f"Dataset: {n_in} features, {n_cls} classes, {tst_x.shape[0]} test")

    root = os.path.join(here, '..')
    out_dir = os.path.join(root, 'rtl', 'weights', args.dataset)
    data_dir = (os.path.join(root, 'tb', 'mnist_data') if args.dataset == "mnist"
                else os.path.join(root, 'tb', f"{args.dataset}_data"))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    w1 = (binarize_w(model.fc1.weight).detach().cpu().numpy() > 0).astype(np.float32)
    w2 = (binarize_w(model.fc2.weight).detach().cpu().numpy() > 0).astype(np.float32)
    w3 = (binarize_w(model.fc3.weight).detach().cpu().numpy() > 0).astype(np.float32)
    w4 = model.fc4.weight.detach().cpu().numpy()
    b4 = model.fc4.bias.detach().cpu().numpy()

    n_words_in = (n_in + 63) // 64
    n_words_h = (hidden + 63) // 64
    print(f"Input words: {n_words_in} (pad {n_words_in*64 - n_in}), "
          f"hidden words: {n_words_h}")

    # ---- Golden binary activations (model forward path) ----
    cal_n = min(args.cal_n, len(tst_x))
    x_cal = torch.tensor(tst_x[:cal_n], dtype=torch.float32)
    with torch.no_grad():
        z = model.bn1(F.linear(binarize_a(x_cal), binarize_w(model.fc1.weight)))
        a1 = (z > 0).float()
        z = model.bn2(F.linear(binarize_a(a1), binarize_w(model.fc2.weight)))
        a2 = (z > 0).float()
        z = model.bn3(F.linear(binarize_a(a2), binarize_w(model.fc3.weight)))
        a3 = (z > 0).float()
    a1_np = a1.cpu().numpy(); a2_np = a2.cpu().numpy(); a3_np = a3.cpu().numpy()

    # ---- Calibrate thresholds in hardware popcount domain ----
    def pop_all(x_in, W, n_words):
        out = []
        for s in range(0, cal_n, 500):
            e = min(s + 500, cal_n)
            out.append(compute_xnor_popcount_padded(x_in[s:e], W, n_words))
        return np.concatenate(out)

    print("Layer 1 popcounts...")
    z1 = pop_all(tst_x[:cal_n], w1, n_words_in)
    print("Calibrating layer 1...")
    thr1 = calibrate(z1, a1_np)
    a1_cal = (z1 > thr1).astype(np.float32)

    print("Layer 2 popcounts...")
    z2 = pop_all(a1_cal, w2, n_words_h)
    print("Calibrating layer 2...")
    thr2 = calibrate(z2, a2_np)
    a2_cal = (z2 > thr2).astype(np.float32)

    print("Layer 3 popcounts...")
    z3 = pop_all(a2_cal, w3, n_words_h)
    print("Calibrating layer 3...")
    thr3 = calibrate(z3, a3_np)

    match = ((z1 > thr1).astype(np.float32) == a1_np).mean()
    print(f"Layer 1 threshold match: {match:.4f}")

    # ---- Export ----
    print("Exporting .mem files...")
    emem_bin(w1, f"{out_dir}/layer1_weights.mem", f"Layer1 {n_in}->{hidden}")
    ethr(thr1, f"{out_dir}/layer1_threshold.mem", "Layer1 threshold")
    emem_bin(w2, f"{out_dir}/layer2_weights.mem", f"Layer2 {hidden}->{hidden}")
    ethr(thr2, f"{out_dir}/layer2_threshold.mem", "Layer2 threshold")
    emem_bin(w3, f"{out_dir}/layer3_weights.mem", f"Layer3 {hidden}->{hidden}")
    ethr(thr3, f"{out_dir}/layer3_threshold.mem", "Layer3 threshold")
    emem_float(w4, f"{out_dir}/layer4_weights.mem", f"Layer4 {hidden}->{n_cls}")
    ebias(b4, f"{out_dir}/layer4_bias.mem", "Layer4 output bias")
    eimg(tst_x, f"{data_dir}/test_images.mem", n_show=args.n_show)
    elbl(tst_y, f"{data_dir}/test_labels.mem", n_show=args.n_show)

    with open(f"{data_dir}/model_info.txt", "w") as f:
        f.write(f"dataset {args.dataset}\n")
        f.write(f"n_input {n_in}\n")
        f.write(f"n_hidden 3\n")
        f.write(f"hidden {hidden}\n")
        f.write(f"n_output {n_cls}\n")

    for fn in os.listdir(out_dir):
        if fn.endswith('.mem'):
            shutil.copy2(f"{out_dir}/{fn}", f"{data_dir}/{fn}")

    # ---- Software hardware-equivalent verification ----
    def hw_layer(x_bin, W, thr, n_words):
        return (compute_xnor_popcount_padded(x_bin, W, n_words) > thr).astype(np.float32)

    def quantize_q412(arr):
        """Match the testbench/FPGA Q4.12 fixed-point packing exactly:
        q = int(clip(v,-8,7.999)*4096) & 0xFFFF, reinterpreted int16 /4096.
        """
        q = np.clip(arr, -8.0, 7.999) * 4096.0
        q = (q.astype(np.int32) & 0xFFFF).astype(np.uint16).astype(np.int16)
        return q.astype(np.float32) / 4096.0

    w4_q = quantize_q412(w4)          # (n_cls, hidden), exactly what .mem stores
    b4_q = quantize_q412(b4)          # (n_cls,)

    def hw_scores(a3):
        """Output layer REPLICATING mnist_verify_tb.cpp: float32 accumulation,
        same order (o -> idx ascending), Q4.12 weights + bias. Bit-identical."""
        n = a3.shape[0]
        scores = np.zeros((n, n_cls), dtype=np.float32)
        for o in range(n_cls):
            acc = scores[:, o]
            for idx in range(hidden):
                acc += a3[:, idx] * w4_q[o, idx]
            acc += b4_q[o]
        return scores

    n_show = args.n_show
    xs = tst_x[:n_show]
    ys = tst_y[:n_show]
    a1 = hw_layer(xs, w1, thr1, n_words_in)
    a2 = hw_layer(a1, w2, thr2, n_words_h)
    a3 = hw_layer(a2, w3, thr3, n_words_h)
    pred = hw_scores(a3).argmax(axis=1)
    correct = (pred == ys).sum()
    print(f"\nSoftware HW-equivalent accuracy: {correct}/{n_show} "
          f"= {correct/n_show*100:.2f}%")

    a1 = hw_layer(tst_x, w1, thr1, n_words_in)
    a2 = hw_layer(a1, w2, thr2, n_words_h)
    a3 = hw_layer(a2, w3, thr3, n_words_h)
    pred_all = hw_scores(a3).argmax(axis=1)
    correct_all = (pred_all == tst_y).sum()
    print(f"Software HW-equivalent accuracy (full): {correct_all}/{len(tst_y)} "
          f"= {correct_all/len(tst_y)*100:.2f}%")

    print(f"\nDone. Data in {data_dir}")
    print("Run: ./obj_dir_mnist/Vpe_unit " + data_dir)


if __name__ == '__main__':
    main()