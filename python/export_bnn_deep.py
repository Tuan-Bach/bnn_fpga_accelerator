#!/usr/bin/env python3
"""
Export deep BNN (784-2048-2048-2048-10) weights for Verilog PE hardware.
Calibrates thresholds against golden binary activations.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from train_bnn_deep import BNNDeep, download_mnist, binarize_a, binarize_w


def compute_xnor_popcount_padded(x_np, W_np, n_words):
    """XNOR-popcount matching PE hardware (full 64-bit words incl. zero padding).
    Zero padding always matches (0 XNOR 0 = 1), exactly like the PE."""
    batch = x_np.shape[0]
    n_neurons = W_np.shape[0]
    result = np.zeros((batch, n_neurons), dtype=np.float64)
    for w in range(n_words):
        s = w * 64
        e_x = min(s + 64, x_np.shape[1])
        e_w = min(s + 64, W_np.shape[1])
        # Dot product over valid bits (zeros in pad positions contribute 0)
        xw = x_np[:, s:e_x] @ W_np[:, s:e_w].T
        x_sum = x_np[:, s:e_x].sum(axis=1, keepdims=True)
        w_sum = W_np[:, s:e_w].sum(axis=1)
        # Full-word match count: 64 - x_sum - w_sum + 2*xw
        # The 64 counts every pad bit as a match (both are zero)
        result += 64 - x_sum - w_sum + 2 * xw
    return result


def main():
    ckpt = os.path.join(os.path.dirname(__file__), 'best_bnn_deep.pt')
    model = BNNDeep()
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()

    trn_x, trn_y, tst_x, tst_y = download_mnist()

    root = os.path.join(os.path.dirname(__file__), '..')
    out_dir = os.path.join(root, 'rtl', 'weights')
    data_dir = os.path.join(root, 'tb', 'mnist_data')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Binary weights as {0,1}
    w1 = (binarize_w(model.fc1.weight).detach().cpu().numpy() > 0).astype(np.float32)
    w2 = (binarize_w(model.fc2.weight).detach().cpu().numpy() > 0).astype(np.float32)
    w3 = (binarize_w(model.fc3.weight).detach().cpu().numpy() > 0).astype(np.float32)
    w4 = model.fc4.weight.detach().cpu().numpy()
    b4 = model.fc4.bias.detach().cpu().numpy()

    n_words = {784: 13, 2048: 32}  # ceil(784/64)=13, ceil(2048/64)=32

    # Binarize test images the same way the model does (sign: any nonzero pixel)
    x_bin = (tst_x > 0).astype(np.float32)
    cal_n = min(5000, len(tst_x))
    x_cal = torch.tensor(x_bin[:cal_n], dtype=torch.float32)

    # ---- Get golden activations ----
    with torch.no_grad():
        a = binarize_a(x_cal)
        z = model.bn1(F.linear(binarize_a(a), binarize_w(model.fc1.weight)))
        a1 = (z > 0).float()
        z = model.bn2(F.linear(binarize_a(a1), binarize_w(model.fc2.weight)))
        a2 = (z > 0).float()
        z = model.bn3(F.linear(binarize_a(a2), binarize_w(model.fc3.weight)))
        a3 = (z > 0).float()
        a1_np = a1.cpu().numpy(); a2_np = a2.cpu().numpy(); a3_np = a3.cpu().numpy()

    # ---- Layer 1 raw popcounts ----
    print("Layer 1 popcounts...")
    z1_all = []
    for s in range(0, cal_n, 500):
        e = min(s + 500, cal_n)
        z1_all.append(compute_xnor_popcount_padded(x_bin[s:e], w1, n_words[784]))
    z1_all = np.concatenate(z1_all)

    # ---- Layer 2 raw popcounts (feeds from thresholded layer 1) ----
    # Use calibrated threshold to build binary input for layer 2
    def calibrate(z_all, a_golden):
        n_h = z_all.shape[1]
        thr = np.zeros(n_h, dtype=np.float32)
        for h in range(n_h):
            m = a_golden[:, h] > 0.5
            if m.sum() > 0 and (~m).sum() > 0:
                thr[h] = (z_all[m, h].mean() + z_all[~m, h].mean()) / 2.0
            else:
                thr[h] = np.median(z_all[:, h])
        return thr

    print("Calibrating layer 1...")
    thr1 = calibrate(z1_all, a1_np)
    a1_cal = (z1_all > thr1).astype(np.float32)

    print("Layer 2 popcounts...")
    z2_all = []
    for s in range(0, cal_n, 500):
        e = min(s + 500, cal_n)
        z2_all.append(compute_xnor_popcount_padded(a1_cal[s:e], w2, n_words[2048]))
    z2_all = np.concatenate(z2_all)

    print("Calibrating layer 2...")
    thr2 = calibrate(z2_all, a2_np)
    a2_cal = (z2_all > thr2).astype(np.float32)

    print("Layer 3 popcounts...")
    z3_all = []
    for s in range(0, cal_n, 500):
        e = min(s + 500, cal_n)
        z3_all.append(compute_xnor_popcount_padded(a2_cal[s:e], w3, n_words[2048]))
    z3_all = np.concatenate(z3_all)

    print("Calibrating layer 3...")
    thr3 = calibrate(z3_all, a3_np)

    a1_match = ((z1_all > thr1).astype(np.float32) == a1_np).mean()
    print(f"Layer 1 threshold match: {a1_match:.4f}")

    # ---- Export ----
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

    def emem_float(W, b, path, label):
        rows, cols = W.shape
        nw = (cols + 3) // 4  # 4 weights of 16 bits per 64-bit word
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
                b = (x[i] > 0).astype(np.uint8)
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

    print("Exporting .mem files...")
    emem_bin(w1, f"{out_dir}/layer1_weights.mem", "Layer1 784->2048")
    ethr(thr1, f"{out_dir}/layer1_threshold.mem", "Layer1 threshold")
    emem_bin(w2, f"{out_dir}/layer2_weights.mem", "Layer2 2048->2048")
    ethr(thr2, f"{out_dir}/layer2_threshold.mem", "Layer2 threshold")
    emem_bin(w3, f"{out_dir}/layer3_weights.mem", "Layer3 2048->2048")
    ethr(thr3, f"{out_dir}/layer3_threshold.mem", "Layer3 threshold")
    emem_float(w4, b4, f"{out_dir}/layer4_weights.mem", "Layer4 2048->10")
    ebias(b4, f"{out_dir}/layer4_bias.mem", "Layer4 output bias")
    eimg(tst_x, f"{data_dir}/test_images.mem", n_show=1000)
    elbl(tst_y, f"{data_dir}/test_labels.mem", n_show=1000)

    # Copy to data dir
    import shutil
    for f in os.listdir(out_dir):
        if f.endswith('.mem'):
            shutil.copy2(f"{out_dir}/{f}", f"{data_dir}/{f}")

    # ---- Software hardware-equivalent verification on 1000 images ----
    print("\nVerifying hardware-equivalent pipeline in software...")
    n_show = 1000
    xs = (tst_x[:n_show] > 0).astype(np.float32)
    ys = tst_y[:n_show]

    def hw_layer(x_bin, W, thr, n_words):
        pc = compute_xnor_popcount_padded(x_bin, W, n_words)
        return (pc > thr).astype(np.float32)

    a1 = hw_layer(xs, w1, thr1, n_words[784])
    a2 = hw_layer(a1, w2, thr2, n_words[2048])
    a3 = hw_layer(a2, w3, thr3, n_words[2048])

    # Float output layer
    scores = a3 @ w4.T + b4
    pred = scores.argmax(axis=1)
    correct = (pred == ys).sum()
    print(f"Software HW-equivalent accuracy: {correct}/{n_show} = {correct/n_show*100:.2f}%")

    # Also full 10k accuracy for reference
    xs_all = (tst_x > 0).astype(np.float32)
    a1 = hw_layer(xs_all, w1, thr1, n_words[784])
    a2 = hw_layer(a1, w2, thr2, n_words[2048])
    a3 = hw_layer(a2, w3, thr3, n_words[2048])
    scores = a3 @ w4.T + b4
    pred_all = scores.argmax(axis=1)
    correct_all = (pred_all == tst_y).sum()
    print(f"Software HW-equivalent accuracy (10k): {correct_all}/{len(tst_y)} = {correct_all/len(tst_y)*100:.2f}%")

    print("Done. Next: run Verilator (./obj_dir_mnist/Vpe_unit)")


if __name__ == '__main__':
    main()