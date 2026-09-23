# Verification

## PE Unit Tests (Verilator)

8/8 tests passing, covering core XNOR-popcount-accumulator functionality.

### Running

```bash
make sim-pe
```

### Test Results

```
Test 1: All 1s XNOR All 1s -> popcount=64 .............. PASS
Test 2: All 0s XNOR All 1s -> popcount=0 .............. PASS
Test 3: 0xAAAA XNOR 0xFFFF -> popcount=32 .............. PASS
Test 4: 4-cycle accumulation (4 x 32 = 128) ........... PASS
Test 5: Threshold: acc=64 > thresh=32 -> binary=1 ...... PASS
Test 6: Threshold: acc=64 < thresh=128 -> binary=0 ..... PASS
Test 7: Latency = 1 cycle output delay ................ PASS
Test 8: Throughput calculation (theoretical) ........... PASS

Results: 8 / 8 PASSED
```

### Test Descriptions

| Test | Input | Expected | Verifies |
|------|-------|----------|----------|
| 1 | data=0xFF..FF, weight=0xFF..FF | acc=64, binary=1 | Full match XNOR popcount |
| 2 | data=0x00..00, weight=0xFF..FF | acc=0, binary=0 | No match XNOR popcount |
| 3 | data=0xAA..AA, weight=0xFF..FF | acc=32 | Half-match popcount |
| 4 | 4 cycles of popcount=32 | acc=128 | Multi-cycle accumulation |
| 5 | acc=64, thresh=32 | binary=1 | Threshold < acc |
| 6 | acc=64, thresh=128 | binary=0 | Threshold > acc |
| 7 | Single cycle | valid_out=1 after 1 cycle | 1-cycle output latency |
| 8 | Theoretical calc | Throughput numbers | Performance model |

### Testbench Files

- `tb/pe_unit_tb.cpp` - Verilator C++ testbench
- `tb/tb_bnn_top.sv` - System-level Verilog testbench (ModelSim)

### Waveform Debug

```bash
# After running sim-pe, view waveforms:
gtkwave obj_dir/pe_unit.vcd
```

Key signals to watch:
- `acc_reg` - Running accumulator value
- `acc_captured` - Latched value at batch end
- `valid_out` - Output valid (1-cycle latency)
- `binary_out` - Threshold comparison result
- `popcnt_result` - Current cycle popcount

## System-Level Testbench

The `tb/tb_bnn_top.sv` testbench verifies full system integration:
- AXI-Lite register configuration
- AXI-Stream input data feeding
- Multi-layer inference execution
- Output collection and validation

Requires ModelSim/QuestaSim:
```bash
cd sim
./run_sim.sh
```

## End-to-End Dataset Verification (Verilator PE)

Real test images for **MNIST, Fashion-MNIST, EMNIST (balanced), CIFAR-10, and PCB-AOI defect patches** are run through the SystemVerilog PE hardware via `tb/mnist_verify_tb.cpp`. The BNN topology is `in_features -> hidden^3 -> classes`, configured at runtime by `model_info.txt` — one testbench binary verifies every dataset and every hidden width.

### Running

```bash
# Train (~2-6h per dataset), export weights, calibrate thresholds, build + simulate
make sim-dataset DATASET=emnist
make sim-dataset DATASET=fashion_mnist
make sim-dataset DATASET=cifar10

# Or, using a saved checkpoint (fast):
cd python && python3 export_bnn_dataset.py --dataset fashion_mnist && cd ..
./obj_dir_mnist/Vpe_unit tb/fashion_mnist_data
```

The data-driven testbench reads `model_info.txt` (input size, hidden width, class count) from the dataset directory, so switching datasets never requires a rebuild.

### Results (Verilator PE hardware simulation)

| Dataset | Classes | Float training | PE hardware (1000 images) | Software HW-eq (full set) | Bit-exact |
|---------|---------|----------------|---------------------------|---------------------------|-----------|
| MNIST | 10 | 97.98% | **97.90% (979/1000)** | 97.33% (10k) | ✅ |
| Fashion-MNIST | 10 | 88.16% | **86.50% (865/1000)** | 86.41% (10k) | ✅ |
| EMNIST (balanced) | 47 | 84.44% | **80.80% (808/1000)** | 78.64% (18.8k) | ✅ |
| CIFAR-10 | 10 | 45.22% | **29.10% (291/1000)** | 30.40% (10k) | ✅ |
| PCB-AOI 2048³ | 6 | 59.40% | **55.50% (555/1000)** | 54.14% (1.16k) | ✅ |
| PCB-AOI 1024³ | 6 | 60.00% | **56.80% (568/1000)** | 55.34% (1.16k) | ✅ |
| PCB-AOI 512³ | 6 | 56.55% | **52.70% (527/1000)** | 51.72% (1.16k) | ✅ |
| PCB-AOI 384³ | 6 | 54.40% | **53.20% (532/1000)** | 51.21% (1.16k) | ✅ |
| PCB-AOI 320³ | 6 | 55.69% | **50.80% (508/1000)** | 50.00% (1.16k) | ✅ |

\* The PCB-AOI rows are the hidden-width sweep (2048→320, same 1,160-patch test set); each row reports its own float training best.

Each 1000-image hardware run matches the software hardware-equivalent pipeline bit-for-bit, confirming the PE is an exact model of the reference XNOR-popcount computation.

### How the pipeline works

1. **Training** (`python/train_bnn_dataset.py --dataset <name> [--hidden W] [--tag t]`): PyTorch, 3 binary hidden layers of width W (default 2048; the PCB sweep used 320–2048).
   - Weights *and* activations binarized to {-1,+1} with the straight-through estimator (STE)
   - BatchNorm before each binarization; cosine-annealed Adam; input binarization policy is per-dataset and shared with export
2. **Export & threshold calibration** (`python/export_bnn_dataset.py --dataset <name>`):
   - Binarized weights written as 64-bit words, LSB-first (`layer{1,2,3}_weights.mem`)
   - Output layer + bias quantized to Q4.12, 4 weights per 64-bit word (`layer4_weights.mem`, `layer4_bias.mem`)
   - Per-neuron thresholds solved in the **hardware popcount domain**: for each neuron, the midpoint between the mean popcount when the neuron is active vs inactive over a calibration subset, using the golden binary activations as labels
   - Writes `model_info.txt` + test images/labels into `tb/<dataset>_data/`
3. **Hardware simulation** (`tb/mnist_verify_tb.cpp`): each neuron's XNOR-popcount is computed by the real PE (25 words/layer-1 input for PCB's 1600 features, 48 words for CIFAR's 3072, 13 for 784 — and W/64 words for each hidden layer at width W), thresholds are applied, intermediate activations are re-packed into 64-bit words, and the output layer + argmax complete the prediction.

### Important details for matching hardware exactly

- **Padding bits**: inputs are packed into full 64-bit words; the trailing bits are always zero in both data and weights, so they always XNOR-match and add to every neuron's popcount. Calibration runs in this padded domain.
- **Input binarization matches training exactly**: MNIST/Fashion binarize any-nonzero pixel; EMNIST is stored inverted (white background) so it is inverted then thresholded at >0.5; CIFAR-10 (dense RGB) is thresholded at >0.5; PCB-AOI patches are Otsu-binarized per patch in `prepare_pcb.py`. The loader in `train_bnn_dataset.py` and the exporter share one policy (`dataset_config()`).
- **Activation packing**: binary activations are packed LSB-first (`activation[j]` -> bit `j%64` of word `j/64`), matching the weight file layout.
- **Q4.12 output layer**: exactly 4 fixed-point weights per 64-bit word; the testbench unpacks with 16-bit fields.

## Bit-Matching Verification

Every 1000-image hardware run matched the software reference bit-for-bit for all verified datasets: MNIST 979, Fashion-MNIST 865, EMNIST 808, CIFAR-10 291, and PCB-AOI 555/568/527/532/508 across the 2048/1024/512/384/320³ widths — identical per-image predictions between the Verilator PE run and the Python reference. The software reference replicates the testbench's output layer exactly (Q4.12 weights/bias, same float32 accumulation order) so equivalence holds up to hard near-ties. The verification flow:

1. Train deep BNN with PyTorch
2. Export weights + thresholds to .mem files
3. Load weights into the Verilator PE testbench
4. Feed the same input data to both HW and SW
5. Compare predictions; identical per-image results, dataset after dataset
