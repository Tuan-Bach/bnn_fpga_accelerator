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

## End-to-End MNIST Verification (Verilator PE)

Real MNIST images are run through the SystemVerilog PE hardware via `tb/mnist_verify_tb.cpp`. The deep BNN is 784 -> 2048 -> 2048 -> 2048 -> 10.

### Running

```bash
# Train (~1.5h), export weights + calibrate thresholds, build + run simulation
make sim-mnist

# Or, using the saved checkpoint (fast):
cd python && python3 export_bnn_deep.py && cd ..
make -s build-mnist   # or: ./obj_dir_mnist/Vpe_unit (after building)
```

### Results (Verilator PE hardware simulation)

```
Correct: 979 / 1000
Accuracy: 97.90%
```

The 1000-image hardware result matches the software hardware-equivalent pipeline bit-for-bit (979/1000), confirming the PE is an exact bit-exact model of the reference computation.

### How the pipeline works

1. **Training** (`python/train_bnn_deep.py`): PyTorch, 3 binary hidden layers of 2048 neurons.
   - Weights *and* activations binarized to {-1,+1} with the straight-through estimator (STE)
   - BatchNorm before each binarization; cosine-annealed Adam
2. **Export & threshold calibration** (`python/export_bnn_deep.py`):
   - Binarized weights written as 64-bit words, LSB-first (`layer{1,2,3}_weights.mem`)
   - Output layer + bias quantized to Q4.12, 4 weights per 64-bit word (`layer4_weights.mem`, `layer4_bias.mem`)
   - Per-neuron thresholds solved in the **hardware popcount domain**: for each neuron, the midpoint between the mean popcount when the neuron is active vs inactive over a calibration subset, using the golden binary activations as labels
3. **Hardware simulation** (`tb/mnist_verify_tb.cpp`): each neuron's XNOR-popcount is computed by the real PE (13 words for layer 1, 32 words for layers 2-3), thresholds are applied, intermediate activations are re-packed into 64-bit words, and the output layer + argmax complete the prediction.

### Important details for matching hardware exactly

- **Padding bits**: layer 1 input is 784 features packed in 13 x 64-bit words; the final 48 bits are always zero in both data and weights, so they always XNOR-match and add +48 to every neuron's popcount. Calibration runs in this padded domain.
- **Input binarization**: images binarize as `sign(x)` (any nonzero pixel) to match the way the model was trained.
- **Activation packing**: binary activations are packed LSB-first (`activation[j]` -> bit `j%64` of word `j/64`), matching the weight file layout.
- **Q4.12 output layer**: exactly 4 fixed-point weights per 64-bit word; the testbench unpacks with 16-bit fields.

## Bit-Matching Verification

The 1000-image hardware run matched the Python reference bit-for-bit. The verification flow:

1. Train deep BNN with PyTorch
2. Export weights + thresholds to .mem files
3. Load weights into the Verilator PE testbench
4. Feed the same input data to both HW and SW
5. Compare predictions; 979/1000 both, identical per-image results
