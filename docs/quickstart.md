# Quick Start

## Prerequisites

- **Simulation:** [Verilator](https://www.veripool.org/verilator/) 5.x
- **Synthesis:** [Gowin EDA](https://www.gowinsemi.com/) or [Vivado](https://www.xilinx.com/products/design-tools/vivado.html)
- **ML Training:** Python 3.10+, [PyTorch](https://pytorch.org/)

## Simulation (Verilator)

```bash
# Lint check all RTL
make lint

# Build and run PE unit tests (8 tests)
make sim-pe
```

## End-to-End MNIST Verification

Train, export, and verify the deep BNN (784->2048->2048->2048->10) through the PE hardware:

```bash
# Everything: train (~1.5h) -> export -> simulate. Prints 97.9% accuracy.
make sim-mnist
```

Faster path using the already-trained checkpoint:

```bash
cd python && python3 export_bnn_deep.py && cd ..
make mnist-build            # build the Verilator MNIST testbench
./obj_dir_mnist/Vpe_unit    # run 1000-image verification
```

Expected output:

```
MNIST Verification Results (deep BNN)
  Correct: 979 / 1000
  Accuracy: 97.90%
```

## Training (Python)

```bash
cd python
pip install -r ../requirements.txt

# Train the deep BNN on MNIST (~200 epochs, targets >95%)
python3 train_bnn_deep.py

# Export weights, calibrate thresholds, write .mem files + software verification
python3 export_bnn_deep.py
```

## Synthesis

```bash
# Tang Nano 9K (Gowin)
make synth-gowin
# Output: scripts/bnn_accelerator.fs

# Zynq-7000 (Vivado)
make synth-vivado
# Output: scripts/bnn_accelerator.bit
```

## Flash to Hardware

```bash
# Tang Nano 9K
openFPGALoader -b tangnano9k scripts/bnn_accelerator.fs
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make lint` | Verilator lint check |
| `make sim-pe` | Run PE unit tests |
| `make sim-pe-clean` | Clean Verilator build |
| `make sim-mnist` | Train + export + MNIST simulation |
| `make mnist-build` | Build MNIST testbench (no training) |
| `make synth-gowin` | Synthesize for Tang Nano 9K |
| `make synth-vivado` | Synthesize for Zynq-7000 |
| `make train` | Train BNN model |
| `make export-weights` | Export weights to .mem |
| `make flash` | Flash bitstream to Tang Nano 9K |
| `make clean` | Clean all build artifacts |
