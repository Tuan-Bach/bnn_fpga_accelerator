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

## End-to-End Dataset Verification

Train, export, and verify the BNN (`in->2048->2048->2048->classes`) through the PE hardware for any supported dataset:

```bash
# Everything: train (~2-6h) -> export -> simulate
make sim-dataset DATASET=mnist            # 97.9%
make sim-dataset DATASET=fashion_mnist    # 86.5%
make sim-dataset DATASET=emnist           # 80.8%
make sim-dataset DATASET=cifar10
```

Faster path using the already-trained checkpoint:

```bash
cd python && python3 export_bnn_dataset.py --dataset fashion_mnist && cd ..
make mnist-build              # build the data-driven Verilator testbench once
./obj_dir_mnist/Vpe_unit tb/fashion_mnist_data
```

The testbench reads `model_info.txt` from the dataset directory, so the same binary verifies every dataset — just point it at a different `tb/<dataset>_data/` folder.

Expected output:

```
BNN Verification Results (fashion_mnist)
  Correct: 865 / 1000
  Accuracy: 86.50%
```

## Training (Python)

```bash
cd python
pip install -r ../requirements.txt

# Train the deep BNN on any dataset (~200 epochs)
python3 train_bnn_dataset.py --dataset mnist
python3 train_bnn_dataset.py --dataset fashion_mnist
python3 train_bnn_dataset.py --dataset emnist
python3 train_bnn_dataset.py --dataset cifar10

# Export weights, calibrate thresholds, write .mem files + software verification
python3 export_bnn_dataset.py --dataset <name>
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
| `make sim-dataset DATASET=<name>` | Train + export + simulate a named dataset |
| `make mnist-build` | Build data-driven testbench binary (once) |
| `make synth-gowin` | Synthesize for Tang Nano 9K |
| `make synth-vivado` | Synthesize for Zynq-7000 |
| `make train` | Train BNN model |
| `make export-weights` | Export weights to .mem |
| `make flash` | Flash bitstream to Tang Nano 9K |
| `make clean` | Clean all build artifacts |
