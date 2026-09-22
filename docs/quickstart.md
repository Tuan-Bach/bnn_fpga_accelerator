# Quick Start

## Prerequisites

- **Simulation:** [Verilator](https://www.veripool.org/verilator/) 5.x
- **Synthesis:** [Gowin EDA](https://www.gowinsemi.com/) or [Vivado](https://www.xilinx.com/products/design-tools/vivado.html)
- **ML Training:** Python 3.10+, TensorFlow, [Larq](https://github.com/larq/larq)

## Simulation (Verilator)

```bash
# Lint check all RTL
make lint

# Build and run PE unit tests (8 tests)
make sim-pe
```

## Training (Python)

```bash
cd python
pip install -r ../requirements.txt

# Train BNN on MNIST
python train_bnn.py --model mlp --epochs 50 --batch-size 128

# Export weights to .mem format for BRAM init
python export_weights.py --model bnn_best.h5 --output-dir ../rtl/weights
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
| `make synth-gowin` | Synthesize for Tang Nano 9K |
| `make synth-vivado` | Synthesize for Zynq-7000 |
| `make train` | Train BNN model |
| `make export-weights` | Export weights to .mem |
| `make flash` | Flash bitstream to Tang Nano 9K |
| `make clean` | Clean all build artifacts |
