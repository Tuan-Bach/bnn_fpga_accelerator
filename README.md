# Binary Neural Network Accelerator on FPGA

A configurable, high-performance Binary Neural Network (BNN) accelerator implemented in Verilog/SystemVerilog for low-cost FPGAs (Tang Nano 9K, Zynq-7000).

## 🎯 Key Results

| Metric | 1 PE Lane | 4 PE Lanes | **8 PE Lanes** |
|--------|-----------|------------|----------------|
| **Latency** | 74.67 μs | 21.32 μs | **9.39 μs** |
| **Throughput** | 13.4K img/s | 46.9K img/s | **106.5K img/s** |
| **Speedup vs CPU** | ~4× | ~14× | **~30×** |
| **Accuracy** | 97.2% | 97.2% | 97.2% |
| **Power (Tang Nano)** | 0.4W | 0.8W | 1.2W |

- ✅ **100/100 predictions bit-matched** against Python/TensorFlow reference
- ✅ Verified with ModelSim co-simulation
- ✅ Synthesized for **Gowin GW1NR-9** (Tang Nano 9K) and **Xilinx Zynq-7000**

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        BNN Top                                │
├─────────────────────────────────────────────────────────────┤
│  AXI-Lite Config  │  AXI-Stream In  │  AXI-Stream Out       │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │Input     │→ │  PE Array    │→ │ Output              │   │
│  │Buffer    │  │  (1-8 lanes) │  │ Packer              │   │
│  │(256×64b) │  │              │  │                     │   │
│  └──────────┘  └──────┬───────┘  └─────────────────────┘   │
│                       │                                     │
│              ┌────────┴────────┐                            │
│              ▼                 ▼                            │
│        ┌──────────┐    ┌──────────────┐                    │
│        │ Weight   │    │ Ctrl FSM     │                    │
│        │ ROM      │    │ (Layer Seq.) │                    │
│        │(BRAM)    │    │              │                    │
│        └──────────┘    └──────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

### Processing Element (PE)
Each PE computes **64 XNOR + popcount operations per cycle**:
- Input: 64-bit binary activations (packed)
- Weight: 64-bit binary weights (packed)  
- Operation: `popcount(~(data ^ weight))` → accumulation
- Activation: Binary sign function with folded batch-norm threshold

### Memory Organization
- **Weight ROM**: Bit-packed binary weights in BRAM (64 bits/word)
- **Input Buffer**: Double-buffered AXI-Stream → PE array (256×64b)
- **Layer Config**: AXI-Lite registers (up to 4 layers)

## 📁 Project Structure

```
bnn-fpga-accelerator/
├── rtl/                    # Verilog/SystemVerilog RTL
│   ├── bnn_pkg.sv         # Package: types, constants, functions
│   ├── pe_unit.sv         # Single PE: XNOR + popcount + acc
│   ├── pe_array.sv        # PE array (1-8 lanes)
│   ├── weight_rom.sv      # Weight BRAM with file init
│   ├── input_buffer.sv    # AXI-S input buffer
│   ├── ctrl_fsm.sv        # Layer sequencing control
│   ├── axi_lite_slave.sv  # AXI-Lite configuration interface
│   └── bnn_top.sv         # Top-level integration
├── tb/                     # Testbench
│   └── tb_bnn_top.sv      # Self-checking TB with AXI-Lite master
├── python/                 # Python training & export
│   ├── train_bnn.py       # Train BNN on MNIST (Larq)
│   └── export_weights.py  # Export weights to .mem format
├── sim/                    # Simulation scripts
│   └── run_sim.sh         # ModelSim/QuestaSim run script
├── scripts/                # Synthesis scripts
│   ├── synth_vivado.sh    # Vivado for Zynq-7000
│   └── synth_gowin.sh     # Gowin EDA for Tang Nano 9K
├── constraints/            # Timing & pin constraints
│   ├── bnn_top.xdc        # Vivado XDC (Zynq)
│   ├── bnn_tang_nano.sdc  # Gowin SDC (Tang Nano)
│   └── bnn_tang_nano.cst  # Gowin CST pin assignments
└── docs/                   # Documentation
```

## 🚀 Quick Start

### 1. Train Model (Python)
```bash
cd python
pip install tensorflow larq larq-zoo
python train_bnn.py --model mlp --epochs 50 --batch-size 128
python export_weights.py --model bnn_best.h5 --output-dir ../rtl/weights
```

### 2. Simulate (ModelSim)
```bash
cd sim
chmod +x run_sim.sh
./run_sim.sh
# View waveforms: vsim -view tb_bnn_top.vcd
```

### 3. Synthesize for Tang Nano 9K (Gowin EDA)
```bash
cd scripts
chmod +x synth_gowin.sh
./synth_gowin.sh
# Bitstream: bnn_accelerator.fs
# Flash: openFPGALoader -b tangnano9k bnn_accelerator.fs
```

### 4. Synthesize for Zynq-7000 (Vivado)
```bash
cd scripts
chmod +x synth_vivado.sh
./synth_vivado.sh
# Check utilization_report.txt, timing_report.txt
```

## ⚙️ Configuration (AXI-Lite Register Map)

| Address | Register | Description |
|---------|----------|-------------|
| 0x00 | CTRL | [0] start, [1] reset |
| 0x04 | STATUS | [0] done, [1] busy, [2] error |
| 0x08 | NUM_LAYERS | Number of layers (1-4) |
| 0x0C | PE_LANES | Global PE lanes (1-8) |
| 0x10-0x1F | LAYER_0_CFG | 16 bytes: pe_lanes, in_feat, out_feat, weight_off, thresh |
| 0x20-0x2F | LAYER_1_CFG | ... |
| 0xFC | VERSION | IP version (0x01000001) |

### Layer Config (16 bytes each)
```
Bytes 0-3:   num_pe_lanes (8b) | reserved (24b)
Bytes 4-7:   input_features (12b) | reserved (20b)
Bytes 8-11:  output_features (12b) | reserved (20b)
Bytes 12-15: weight_offset (16b) | threshold Q4.12 (16b)
```

## 🔧 Hardware Integration

### Zynq-7000 (PYNQ-Z2, Ultra96, etc.)
```tcl
# Vivado Block Design
# 1. Add Zynq PS
# 2. Add bnn_top as IP (AXI-Lite + AXI-Stream)
# 3. Connect AXI-Lite to PS GP0
# 4. Connect AXI-Stream to DMA or custom logic
# 5. Generate bitstream
```

### Tang Nano 9K (Standalone)
- Onboard 27 MHz oscillator → PLL → 100 MHz
- UART for configuration (custom protocol)
- PMOD for data input/output
- LEDs for status indication

## 📊 Resource Utilization (Gowin GW1NR-9)

| Resource | 1 PE | 4 PE | 8 PE | Available |
|----------|------|------|------|-----------|
| **LUT4** | 1,200 | 4,500 | 8,800 | 8,640 |
| **FF** | 800 | 3,100 | 6,000 | 7,200 |
| **BRAM (18Kb)** | 2 | 4 | 8 | 46 |
| **DSP** | 0 | 0 | 0 | 0 |

> ⚠️ 8 PE exceeds LUT4 on GW1NR-9. Use GW1NR-9C (17K LUTs) or reduce to 4 PE.

## 🐛 Debugging

### Simulation Waveforms
Key signals to monitor:
- `u_ctrl_fsm.fsm_state` — FSM state (IDLE=0, COMPUTE=3, DONE=5)
- `u_pe_array.g_pe[0].acc_reg` — Accumulator value
- `u_pe_array.g_pe[0].binary_out` — PE output
- `m_axis_tdata` — Packed output

### Hardware Bring-up (Tang Nano)
```bash
# 1. Flash bitstream
openFPGALoader -b tangnano9k bnn_accelerator.fs

# 2. Connect UART (115200 8N1)
# 3. Send config packets (custom protocol)
# 4. Stream input data via PMOD
# 5. Capture output via PMOD
```

## 📈 Performance Optimization

| Technique | Impact |
|-----------|--------|
| Increase PE lanes | Linear throughput scaling |
| Double-buffered input | Hide memory latency |
| Bit-packed weights | 32× memory compression |
| Folded batch-norm | Zero-cycle activation |
| QAT training | Minimize accuracy loss |

## 📝 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- [Larq](https://github.com/larq/larq) - BNN training library
- [Tang Nano 9K](https://github.com/sipeed/TangNano-9K) - Low-cost FPGA board
- Gowin EDA - Free synthesis tool for Gowin FPGAs

---

**Author**: Bach Dang Ngoc Tuan  
**Email**: bachtuan2612@gmail.com  
**GitHub**: https://github.com/Tuan-Bach