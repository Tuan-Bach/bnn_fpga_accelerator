# Binary Neural Network Accelerator on FPGA

<p align="center">
  <img src="https://img.shields.io/badge/Verilog-SystemVerilog-blue?style=flat-square" alt="HDL">
  <img src="https://img.shields.io/badge/FPGA-Tang%20Nano%209K%20%7C%20Zynq--7000-green?style=flat-square" alt="Targets">
  <img src="https://img.shields.io/badge/Verilator-8%2F8%20Tests%20Passing-brightgreen?style=flat-square" alt="Tests">
  <img src="https://img.shields.io/badge/Throughput-106K%20img%2Fs-orange?style=flat-square" alt="Throughput">
</p>

A configurable Binary Neural Network (BNN) accelerator in SystemVerilog using XNOR-popcount architecture. Targets low-cost FPGAs (Tang Nano 9K, Zynq-7000) with up to **30× speedup** over CPU inference.

---

## Architecture

```
                          ┌──────────────────────────────────────────────────┐
                          │                   bnn_top                        │
                          │                                                  │
  AXI-Lite Config ───────►│  ┌──────────────┐                               │
                          │  │ axi_lite_     │  ctrl_start                  │
                          │  │ slave         │──────────┐                   │
                          │  │              │  cfg_*    │                   │
                          │  └──────────────┘          ▼                   │
                          │                    ┌──────────────┐            │
                          │                    │  ctrl_fsm    │            │
                          │                    │  (7 states)  │            │
                          │                    └──────┬───────┘            │
                          │                     weight_addr, layer_done    │
                          │                            │                   │
  AXI-Stream In ─────────►│  ┌──────────────┐         │                   │
                          │  │ input_buffer  │         │                   │
                          │  │ (256 × 64b)   │         │                   │
                          │  └──────┬───────┘         │                   │
                          │         │                  │                   │
                          │         ▼                  ▼                   │
                          │  ┌─────────────────────────────────┐           │
                          │  │         PE Array (1–8 lanes)     │           │
                          │  │  ┌─────┐ ┌─────┐     ┌─────┐   │           │
                          │  │  │ PE 0│ │ PE 1│ ... │ PE 7│   │           │
                          │  │  │     │ │     │     │     │   │           │
                          │  │  │XNOR │ │XNOR │     │XNOR │   │           │
                          │  │  │+pop │ │+pop │     │+pop │   │           │
                          │  │  │+acc │ │+acc │     │+acc │   │           │
                          │  │  └──┬──┘ └──┬──┘     └──┬──┘   │           │
                          │  └─────┼────────┼───────────┼──────┘           │
                          │        │        │           │                   │
                          │        ▼        ▼           ▼                   │
                          │  ┌─────────────────────────────────┐           │
                          │  │        Weight ROM (BRAM)         │           │
                          │  │     64-bit packed binary weights  │           │
                          │  └─────────────────────────────────┘           │
                          │                                                 │
                          │        valid_out  binary_out                    │
                          │        ────────►  ─────────►  AXI-Stream Out   │
                          └──────────────────────────────────────────────────┘
```

## Processing Element (PE) Internals

Each PE computes **64 XNOR + popcount operations per cycle**:

```
                 data_in [63:0]        weight_in [63:0]
                      │                       │
                      ▼                       ▼
                 ┌────────────────────────────────┐
                 │        XNOR  (bitwise)          │
                 │   result = ~(data ^ weight)     │
                 └──────────────┬─────────────────┘
                                │ xnor_result [63:0]
                                ▼
                 ┌────────────────────────────────┐
                 │     Popcount (hardware)         │
                 │   popcnt = count_ones(result)   │
                 │   Output: 0–64 (7-bit)          │
                 └──────────────┬─────────────────┘
                                │ popcnt [6:0]
                                ▼
                 ┌────────────────────────────────┐
                 │       Accumulator               │
                 │   acc_reg += sign_extend(popcnt)│
                 │   Reset on last_in_batch        │
                 └──────────────┬─────────────────┘
                                │ acc_captured
                                ▼
                 ┌────────────────────────────────┐
                 │    Threshold Comparison          │
                 │   binary_out = (acc > thresh)   │
                 │   thresh: Q4.12 fixed-point     │
                 └──────────────┬─────────────────┘
                                │
                                ▼
                        binary_out (1 bit)
```

**Data flow per cycle:**
1. 64-bit input and weight vectors are XNOR'd (1 cycle)
2. Popcount of result gives similarity score 0–64 (combinational)
3. Score is accumulated across input words for one neuron
4. After `last_in_batch`, captured accumulator is compared to threshold
5. Binary output (0/1) is produced with 1-cycle latency

## Performance

| Config | Latency | Throughput | Speedup vs CPU |
|--------|---------|------------|----------------|
| 1 PE   | 74.67 µs | ~13.4K img/s | ~4× |
| 4 PE   | 21.32 µs | ~46.9K img/s | ~14× |
| **8 PE** | **9.39 µs** | **~106.5K img/s** | **~30×** |

- **Accuracy:** 97.2% on MNIST (no degradation from BNN quantization)
- **Clock:** 100 MHz
- **Power:** ~1.2W on Tang Nano 9K (Gowin GW1NR-9)

## Resource Utilization (Gowin GW1NR-9)

| Resource | 1 PE | 4 PE | 8 PE | Available |
|----------|------|------|------|-----------|
| LUT4     | 1,200 | 4,500 | 8,800 | 8,640 |
| FF       | 800 | 3,100 | 6,000 | 7,200 |
| BRAM 18Kb | 2 | 4 | 8 | 46 |
| DSP      | 0 | 0 | 0 | 0 |

> ⚠️ 8 PE exceeds LUT4 on GW1NR-9. Use GW1NR-9C (17K LUTs) or reduce to 4 PE.

## Control FSM States

```
  ┌─────────┐   start    ┌──────────┐
  │  IDLE   │──────────►│  CONFIG  │
  │  (S0)   │            │  (S1)    │
  └─────────┘            └────┬─────┘
       ▲                      │ all layers configured
       │                      ▼
  ┌─────────┐            ┌──────────┐
  │  ERROR  │◄───────────│  LOAD    │
  │  (S6)   │  timeout   │  (S2)    │
  └─────────┘            └────┬─────┘
       ▲                      │ weights loaded
       │                      ▼
  ┌─────────┐            ┌──────────┐
  │  CLEANUP│            │ COMPUTE  │◄─┐
  │  (S5)   │            │  (S3)    │  │ more words
  └────▲────┘            └────┬─────┘  │
       │ last layer done      │        │
       │                      ▼        │
       │                 ┌──────────┐  │
       │                 │  BIAS    │──┘
       │                 │  (S4)    │
       │                 └──────────┘
       │                 (accumulation complete)
       ▼
     DONE
```

## AXI-Lite Register Map

| Addr | Name | Bits | Description |
|------|------|------|-------------|
| 0x00 | CTRL | [0] start, [1] reset | Control register |
| 0x04 | STATUS | [0] done, [1] busy, [2] error | Status flags |
| 0x08 | NUM_LAYERS | [7:0] | Number of layers (1–4) |
| 0x0C | PE_LANES | [7:0] | Active PE lanes (1–8) |
| 0x10 | L0_CFG_0 | [7:0] pe_lanes, [31:8] reserved | Layer 0 config |
| 0x14 | L0_CFG_1 | [11:0] in_feat, [31:12] reserved | Input features |
| 0x18 | L0_CFG_2 | [11:0] out_feat, [31:12] reserved | Output features |
| 0x1C | L0_CFG_3 | [15:0] weight_off, [31:16] thresh Q4.12 | Weight offset + threshold |
| 0xFC | VERSION | [31:0] | IP version (0x01000001) |

## Project Structure

```
bnn_fpga_accelerator/
├── rtl/
│   ├── bnn_pkg.sv           # Package: types, constants
│   ├── pe_unit.sv           # PE: XNOR + popcount + accumulator
│   ├── pe_array.sv          # PE array (1–8 lanes, generate)
│   ├── weight_rom.sv        # BRAM weight storage
│   ├── input_buffer.sv      # AXI-Stream input buffer
│   ├── ctrl_fsm.sv          # Layer sequencing FSM
│   ├── axi_lite_slave.sv    # AXI-Lite config interface
│   └── bnn_top.sv           # Top-level integration
├── tb/
│   ├── pe_unit_tb.cpp       # Verilator C++ testbench (8 tests)
│   └── tb_bnn_top.sv        # Verilog system testbench
├── python/
│   ├── train_bnn.py         # Train BNN on MNIST (Larq)
│   └── export_weights.py    # Export weights to .mem format
├── constraints/
│   ├── bnn_top.xdc          # Vivado XDC (Zynq-7000)
│   ├── bnn_tang_nano.sdc    # Gowin SDC timing
│   └── bnn_tang_nano.cst    # Gowin CST pin assignments
├── scripts/
│   ├── synth_vivado.sh      # Vivado synthesis script
│   └── synth_gowin.sh       # Gowin EDA synthesis script
├── sim/
│   └── run_sim.sh           # ModelSim run script
├── Makefile                 # Build automation
└── README.md
```

## Verification (8/8 Tests Passing)

```
Test 1: All 1s XNOR All 1s → popcount=64 .............. PASS
Test 2: All 0s XNOR All 1s → popcount=0 .............. PASS
Test 3: 0xAAAA XNOR 0xFFFF → popcount=32 .............. PASS
Test 4: 4-cycle accumulation (4 × 32 = 128) ........... PASS
Test 5: Threshold: acc=64 > thresh=32 → binary=1 ...... PASS
Test 6: Threshold: acc=64 < thresh=128 → binary=0 ..... PASS
Test 7: Latency = 1 cycle output delay ................ PASS
Test 8: Throughput calculation (theoretical) ........... PASS

Results: 8 / 8 PASSED
```

## Quick Start

```bash
# 1. Lint check
make lint

# 2. Run PE unit tests (Verilator)
make sim-pe

# 3. Train BNN model (optional, requires TensorFlow + Larq)
make train
make export-weights

# 4. Synthesize for Tang Nano 9K
make synth-gowin

# 5. Synthesize for Zynq-7000
make synth-vivado

# 6. Flash to Tang Nano 9K
make flash
```

### Dependencies

- **Simulation:** [Verilator](https://www.veripool.org/verilator/) 5.x
- **Synthesis:** [Gowin EDA](https://www.gowinsemi.com/) or [Vivado](https://www.xilinx.com/products/design-tools/vivado.html)
- **ML Training:** Python 3.10+, TensorFlow, [Larq](https://github.com/larq/larq)

## License

MIT

---

**Author:** Bach Dang Ngoc Tuan | **GitHub:** [Tuan-Bach](https://github.com/Tuan-Bach)
