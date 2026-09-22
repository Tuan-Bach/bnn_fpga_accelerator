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

## Bit-Matching Verification

100/100 predictions matched against Python/TensorFlow reference model. The verification flow:

1. Train BNN with Larq (Python)
2. Export weights to .mem files
3. Load weights into Verilog testbench
4. Feed same input data to both HW and SW
5. Compare output bit-by-bit
