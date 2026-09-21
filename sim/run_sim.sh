#!/bin/bash
# Simulation script for ModelSim/QuestaSim

set -e

WORK_DIR="work"
TOP_MODULE="tb_bnn_top"
SIM_TIME="100us"

echo "=== BNN FPGA Accelerator Simulation ==="

# Create work library
if [ ! -d "$WORK_DIR" ]; then
    echo "Creating work library..."
    vlib $WORK_DIR
    vmap work $WORK_DIR
fi

# Compile packages first
echo "Compiling packages..."
vlog -work $WORK_DIR ../rtl/bnn_pkg.sv

# Compile RTL modules
echo "Compiling RTL..."
vlog -work $WORK_DIR ../rtl/pe_unit.sv
vlog -work $WORK_DIR ../rtl/pe_array.sv
vlog -work $WORK_DIR ../rtl/weight_rom.sv
vlog -work $WORK_DIR ../rtl/input_buffer.sv
vlog -work $WORK_DIR ../rtl/ctrl_fsm.sv
vlog -work $WORK_DIR ../rtl/axi_lite_slave.sv
vlog -work $WORK_DIR ../rtl/bnn_top.sv

# Compile testbench
echo "Compiling testbench..."
vlog -work $WORK_DIR tb_bnn_top.sv

# Run simulation
echo "Running simulation..."
vsim -c -do "run -all; quit" -voptargs="+acc" $TOP_MODULE

echo "=== Simulation Complete ==="