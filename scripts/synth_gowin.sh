#!/bin/bash
# Gowin EDA synthesis script for Tang Nano 9K

set -e

PROJECT_NAME="bnn_accelerator"
DEVICE="GW1NR-LV9QN88PC6/I5"
TOP_MODULE="bnn_top"
RTL_DIR="../rtl"

echo "=== Gowin EDA Synthesis for Tang Nano 9K ==="

# Create project file (.gprj)
cat > ${PROJECT_NAME}.gprj << EOF
<?xml version="1.0" encoding="UTF-8"?>
<project version="1.9.9.01">
  <setting>
    <name>$PROJECT_NAME</name>
    <device>$DEVICE</device>
    <top>$TOP_MODULE</top>
    <syn_tool>GowinSynthesis</syn_tool>
    <sim_tool>ModelSim</sim_tool>
  </setting>
  <files>
    <file>$RTL_DIR/bnn_pkg.sv</file>
    <file>$RTL_DIR/pe_unit.sv</file>
    <file>$RTL_DIR/pe_array.sv</file>
    <file>$RTL_DIR/weight_rom.sv</file>
    <file>$RTL_DIR/input_buffer.sv</file>
    <file>$RTL_DIR/ctrl_fsm.sv</file>
    <file>$RTL_DIR/axi_lite_slave.sv</file>
    <file>$RTL_DIR/bnn_top.sv</file>
  </files>
  <constraints>
    <file>../constraints/bnn_tang_nano.sdc</file>
    <file>../constraints/bnn_tang_nano.cst</file>
  </constraints>
</project>
EOF

# Run Gowin CLI synthesis
echo "Running Gowin Synthesis..."
gw_sh -syn ${PROJECT_NAME}.gprj -log syn.log

echo "Running Place & Route..."
gw_sh -pnr ${PROJECT_NAME}.gprj -log pnr.log

echo "Generating bitstream..."
gw_sh -bit ${PROJECT_NAME}.gprj -log bit.log

echo "=== Gowin Synthesis Complete ==="
echo "Bitstream: ${PROJECT_NAME}.fs"
echo "Check syn.log, pnr.log, bit.log for details"