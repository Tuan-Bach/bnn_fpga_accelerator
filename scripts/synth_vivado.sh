#!/bin/bash
# Vivado synthesis script for Zynq-7000

set -e

PROJECT_NAME="bnn_accelerator"
PART="xc7z020clg400-1"
TOP_MODULE="bnn_top"
RTL_DIR="../rtl"

echo "=== Vivado Synthesis for Zynq-7000 ==="

# Create TCL script for Vivado
cat > synth.tcl << EOF
# Create project
create_project -force $PROJECT_NAME ./$PROJECT_NAME -part $PART

# Add RTL sources
add_files -norecurse $RTL_DIR/bnn_pkg.sv
add_files -norecurse $RTL_DIR/pe_unit.sv
add_files -norecurse $RTL_DIR/pe_array.sv
add_files -norecurse $RTL_DIR/weight_rom.sv
add_files -norecurse $RTL_DIR/input_buffer.sv
add_files -norecurse $RTL_DIR/ctrl_fsm.sv
add_files -norecurse $RTL_DIR/axi_lite_slave.sv
add_files -norecurse $RTL_DIR/bnn_top.sv

# Set top module
set_property top $TOP_MODULE [current_fileset]

# Add constraints
add_files -norecurse ../constraints/bnn_top.xdc

# Synthesis settings
set_property STEPS.SYNTH_DESIGN.ARGS.MORE_OPTIONS {-flatten_hierarchy rebuilt} [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.MORE_OPTIONS {-keep_equivalent_registers} [get_runs synth_1]

# Run synthesis
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# Open synthesized design
open_run synth_1 -name netlist

# Report utilization
report_utilization -hierarchical -file utilization_report.txt

# Report timing
report_timing_summary -file timing_report.txt

# Write checkpoint
write_checkpoint -force $PROJECT_NAME.dcp

# Generate bitstream (optional)
# launch_runs impl_1 -to_step write_bitstream -jobs 4
# wait_on_run impl_1

puts "Synthesis complete!"
EOF

# Run Vivado in batch mode
vivado -mode batch -source synth.tcl -log vivado_synth.log -journal vivado_synth.jou

echo "=== Vivado Synthesis Complete ==="
echo "Check utilization_report.txt and timing_report.txt"