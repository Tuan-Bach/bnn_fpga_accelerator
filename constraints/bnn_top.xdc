## XDC Constraints for Zynq-7000 (Artix-7)
## Target: xc7z020clg400-1

# Clock constraint - 100 MHz
create_clock -period 10.000 -name clk [get_ports clk]

# Reset - asynchronous, active low
set_property ASYNC_REG TRUE [get_cells -hierarchical *rst*]

# Input/Output delays for AXI-Lite (assume 100 MHz clock)
set_input_delay -clock clk -min 1.0 [get_ports s_axi_*]
set_input_delay -clock clk -max 3.0 [get_ports s_axi_*]
set_output_delay -clock clk -min 1.0 [get_ports s_axi_*]
set_output_delay -clock clk -max 3.0 [get_ports s_axi_*]

# AXI-Stream input/output
set_input_delay -clock clk -min 1.0 [get_ports s_axis_t*]
set_input_delay -clock clk -max 3.0 [get_ports s_axis_t*]
set_output_delay -clock clk -min 1.0 [get_ports m_axis_t*]
set_output_delay -clock clk -max 3.0 [get_ports m_axis_t*]

# False paths for async reset
set_false_path -from [get_ports rst_n] -to [all_registers]

# Group PE array for floorplanning (optional)
# create_pblock pblock_pe_array
# add_cells_to_pblock [get_pblocks pblock_pe_array] [get_cells u_pe_array/*]
# resize_pblock [get_pblocks pblock_pe_array] -add {SLICE_X0Y0:SLICE_X20Y50}

# Weight ROM - use block RAM
set_property RAM_STYLE BLOCK [get_cells -hierarchical *weight_rom*]

# Input buffer - use block RAM
set_property RAM_STYLE BLOCK [get_cells -hierarchical *input_buffer*]

# Disable retiming on control FSM
set_property DONT_TOUCH TRUE [get_cells u_ctrl_fsm]

# Max fanout on control signals
set_max_fanout 16 [get_nets -hierarchical *enable*]
set_max_fanout 16 [get_nets -hierarchical *pe_lane_en*]

# Report CDC
report_cdc

# Power optimization
set_property POWER.OPTIMIZATION ON [current_design]