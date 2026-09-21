## SDC Constraints for Tang Nano 9K (Gowin GW1NR-9)
## Device: GW1NR-LV9QN88PC6/I5

# Clock constraint - 27 MHz (onboard oscillator) or 100 MHz (PLL)
create_clock -period 37.037 -name clk_27m [get_ports clk]
# create_clock -period 10.000 -name clk_100m [get_ports clk]  # if using PLL

# Reset
set_false_path -from [get_ports rst_n]

# Input/Output delays
set_input_delay -clock clk_27m -min 1.0 [get_ports s_axi_*]
set_input_delay -clock clk_27m -max 5.0 [get_ports s_axi_*]
set_output_delay -clock clk_27m -min 1.0 [get_ports s_axi_*]
set_output_delay -clock clk_27m -max 5.0 [get_ports s_axi_*]

# AXI-Stream (if used)
set_input_delay -clock clk_27m -min 1.0 [get_ports s_axis_t*]
set_input_delay -clock clk_27m -max 5.0 [get_ports s_axis_t*]
set_output_delay -clock clk_27m -min 1.0 [get_ports m_axis_t*]
set_output_delay -clock clk_27m -max 5.0 [get_ports m_axis_t*]

# Block RAM for weight ROM and input buffer
set_property ramstyle block_ram [get_cells -hierarchical *weight_rom*]
set_property ramstyle block_ram [get_cells -hierarchical *input_buffer*]

# Area optimization for small FPGA
set_option -max_fanout 32
set_option -effort_level high

# Timing analysis
report_timing -max_paths 10