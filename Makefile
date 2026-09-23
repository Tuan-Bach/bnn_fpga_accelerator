# Makefile for BNN FPGA Accelerator
# Supports: simulation, synthesis, linting

.PHONY: all sim synth-vivado synth-gowin lint clean help

# Default target
all: sim

# Tools
VLOG = vlog
VSIM = vsim
VIVADO = vivado
GOWIN = gw_sh
VERILATOR = verilator

# Directories
RTL_DIR = rtl
TB_DIR = tb
SIM_DIR = sim
SCRIPTS_DIR = scripts
CONSTRAINTS_DIR = constraints
PYTHON_DIR = python
WORK_DIR = work

# Top modules
TOP_TB = tb_bnn_top
TOP_RTL = bnn_top

# Part numbers
ZYNQ_PART = xc7z020clg400-1
GOWIN_DEVICE = GW1NR-LV9QN88PC6/I5

# ------------------------------------------------------------
# Simulation (ModelSim/QuestaSim)
# ------------------------------------------------------------
sim: $(WORK_DIR)/._compiled
	@echo "=== Running Simulation ==="
	cd $(SIM_DIR) && $(VSIM) -c -do "run -all; quit" -voptargs="+acc" $(TOP_TB)

$(WORK_DIR)/._compiled: $(RTL_DIR)/*.sv $(TB_DIR)/*.sv
	@echo "=== Compiling RTL & Testbench ==="
	@mkdir -p $(WORK_DIR)
	vlib $(WORK_DIR)
	vmap work $(WORK_DIR)
	$(VLOG) -work $(WORK_DIR) $(RTL_DIR)/bnn_pkg.sv
	$(VLOG) -work $(WORK_DIR) $(RTL_DIR)/pe_unit.sv
	$(VLOG) -work $(WORK_DIR) $(RTL_DIR)/pe_array.sv
	$(VLOG) -work $(WORK_DIR) $(RTL_DIR)/weight_rom.sv
	$(VLOG) -work $(WORK_DIR) $(RTL_DIR)/input_buffer.sv
	$(VLOG) -work $(WORK_DIR) $(RTL_DIR)/ctrl_fsm.sv
	$(VLOG) -work $(WORK_DIR) $(RTL_DIR)/axi_lite_slave.sv
	$(VLOG) -work $(WORK_DIR) $(RTL_DIR)/bnn_top.sv
	$(VLOG) -work $(WORK_DIR) $(TB_DIR)/tb_bnn_top.sv
	@touch $(WORK_DIR)/._compiled

# Simulation with GUI
sim-gui: $(WORK_DIR)/._compiled
	cd $(SIM_DIR) && $(VSIM) -do "run -all" $(TOP_TB)

# Simulation with coverage
sim-cov: $(WORK_DIR)/._compiled
	cd $(SIM_DIR) && $(VSIM) -coverage -c -do "coverage save -onexit cov.ucdb; run -all; quit" $(TOP_TB)

# ------------------------------------------------------------
# Verilator Linting (fast syntax check)
# ------------------------------------------------------------
lint:
	@echo "=== Running Verilator Lint ==="
	$(VERILATOR) --lint-only -Wall -Wno-UNUSEDSIGNAL -Wno-UNOPTFLAT \
		-Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-IMPLICITSTATIC \
		-Wno-UNUSEDPARAM -Wno-EOFNEWLINE \
		-I$(RTL_DIR) \
		$(RTL_DIR)/bnn_pkg.sv \
		$(RTL_DIR)/pe_unit.sv \
		$(RTL_DIR)/pe_array.sv \
		$(RTL_DIR)/weight_rom.sv \
		$(RTL_DIR)/input_buffer.sv \
		$(RTL_DIR)/ctrl_fsm.sv \
		$(RTL_DIR)/axi_lite_slave.sv \
		$(RTL_DIR)/bnn_top.sv

# ------------------------------------------------------------
# Verilator Simulation (PE unit testbench)
# ------------------------------------------------------------
PE_TB_OBJ = obj_dir/Vpe_unit

$(PE_TB_OBJ): $(RTL_DIR)/pe_unit.sv $(TB_DIR)/pe_unit_tb.cpp
	@echo "=== Building PE Unit Testbench ==="
	$(VERILATOR) --cc --exe --build --timing --trace \
		-I$(RTL_DIR) -Wall -Wno-UNUSEDSIGNAL -Wno-UNOPTFLAT \
		-Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-IMPLICITSTATIC \
		-Wno-UNUSEDPARAM -Wno-EOFNEWLINE \
		$(RTL_DIR)/pe_unit.sv $(TB_DIR)/pe_unit_tb.cpp

sim-pe: $(PE_TB_OBJ)
	@echo "=== Running PE Unit Tests ==="
	./$(PE_TB_OBJ)

sim-pe-clean:
	rm -rf obj_dir

# ------------------------------------------------------------
# Dataset Verification (train + export + simulate)
#   make sim-dataset DATASET=fashion_mnist
#   make sim-dataset DATASET=emnist
#   make sim-dataset DATASET=cifar10
#   make sim-dataset DATASET=pcb   (run python/prepare_pcb.py first)
# ------------------------------------------------------------
DATASET ?= mnist

mnist-train:
	@echo "=== Training BNN (PyTorch, dataset=$(DATASET)) ==="
	cd python && python3 train_bnn_dataset.py --dataset $(DATASET)

mnist-export:
	@echo "=== Exporting Weights & Calibrating Thresholds ==="
	cd python && python3 export_bnn_dataset.py --dataset $(DATASET)

mnist-build:
	@echo "=== Building Verification Testbench (data-driven) ==="
	rm -rf obj_dir_mnist
	$(VERILATOR) --cc --exe --build --timing \
		-I$(RTL_DIR) -Wall -Wno-UNUSEDSIGNAL -Wno-UNOPTFLAT \
		-Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-IMPLICITSTATIC \
		-Wno-UNUSEDPARAM -Wno-EOFNEWLINE \
		--top-module pe_unit \
		$(RTL_DIR)/pe_unit.sv $(TB_DIR)/mnist_verify_tb.cpp \
		-Mdir obj_dir_mnist --exe mnist_verify_tb.cpp

# Run against a dataset directory (default: mnist)
DATA_DIR ?= tb/mnist_data
mnist-sim:
	@echo "=== Running Verification: $(DATA_DIR) ==="
	./obj_dir_mnist/Vpe_unit $(DATA_DIR)

# Train + export + simulate the named dataset
sim-dataset: mnist-train mnist-export mnist-build
	@if [ "$(DATASET)" = "mnist" ]; then \
		./obj_dir_mnist/Vpe_unit tb/mnist_data; \
	else \
		./obj_dir_mnist/Vpe_unit tb/$(DATASET)_data; \
	fi

# Full MNIST flow (default dataset)
sim-mnist: sim-dataset

# ------------------------------------------------------------
# Synthesis - Vivado (Zynq-7000)
# ------------------------------------------------------------
synth-vivado:
	@echo "=== Vivado Synthesis for Zynq-7000 ==="
	cd $(SCRIPTS_DIR) && ./synth_vivado.sh

synth-vivado-gui:
	@echo "=== Opening Vivado GUI ==="
	cd $(SCRIPTS_DIR) && $(VIVADO) -mode gui -source synth.tcl

# ------------------------------------------------------------
# Synthesis - Gowin EDA (Tang Nano 9K)
# ------------------------------------------------------------
synth-gowin:
	@echo "=== Gowin EDA Synthesis for Tang Nano 9K ==="
	cd $(SCRIPTS_DIR) && ./synth_gowin.sh

# ------------------------------------------------------------
# Python: Train & Export
# ------------------------------------------------------------
train:
	@echo "=== Training BNN Model ==="
	cd $(PYTHON_DIR) && python3 train_bnn.py --model mlp --epochs 50 --batch-size 128

export-weights:
	@echo "=== Exporting Weights ==="
	cd $(PYTHON_DIR) && python3 export_weights.py --model bnn_best.h5 --output-dir ../$(RTL_DIR)/weights

# ------------------------------------------------------------
# Flash bitstream to Tang Nano 9K
# ------------------------------------------------------------
flash:
	@echo "=== Flashing Bitstream to Tang Nano 9K ==="
	openFPGALoader -b tangnano9k $(SCRIPTS_DIR)/bnn_accelerator.fs

# ------------------------------------------------------------
# Clean
# ------------------------------------------------------------
clean:
	@echo "=== Cleaning Build Artifacts ==="
	rm -rf $(WORK_DIR)
	rm -rf $(SCRIPTS_DIR)/*.log $(SCRIPTS_DIR)/*.jou $(SCRIPTS_DIR)/*.dcp $(SCRIPTS_DIR)/*.fs
	rm -rf $(SCRIPTS_DIR)/bnn_accelerator.gprj
	rm -rf $(SIM_DIR)/*.vcd $(SIM_DIR)/*.ucdb $(SIM_DIR)/transcript
	rm -rf $(PYTHON_DIR)/__pycache__ $(PYTHON_DIR)/*.h5 $(PYTHON_DIR)/*.csv $(PYTHON_DIR)/logs
	rm -rf $(RTL_DIR)/weights
	rm -rf obj_dir

# ------------------------------------------------------------
# Help
# ------------------------------------------------------------
help:
	@echo "BNN FPGA Accelerator - Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make sim            - Run ModelSim simulation (default)"
	@echo "  make sim-gui        - Run simulation with GUI"
	@echo "  make sim-cov        - Run simulation with coverage"
	@echo "  make lint           - Run Verilator lint check"
	@echo "  make sim-pe         - Run PE unit tests (Verilator)"
	@echo "  make sim-pe-clean   - Clean Verilator build artifacts"
	@echo "  make mnist-build    - Build MNIST verification testbench"
	@echo "  make sim-mnist      - Train + export + MNIST verification"
	@echo "  make synth-vivado   - Synthesize for Zynq-7000 (Vivado)"
	@echo "  make synth-gowin    - Synthesize for Tang Nano 9K (Gowin)"
	@echo "  make train          - Train BNN model (Python)"
	@echo "  make export-weights - Export weights to .mem format"
	@echo "  make flash          - Flash bitstream to Tang Nano 9K"
	@echo "  make clean          - Clean all build artifacts"
	@echo "  make help           - Show this help"