`ifndef BNN_PKG_SV
`define BNN_PKG_SV

package bnn_pkg;

  // ------------------------------------------------------------
  // Configuration parameters
  // ------------------------------------------------------------
  parameter int MAX_PE_LANES     = 8;
  parameter int PE_DATA_WIDTH    = 64;     // bits processed per PE per cycle (XNOR width)
  parameter int ACC_WIDTH        = 32;     // accumulator width (prevents overflow)
  parameter int WEIGHT_MEM_DEPTH = 1024;   // number of 64-bit words in weight ROM
  parameter int MAX_LAYERS       = 4;
  parameter int CFG_ADDR_WIDTH   = 8;
  parameter int CFG_DATA_WIDTH   = 32;

  // ------------------------------------------------------------
  // Layer configuration structure
  // ------------------------------------------------------------
  typedef struct packed {
    logic [7:0]  num_pe_lanes;     // active PE lanes (1-8)
    logic [11:0] input_features;   // input dimension (must be multiple of 64)
    logic [11:0] output_features;  // output dimension
    logic [15:0] weight_offset;    // offset in weight ROM (in 64-bit words)
    logic [15:0] threshold;        // batch-norm threshold (Q4.12 format)
    logic        is_last_layer;    // skip ReLU on last layer
  } layer_cfg_t;

  // ------------------------------------------------------------
  // AXI-Lite register map
  // ------------------------------------------------------------
  typedef enum logic [7:0] {
    REG_CTRL           = 8'h00,  // [0] start, [1] reset
    REG_STATUS         = 8'h04,  // [0] done, [1] busy, [2] error
    REG_NUM_LAYERS     = 8'h08,
    REG_LAYER_BASE     = 8'h10,  // layer configs start here (16 bytes each)
    REG_PE_LANES       = 8'h0C,  // global PE lanes override
    REG_INPUT_ADDR     = 8'h20,  // input buffer base address (AXI-S)
    REG_OUTPUT_ADDR    = 8'h24,  // output buffer base address (AXI-S)
    REG_VERSION        = 8'hFC   // IP version
  } reg_addr_e;

  // Control register bits
  parameter logic CTRL_START_BIT  = 0;
  parameter logic CTRL_RESET_BIT  = 1;

  // Status register bits
  parameter logic STATUS_DONE_BIT  = 0;
  parameter logic STATUS_BUSY_BIT  = 1;
  parameter logic STATUS_ERROR_BIT = 2;

  // ------------------------------------------------------------
  // Popcount lookup table for 8-bit (synthesizable)
  // ------------------------------------------------------------
  function automatic logic [3:0] popcount8(input logic [7:0] x);
    logic [3:0] cnt;
    begin
      cnt = 0;
      for (int i = 0; i < 8; i++) begin
        cnt += x[i];
      end
      return cnt;
    end
  endfunction

  // ------------------------------------------------------------
  // Popcount for 64-bit using 8-bit LUT
  // ------------------------------------------------------------
  function automatic logic [6:0] popcount64(input logic [63:0] x);
    logic [6:0] cnt;
    begin
      cnt = 0;
      for (int i = 0; i < 8; i++) begin
        cnt += popcount8(x[i*8 +: 8]);
      end
      return cnt;
    end
  endfunction

  // ------------------------------------------------------------
  // Sign function for binary activation
  // ------------------------------------------------------------
  function automatic logic sign_q4_12(input logic signed [ACC_WIDTH-1:0] x, input logic [15:0] threshold);
    // x is signed accumulator, threshold is Q4.12
    // return 1 if x > threshold, else 0 (binary {+1, -1} encoded as {1, 0})
    logic signed [ACC_WIDTH-1:0] thresh_ext;
    begin
      thresh_ext = {{(ACC_WIDTH-16){threshold[15]}}, threshold}; // sign-extend threshold
      return (x > thresh_ext) ? 1'b1 : 1'b0;
    end
  endfunction

endpackage

`endif