`ifndef PE_UNIT_STANDALONE_SV
`define PE_UNIT_STANDALONE_SV

module pe_unit #(
  parameter int PE_ID          = 0,
  parameter int DATA_WIDTH     = 64,
  parameter int ACC_WIDTH      = 32
)(
  input  logic                   clk,
  input  logic                   rst_n,
  input  logic                   enable,
  input  logic [DATA_WIDTH-1:0]  data_in,
  input  logic [DATA_WIDTH-1:0]  weight_in,
  input  logic                   last_in_batch,
  input  logic signed [15:0]     thresh_in,
  output logic                   valid_out,
  output logic                   binary_out,
  output logic signed [ACC_WIDTH-1:0] acc_out
);

  // ------------------------------------------------------------
  // Popcount functions
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
  // XNOR + Popcount
  // ------------------------------------------------------------
  logic [DATA_WIDTH-1:0] xnor_result;
  logic [6:0]            popcnt_result;
  logic signed [ACC_WIDTH-1:0] acc_reg;
  logic signed [ACC_WIDTH-1:0] acc_next;

  assign xnor_result    = ~(data_in ^ weight_in);
  assign popcnt_result  = popcount64(xnor_result);
  assign acc_next       = acc_reg + $signed({{(ACC_WIDTH-7){1'b0}}, popcnt_result});

  // ------------------------------------------------------------
  // Accumulator: accumulates during batch, captures on last_in_batch
  // ------------------------------------------------------------
  logic signed [ACC_WIDTH-1:0] acc_captured;
  logic last_in_batch_d;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      acc_reg         <= '0;
      acc_captured    <= '0;
      last_in_batch_d <= 1'b0;
    end else if (enable) begin
      last_in_batch_d <= last_in_batch;
      if (last_in_batch) begin
        acc_captured <= acc_next;
        acc_reg      <= '0;
      end else begin
        acc_reg      <= acc_next;
      end
    end
  end

  // ------------------------------------------------------------
  // Output valid: one cycle latency after last_in_batch
  // ------------------------------------------------------------
  assign valid_out = last_in_batch_d;

  // ------------------------------------------------------------
  // Binary activation: combinational comparison
  // binary_out is valid when valid_out is high
  // ------------------------------------------------------------
  logic signed [ACC_WIDTH-1:0] thresh_extended;

  assign thresh_extended = $signed({{(ACC_WIDTH-16){thresh_in[15]}}, thresh_in});

  assign binary_out  = (valid_out && ($signed(acc_captured) > thresh_extended)) ? 1'b1 : 1'b0;
  assign acc_out     = acc_captured;

endmodule

`endif
