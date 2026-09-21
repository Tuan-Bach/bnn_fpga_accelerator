`ifndef PE_ARRAY_SV
`define PE_ARRAY_SV

module pe_array #(
  parameter int MAX_PE_LANES = 8,
  parameter int DATA_WIDTH   = 64,
  parameter int ACC_WIDTH    = 32
)(
  input  logic                         clk,
  input  logic                         rst_n,
  input  logic                         enable,
  input  logic [MAX_PE_LANES-1:0]      pe_lane_en,      // one-hot or binary enable per lane
  input  logic [DATA_WIDTH-1:0]        data_in,         // broadcast to all PEs
  input  logic [DATA_WIDTH-1:0]        weight_in [MAX_PE_LANES-1:0], // per-PE weights
  input  logic                         last_in_batch,
  input  logic signed [15:0]           threshold,
  output logic [MAX_PE_LANES-1:0]      valid_out,
  output logic [MAX_PE_LANES-1:0]      binary_out,
  output logic signed [ACC_WIDTH-1:0]  acc_out [MAX_PE_LANES-1:0]
);

  import bnn_pkg::*;

  genvar i;
  generate
    for (i = 0; i < MAX_PE_LANES; i++) begin : g_pe
      pe_unit #(
        .PE_ID      (i),
        .DATA_WIDTH (DATA_WIDTH),
        .ACC_WIDTH  (ACC_WIDTH)
      ) u_pe (
        .clk            (clk),
        .rst_n          (rst_n),
        .enable         (enable & pe_lane_en[i]),
        .data_in        (data_in),
        .weight_in      (weight_in[i]),
        .last_in_batch  (last_in_batch),
        .thresh_in      (threshold),
        .valid_out      (valid_out[i]),
        .binary_out     (binary_out[i]),
        .acc_out        (acc_out[i])
      );
    end
  endgenerate

endmodule

`endif