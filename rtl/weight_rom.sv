`ifndef WEIGHT_ROM_SV
`define WEIGHT_ROM_SV

module weight_rom #(
  parameter int DEPTH        = 1024,
  parameter int DATA_WIDTH   = 64,
  parameter string INIT_FILE = ""
)(
  input  logic                    clk,
  input  logic [$clog2(DEPTH)-1:0] addr,
  input  logic                    en,
  output logic [DATA_WIDTH-1:0]   data_out
);

  logic [DATA_WIDTH-1:0] mem [0:DEPTH-1];

  // Initialize from file if provided
  initial begin
    if (INIT_FILE != "") begin
      $readmemh(INIT_FILE, mem);
    end else begin
      // Default: all zeros
      for (int i = 0; i < DEPTH; i++) begin
        mem[i] = '0;
      end
    end
  end

  // Single-cycle read
  always_ff @(posedge clk) begin
    if (en) begin
      data_out <= mem[addr];
    end
  end

endmodule

`endif