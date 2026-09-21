`ifndef INPUT_BUFFER_SV
`define INPUT_BUFFER_SV

module input_buffer #(
  parameter int BUFFER_DEPTH = 256,
  parameter int DATA_WIDTH   = 64
)(
  input  logic                    clk,
  input  logic                    rst_n,
  // AXI-Stream input
  input  logic [DATA_WIDTH-1:0]   s_axis_tdata,
  input  logic                    s_axis_tvalid,
  input  logic                    s_axis_tlast,
  output logic                    s_axis_tready,
  // Output to PE array
  output logic [DATA_WIDTH-1:0]   data_out,
  output logic                    data_valid,
  output logic                    data_last,
  input  logic                    data_ready,
  // Status
  output logic                    buffer_full,
  output logic                    buffer_empty
);

  typedef enum logic [1:0] {
    IDLE,
    FILL,
    STREAM
  } state_t;

  state_t current_state, next_state;

  logic [DATA_WIDTH-1:0] buffer_mem [0:BUFFER_DEPTH-1];
  logic [$clog2(BUFFER_DEPTH):0] write_ptr, read_ptr;
  logic [$clog2(BUFFER_DEPTH):0] count;

  // Write side (AXI-Stream)
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      write_ptr <= '0;
      count     <= '0;
    end else if (s_axis_tvalid && s_axis_tready) begin
      buffer_mem[write_ptr[$clog2(BUFFER_DEPTH)-1:0]] <= s_axis_tdata;
      write_ptr <= write_ptr + 1'b1;
      count     <= count + 1'b1;
    end else if (data_valid && data_ready && !(s_axis_tvalid && s_axis_tready)) begin
      count <= count - 1'b1;
    end
  end

  // Read side
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      read_ptr   <= '0;
      data_out   <= '0;
      data_valid <= 1'b0;
      data_last  <= 1'b0;
    end else if (data_ready && data_valid) begin
      read_ptr   <= read_ptr + 1'b1;
      data_out   <= buffer_mem[read_ptr[$clog2(BUFFER_DEPTH)-1:0]];
      data_valid <= (count > 1);
      data_last  <= (count == 1);
    end else if (!(data_ready && data_valid)) begin
      data_valid <= (count > 0);
      data_last  <= (count == 1);
    end
  end

  // Ready/Full/Empty
  assign s_axis_tready = (count < BUFFER_DEPTH);
  assign buffer_full   = (count == BUFFER_DEPTH);
  assign buffer_empty  = (count == 0);

endmodule

`endif