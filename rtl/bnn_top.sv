`ifndef BNN_TOP_SV
`define BNN_TOP_SV

module bnn_top #(
  parameter int MAX_PE_LANES     = 8,
  parameter int PE_DATA_WIDTH    = 64,
  parameter int ACC_WIDTH        = 32,
  parameter int WEIGHT_MEM_DEPTH = 1024,
  parameter int MAX_LAYERS       = 4,
  parameter int CFG_ADDR_WIDTH   = 8,
  parameter int CFG_DATA_WIDTH   = 32,
  parameter string WEIGHT_INIT_FILE = ""
)(
  input  logic                     clk,
  input  logic                     rst_n,
  // AXI-Lite configuration interface
  input  logic [CFG_ADDR_WIDTH-1:0] s_axi_awaddr,
  input  logic                     s_axi_awvalid,
  output logic                     s_axi_awready,
  input  logic [CFG_DATA_WIDTH-1:0] s_axi_wdata,
  input  logic [CFG_DATA_WIDTH/8-1:0] s_axi_wstrb,
  input  logic                     s_axi_wvalid,
  output logic                     s_axi_wready,
  output logic [1:0]               s_axi_bresp,
  output logic                     s_axi_bvalid,
  input  logic                     s_axi_bready,
  input  logic [CFG_ADDR_WIDTH-1:0] s_axi_araddr,
  input  logic                     s_axi_arvalid,
  output logic                     s_axi_arready,
  output logic [CFG_DATA_WIDTH-1:0] s_axi_rdata,
  output logic [1:0]               s_axi_rresp,
  output logic                     s_axi_rvalid,
  input  logic                     s_axi_rready,
  // AXI-Stream input
  input  logic [PE_DATA_WIDTH-1:0] s_axis_tdata,
  input  logic                     s_axis_tvalid,
  input  logic                     s_axis_tlast,
  output logic                     s_axis_tready,
  // AXI-Stream output
  output logic [MAX_PE_LANES-1:0]  m_axis_tdata,
  output logic                     m_axis_tvalid,
  output logic                     m_axis_tlast,
  input  logic                     m_axis_tready,
  // Status/IRQ
  output logic                     irq_done,
  output logic                     irq_error
);

  import bnn_pkg::*;

  // ------------------------------------------------------------
  // Configuration registers from AXI-Lite
  // ------------------------------------------------------------
  logic                    ctrl_start;
  logic                    ctrl_reset;
  logic [7:0]              cfg_num_layers;
  logic [7:0]              cfg_pe_lanes;
  layer_cfg_t              cfg_layer [MAX_LAYERS-1:0];
  logic                    status_done, status_busy, status_error;
  logic [31:0]             version_reg;

  assign version_reg = 32'h0100_0001; // v1.0.1

  // ------------------------------------------------------------
  // AXI-Lite slave
  // ------------------------------------------------------------
  axi_lite_slave #(
    .ADDR_WIDTH (CFG_ADDR_WIDTH),
    .DATA_WIDTH (CFG_DATA_WIDTH),
    .MAX_LAYERS (MAX_LAYERS)
  ) u_axi_lite (
    .clk            (clk),
    .rst_n          (rst_n),
    .s_axi_awaddr   (s_axi_awaddr),
    .s_axi_awvalid  (s_axi_awvalid),
    .s_axi_awready  (s_axi_awready),
    .s_axi_wdata    (s_axi_wdata),
    .s_axi_wstrb    (s_axi_wstrb),
    .s_axi_wvalid   (s_axi_wvalid),
    .s_axi_wready   (s_axi_wready),
    .s_axi_bresp    (s_axi_bresp),
    .s_axi_bvalid   (s_axi_bvalid),
    .s_axi_bready   (s_axi_bready),
    .s_axi_araddr   (s_axi_araddr),
    .s_axi_arvalid  (s_axi_arvalid),
    .s_axi_arready  (s_axi_arready),
    .s_axi_rdata    (s_axi_rdata),
    .s_axi_rresp    (s_axi_rresp),
    .s_axi_rvalid   (s_axi_rvalid),
    .s_axi_rready   (s_axi_rready),
    .ctrl_start     (ctrl_start),
    .ctrl_reset     (ctrl_reset),
    .cfg_num_layers (cfg_num_layers),
    .cfg_pe_lanes   (cfg_pe_lanes),
    .cfg_layer      (cfg_layer),
    .status_done    (status_done),
    .status_busy    (status_busy),
    .status_error   (status_error),
    .version_reg    (version_reg)
  );

  // ------------------------------------------------------------
  // Input buffer
  // ------------------------------------------------------------
  logic [PE_DATA_WIDTH-1:0] buf_data_out;
  logic                     buf_data_valid;
  logic                     buf_data_last;
  logic                     buf_data_ready;
  logic                     buf_full, buf_empty;

  input_buffer #(
    .BUFFER_DEPTH (256),
    .DATA_WIDTH   (PE_DATA_WIDTH)
  ) u_input_buf (
    .clk            (clk),
    .rst_n          (rst_n),
    .s_axis_tdata   (s_axis_tdata),
    .s_axis_tvalid  (s_axis_tvalid),
    .s_axis_tlast   (s_axis_tlast),
    .s_axis_tready  (s_axis_tready),
    .data_out       (buf_data_out),
    .data_valid     (buf_data_valid),
    .data_last      (buf_data_last),
    .data_ready     (buf_data_ready),
    .buffer_full    (buf_full),
    .buffer_empty   (buf_empty)
  );

  // ------------------------------------------------------------
  // Weight ROM (one per PE lane)
  // ------------------------------------------------------------
  logic [$clog2(WEIGHT_MEM_DEPTH)-1:0] weight_addr;
  logic                                weight_rom_en;
  logic [PE_DATA_WIDTH-1:0]            weight_rom_data [MAX_PE_LANES-1:0];

  genvar w;
  generate
    for (w = 0; w < MAX_PE_LANES; w++) begin : g_weight_rom
      weight_rom #(
        .DEPTH        (WEIGHT_MEM_DEPTH),
        .DATA_WIDTH   (PE_DATA_WIDTH),
        .INIT_FILE    (WEIGHT_INIT_FILE)
      ) u_weight_rom (
        .clk       (clk),
        .addr      (weight_addr),
        .en        (weight_rom_en),
        .data_out  (weight_rom_data[w])
      );
    end
  endgenerate

  // ------------------------------------------------------------
  // PE lane enable mask
  // ------------------------------------------------------------
  logic [MAX_PE_LANES-1:0] pe_lane_en;
  always_comb begin
    pe_lane_en = '0;
    for (int i = 0; i < MAX_PE_LANES; i++) begin
      if (i < cfg_pe_lanes) pe_lane_en[i] = 1'b1;
    end
  end

  // ------------------------------------------------------------
  // Control FSM
  // ------------------------------------------------------------
  logic                    fsm_layer_done;
  logic                    fsm_all_done;
  logic                    fsm_error;
  logic [3:0]              fsm_current_layer;
  logic [11:0]             fsm_current_neuron;
  logic [2:0]              fsm_state;

  ctrl_fsm #(
    .MAX_LAYERS   (MAX_LAYERS),
    .MAX_PE_LANES (MAX_PE_LANES)
  ) u_ctrl_fsm (
    .clk                   (clk),
    .rst_n                 (rst_n),
    .start                 (ctrl_start),
    .num_layers            (cfg_num_layers),
    .num_pe_lanes          (cfg_pe_lanes),
    .layer_cfg             (cfg_layer),
    .input_buffer_empty    (buf_empty),
    .input_buffer_valid    (buf_data_valid),
    .pe_valid_out          (pe_valid_out),
    .weight_addr           (weight_addr),
    .weight_rom_en         (weight_rom_en),
    .weight_rom_data       (weight_rom_data),
    .layer_done            (fsm_layer_done),
    .all_done              (fsm_all_done),
    .error                 (fsm_error),
    .current_layer         (fsm_current_layer),
    .current_neuron        (fsm_current_neuron),
    .fsm_state             (fsm_state)
  );

  // ------------------------------------------------------------
  // PE Array
  // ------------------------------------------------------------
  logic [MAX_PE_LANES-1:0] pe_valid_out;
  logic [MAX_PE_LANES-1:0] pe_binary_out;
  logic signed [ACC_WIDTH-1:0] pe_acc_out [MAX_PE_LANES-1:0];

  // Current layer threshold (Q4.12)
  logic signed [15:0] current_threshold;
  always_comb begin
    current_threshold = cfg_layer[fsm_current_layer].threshold;
  end

  pe_array #(
    .MAX_PE_LANES (MAX_PE_LANES),
    .DATA_WIDTH   (PE_DATA_WIDTH),
    .ACC_WIDTH    (ACC_WIDTH)
  ) u_pe_array (
    .clk            (clk),
    .rst_n          (rst_n),
    .enable         (buf_data_valid & ~buf_empty),
    .pe_lane_en     (pe_lane_en),
    .data_in        (buf_data_out),
    .weight_in      (weight_rom_data),
    .last_in_batch  (buf_data_last),
    .threshold      (current_threshold),
    .valid_out      (pe_valid_out),
    .binary_out     (pe_binary_out),
    .acc_out        (pe_acc_out)
  );

  // ------------------------------------------------------------
  // Output handling
  // ------------------------------------------------------------
  assign buf_data_ready = m_axis_tready;

  // Pack PE outputs into AXI-Stream
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      m_axis_tdata  <= '0;
      m_axis_tvalid <= 1'b0;
      m_axis_tlast  <= 1'b0;
    end else begin
      m_axis_tdata  <= pe_binary_out;
      m_axis_tvalid <= |pe_valid_out;
      m_axis_tlast  <= fsm_layer_done;
    end
  end

  // Status registers
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      status_done  <= 1'b0;
      status_busy  <= 1'b0;
      status_error <= 1'b0;
    end else begin
      status_done  <= fsm_all_done;
      status_busy  <= (fsm_state != 3'd0) && (fsm_state != 3'd5) && (fsm_state != 3'd7);
      status_error <= fsm_error;
    end
  end

  // IRQs
  assign irq_done  = fsm_all_done;
  assign irq_error = fsm_error;

endmodule

`endif