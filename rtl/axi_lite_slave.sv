`ifndef AXI_LITE_SLAVE_SV
`define AXI_LITE_SLAVE_SV

module axi_lite_slave #(
  parameter int ADDR_WIDTH = 8,
  parameter int DATA_WIDTH = 32,
  parameter int MAX_LAYERS = 4
)(
  input  logic                    clk,
  input  logic                    rst_n,
  // AXI-Lite write address channel
  input  logic [ADDR_WIDTH-1:0]   s_axi_awaddr,
  input  logic                    s_axi_awvalid,
  output logic                    s_axi_awready,
  // AXI-Lite write data channel
  input  logic [DATA_WIDTH-1:0]   s_axi_wdata,
  input  logic [DATA_WIDTH/8-1:0] s_axi_wstrb,
  input  logic                    s_axi_wvalid,
  output logic                    s_axi_wready,
  // AXI-Lite write response channel
  output logic [1:0]              s_axi_bresp,
  output logic                    s_axi_bvalid,
  input  logic                    s_axi_bready,
  // AXI-Lite read address channel
  input  logic [ADDR_WIDTH-1:0]   s_axi_araddr,
  input  logic                    s_axi_arvalid,
  output logic                    s_axi_arready,
  // AXI-Lite read data channel
  output logic [DATA_WIDTH-1:0]   s_axi_rdata,
  output logic [1:0]              s_axi_rresp,
  output logic                    s_axi_rvalid,
  input  logic                    s_axi_rready,
  // Register interface
  output logic                    ctrl_start,
  output logic                    ctrl_reset,
  output logic [7:0]              cfg_num_layers,
  output logic [7:0]              cfg_pe_lanes,
  output bnn_pkg::layer_cfg_t     cfg_layer [MAX_LAYERS-1:0],
  input  logic                    status_done,
  input  logic                    status_busy,
  input  logic                    status_error,
  input  logic [31:0]             version_reg
);

  import bnn_pkg::*;

  typedef enum logic [1:0] {
    WR_IDLE,
    WR_ADDR,
    WR_DATA,
    WR_RESP
  } wr_state_e;

  typedef enum logic [1:0] {
    RD_IDLE,
    RD_ADDR,
    RD_DATA
  } rd_state_e;

  wr_state_e wr_state, wr_next;
  rd_state_e rd_state, rd_next;

  logic [ADDR_WIDTH-1:0] awaddr_reg, araddr_reg;
  logic [DATA_WIDTH-1:0] wdata_reg;
  logic [DATA_WIDTH/8-1:0] wstrb_reg;

  // Write FSM
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) wr_state <= WR_IDLE;
    else wr_state <= wr_next;
  end

  always_comb begin
    wr_next = wr_state;
    case (wr_state)
      WR_IDLE:  if (s_axi_awvalid && s_axi_wvalid) wr_next = WR_DATA;
      WR_DATA:  wr_next = WR_RESP;
      WR_RESP:  if (s_axi_bready) wr_next = WR_IDLE;
      default:  wr_next = WR_IDLE;
    endcase
  end

  assign s_axi_awready = (wr_state == WR_IDLE);
  assign s_axi_wready  = (wr_state == WR_DATA);
  assign s_axi_bvalid  = (wr_state == WR_RESP);
  assign s_axi_bresp   = 2'b00;

  // Register write
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ctrl_start     <= 1'b0;
      ctrl_reset     <= 1'b0;
      cfg_num_layers <= '0;
      cfg_pe_lanes   <= '0;
      for (int i = 0; i < MAX_LAYERS; i++) cfg_layer[i] <= '0;
    end else if (wr_state == WR_DATA) begin
      case (awaddr_reg)
        REG_CTRL: begin
          ctrl_start <= s_axi_wdata[CTRL_START_BIT];
          ctrl_reset <= s_axi_wdata[CTRL_RESET_BIT];
        end
        REG_NUM_LAYERS: cfg_num_layers <= s_axi_wdata[7:0];
        REG_PE_LANES:   cfg_pe_lanes   <= s_axi_wdata[7:0];
        default: begin
          // Layer configs: REG_LAYER_BASE + layer_idx * 16
          if (awaddr_reg >= REG_LAYER_BASE && awaddr_reg < REG_LAYER_BASE + MAX_LAYERS*16) begin
            int layer_idx = (awaddr_reg - REG_LAYER_BASE) >> 4;
            int reg_offset = (awaddr_reg - REG_LAYER_BASE) & 4'hF;
            case (reg_offset)
              4'h0: cfg_layer[layer_idx].num_pe_lanes    <= s_axi_wdata[7:0];
              4'h4: cfg_layer[layer_idx].input_features  <= s_axi_wdata[11:0];
              4'h8: cfg_layer[layer_idx].output_features <= s_axi_wdata[11:0];
              4'hC: begin
                cfg_layer[layer_idx].weight_offset <= s_axi_wdata[15:0];
                cfg_layer[layer_idx].threshold     <= s_axi_wdata[31:16];
              end
            endcase
          end
        end
      endcase
    end
  end

  // Capture awaddr/wdata on valid
  always_ff @(posedge clk) begin
    if (wr_state == WR_IDLE && s_axi_awvalid && s_axi_wvalid) begin
      awaddr_reg <= s_axi_awaddr;
      wdata_reg  <= s_axi_wdata;
      wstrb_reg  <= s_axi_wstrb;
    end
  end

  // Read FSM
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) rd_state <= RD_IDLE;
    else rd_state <= rd_next;
  end

  always_comb begin
    rd_next = rd_state;
    case (rd_state)
      RD_IDLE:  if (s_axi_arvalid) rd_next = RD_ADDR;
      RD_ADDR:  rd_next = RD_DATA;
      RD_DATA:  if (s_axi_rready) rd_next = RD_IDLE;
      default:  rd_next = RD_IDLE;
    endcase
  end

  assign s_axi_arready = (rd_state == RD_IDLE);
  assign s_axi_rvalid  = (rd_state == RD_DATA);
  assign s_axi_rresp   = 2'b00;

  // Read data mux
  always_comb begin
    s_axi_rdata = '0;
    case (araddr_reg)
      REG_CTRL:       s_axi_rdata = {30'd0, ctrl_start, ctrl_reset};
      REG_STATUS:     s_axi_rdata = {29'd0, status_error, status_busy, status_done};
      REG_NUM_LAYERS: s_axi_rdata = {24'd0, cfg_num_layers};
      REG_PE_LANES:   s_axi_rdata = {24'd0, cfg_pe_lanes};
      REG_VERSION:    s_axi_rdata = version_reg;
      default: begin
        if (araddr_reg >= REG_LAYER_BASE && araddr_reg < REG_LAYER_BASE + MAX_LAYERS*16) begin
          int layer_idx = (araddr_reg - REG_LAYER_BASE) >> 4;
          int reg_offset = (araddr_reg - REG_LAYER_BASE) & 4'hF;
          case (reg_offset)
            4'h0: s_axi_rdata = {24'd0, cfg_layer[layer_idx].num_pe_lanes};
            4'h4: s_axi_rdata = {20'd0, cfg_layer[layer_idx].input_features};
            4'h8: s_axi_rdata = {20'd0, cfg_layer[layer_idx].output_features};
            4'hC: s_axi_rdata = {cfg_layer[layer_idx].threshold, cfg_layer[layer_idx].weight_offset};
          endcase
        end
      end
    endcase
  end

  always_ff @(posedge clk) begin
    if (rd_state == RD_IDLE && s_axi_arvalid) begin
      araddr_reg <= s_axi_araddr;
    end
  end

endmodule

`endif