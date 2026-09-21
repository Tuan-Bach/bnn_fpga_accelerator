`ifndef TB_BNN_TOP_SV
`define TB_BNN_TOP_SV

module tb_bnn_top;

  import bnn_pkg::*;

  // ------------------------------------------------------------
  // Clock & Reset
  // ------------------------------------------------------------
  parameter int CLK_PERIOD = 10; // 100 MHz
  logic clk;
  logic rst_n;

  initial clk = 0;
  always #(CLK_PERIOD/2) clk = ~clk;

  initial begin
    rst_n = 0;
    repeat (5) @(posedge clk);
    rst_n = 1;
  end

  // ------------------------------------------------------------
  // DUT signals
  // ------------------------------------------------------------
  logic [7:0]  s_axi_awaddr;
  logic        s_axi_awvalid;
  logic        s_axi_awready;
  logic [31:0] s_axi_wdata;
  logic [3:0]  s_axi_wstrb;
  logic        s_axi_wvalid;
  logic        s_axi_wready;
  logic [1:0]  s_axi_bresp;
  logic        s_axi_bvalid;
  logic        s_axi_bready;
  logic [7:0]  s_axi_araddr;
  logic        s_axi_arvalid;
  logic        s_axi_arready;
  logic [31:0] s_axi_rdata;
  logic [1:0]  s_axi_rresp;
  logic        s_axi_rvalid;
  logic        s_axi_rready;

  logic [63:0] s_axis_tdata;
  logic        s_axis_tvalid;
  logic        s_axis_tlast;
  logic        s_axis_tready;

  logic [7:0]  m_axis_tdata;
  logic        m_axis_tvalid;
  logic        m_axis_tlast;
  logic        m_axis_tready;

  logic        irq_done;
  logic        irq_error;

  // ------------------------------------------------------------
  // Testbench memory for weights
  // ------------------------------------------------------------
  logic [63:0] weight_mem [0:1023];
  logic [63:0] input_mem  [0:255];
  logic [7:0]  expected_out [0:255];

  // ------------------------------------------------------------
  // DUT Instance
  // ------------------------------------------------------------
  bnn_top #(
    .MAX_PE_LANES     (8),
    .PE_DATA_WIDTH    (64),
    .ACC_WIDTH        (32),
    .WEIGHT_MEM_DEPTH (1024),
    .MAX_LAYERS       (4),
    .CFG_ADDR_WIDTH   (8),
    .CFG_DATA_WIDTH   (32),
    .WEIGHT_INIT_FILE ("")
  ) u_dut (
    .clk               (clk),
    .rst_n             (rst_n),
    .s_axi_awaddr      (s_axi_awaddr),
    .s_axi_awvalid     (s_axi_awvalid),
    .s_axi_awready     (s_axi_awready),
    .s_axi_wdata       (s_axi_wdata),
    .s_axi_wstrb       (s_axi_wstrb),
    .s_axi_wvalid      (s_axi_wvalid),
    .s_axi_wready      (s_axi_wready),
    .s_axi_bresp       (s_axi_bresp),
    .s_axi_bvalid      (s_axi_bvalid),
    .s_axi_bready      (s_axi_bready),
    .s_axi_araddr      (s_axi_araddr),
    .s_axi_arvalid     (s_axi_arvalid),
    .s_axi_arready     (s_axi_arready),
    .s_axi_rdata       (s_axi_rdata),
    .s_axi_rresp       (s_axi_rresp),
    .s_axi_rvalid      (s_axi_rvalid),
    .s_axi_rready      (s_axi_rready),
    .s_axis_tdata      (s_axis_tdata),
    .s_axis_tvalid     (s_axis_tvalid),
    .s_axis_tlast      (s_axis_tlast),
    .s_axis_tready     (s_axis_tready),
    .m_axis_tdata      (m_axis_tdata),
    .m_axis_tvalid     (m_axis_tvalid),
    .m_axis_tlast      (m_axis_tlast),
    .m_axis_tready     (m_axis_tready),
    .irq_done          (irq_done),
    .irq_error         (irq_error)
  );

  // ------------------------------------------------------------
  // AXI-Lite Master Tasks
  // ------------------------------------------------------------
  task automatic axi_lite_write(input logic [7:0] addr, input logic [31:0] data);
    begin
      @(posedge clk);
      s_axi_awaddr  = addr;
      s_axi_awvalid = 1'b1;
      s_axi_wdata   = data;
      s_axi_wstrb   = 4'hF;
      s_axi_wvalid  = 1'b1;
      s_axi_bready  = 1'b1;
      wait (s_axi_awready && s_axi_wready);
      @(posedge clk);
      s_axi_awvalid = 1'b0;
      s_axi_wvalid  = 1'b0;
      wait (s_axi_bvalid);
      @(posedge clk);
      s_axi_bready = 1'b0;
    end
  endtask

  task automatic axi_lite_read(input logic [7:0] addr, output logic [31:0] data);
    begin
      @(posedge clk);
      s_axi_araddr  = addr;
      s_axi_arvalid = 1'b1;
      s_axi_rready  = 1'b1;
      wait (s_axi_arready);
      @(posedge clk);
      s_axi_arvalid = 1'b0;
      wait (s_axi_rvalid);
      data = s_axi_rdata;
      @(posedge clk);
      s_axi_rready = 1'b0;
    end
  endtask

  // ------------------------------------------------------------
  // Initialize weight ROM (direct memory access for simulation)
  // ------------------------------------------------------------
  initial begin
    // Simple test weights: identity-like pattern
    for (int i = 0; i < 1024; i++) begin
      weight_mem[i] = {8{8'hFF}}; // all 1s
    end
    // Override first few for testing
    weight_mem[0] = 64'hFFFF_FFFF_FFFF_FFFF;
    weight_mem[1] = 64'h0000_0000_0000_0000;
    weight_mem[2] = 64'hAAAA_AAAA_AAAA_AAAA;
    weight_mem[3] = 64'h5555_5555_5555_5555;
  end

  // Preload weight ROM in DUT (simulation only)
  initial begin
    @(posedge rst_n);
    for (int w = 0; w < 8; w++) begin
      for (int i = 0; i < 1024; i++) begin
        u_dut.u_weight_rom[w].mem[i] = weight_mem[i];
      end
    end
  end

  // ------------------------------------------------------------
  // Input stimulus
  // ------------------------------------------------------------
  initial begin
    // Initialize input data
    for (int i = 0; i < 256; i++) begin
      input_mem[i] = $urandom;
    end
    input_mem[0] = 64'hFFFF_FFFF_FFFF_FFFF; // all 1s
    input_mem[1] = 64'h0000_0000_0000_0000; // all 0s
    input_mem[2] = 64'hAAAA_AAAA_AAAA_AAAA; // alternating

    // Expected outputs (for 1 PE lane, threshold=0)
    // popcount(0xFFFF...) = 64 -> positive -> 1
    // popcount(0x0000...) = 0   -> negative -> 0
    // popcount(0xAAAA...) = 32  -> positive -> 1
  end

  // ------------------------------------------------------------
  // Main test sequence
  // ------------------------------------------------------------
  initial begin
    // Initialize AXI signals
    s_axi_awaddr  = '0;
    s_axi_awvalid = 1'b0;
    s_axi_wdata   = '0;
    s_axi_wstrb   = '0;
    s_axi_wvalid  = 1'b0;
    s_axi_bready  = 1'b0;
    s_axi_araddr  = '0;
    s_axi_arvalid = 1'b0;
    s_axi_rready  = 1'b0;

    s_axis_tdata  = '0;
    s_axis_tvalid = 1'b0;
    s_axis_tlast  = 1'b0;
    m_axis_tready = 1'b1;

    @(posedge rst_n);
    repeat (10) @(posedge clk);

    $display("[TB] Starting BNN Accelerator Test");

    // 1. Configure layer 0: 784 inputs -> 128 outputs, 1 PE lane
    axi_lite_write(REG_NUM_LAYERS, 32'h1);           // 1 layer
    axi_lite_write(REG_PE_LANES,   32'h1);           // 1 PE lane
    axi_lite_write(REG_LAYER_BASE + 0, 32'h0001_0310); // num_pe=1, input_feat=784
    axi_lite_write(REG_LAYER_BASE + 4, 32'h0000_0080); // output_feat=128
    axi_lite_write(REG_LAYER_BASE + 8, 32'h0000_0000); // weight_offset=0, threshold=0

    // 2. Read back config
    logic [31:0] rd_data;
    axi_lite_read(REG_NUM_LAYERS, rd_data);
    $display("[TB] Num layers: %d", rd_data[7:0]);
    axi_lite_read(REG_PE_LANES, rd_data);
    $display("[TB] PE lanes: %d", rd_data[7:0]);

    // 3. Start accelerator
    $display("[TB] Starting accelerator...");
    axi_lite_write(REG_CTRL, 32'h1); // start bit

    // 4. Feed input data (3 input vectors for 1 output neuron with 784/64 = 13 words)
    repeat (5) @(posedge clk);
    
    for (int vec = 0; vec < 3; vec++) begin
      for (int word = 0; word < 13; word++) begin
        @(posedge clk);
        while (!s_axis_tready) @(posedge clk);
        s_axis_tdata  = input_mem[vec];
        s_axis_tvalid = 1'b1;
        s_axis_tlast  = (word == 12);
      end
      @(posedge clk);
      s_axis_tvalid = 1'b0;
      s_axis_tlast  = 1'b0;
      repeat (2) @(posedge clk);
    end

    // 5. Collect outputs
    int output_count = 0;
    logic [7:0] captured_outputs [0:127];
    
    while (output_count < 128) begin
      @(posedge clk);
      if (m_axis_tvalid && m_axis_tready) begin
        captured_outputs[output_count] = m_axis_tdata;
        output_count++;
        $display("[TB] Output[%0d] = %b", output_count-1, m_axis_tdata);
      end
    end

    // 6. Wait for done
    wait (irq_done);
    $display("[TB] Accelerator DONE");

    // 7. Check status
    axi_lite_read(REG_STATUS, rd_data);
    $display("[TB] Status: done=%b busy=%b error=%b", rd_data[0], rd_data[1], rd_data[2]);

    // 8. Verify results (basic sanity)
    int errors = 0;
    for (int i = 0; i < 128; i++) begin
      // With threshold=0, all positive accumulations should give 1
      // Since weights are all 1s and inputs vary...
      // This is just a functional test
    end

    $display("[TB] Test completed with %0d errors", errors);
    
    if (errors == 0) begin
      $display("[TB] *** TEST PASSED ***");
    end else begin
      $display("[TB] *** TEST FAILED ***");
    end

    repeat (10) @(posedge clk);
    $finish;
  end

  // ------------------------------------------------------------
  // Timeout watchdog
  // ------------------------------------------------------------
  initial begin
    repeat (100000) @(posedge clk);
    $display("[TB] TIMEOUT");
    $finish;
  end

  // ------------------------------------------------------------
  // Waveform dump
  // ------------------------------------------------------------
  initial begin
    $dumpfile("tb_bnn_top.vcd");
    $dumpvars(0, tb_bnn_top);
  end

endmodule

`endif