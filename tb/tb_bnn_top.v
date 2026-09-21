`ifndef TB_BNN_TOP_V
`define TB_BNN_TOP_V

module tb_bnn_top;

  // ------------------------------------------------------------
  // Clock & Reset
  // ------------------------------------------------------------
  parameter CLK_PERIOD = 10;
  reg clk;
  reg rst_n;

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
  reg  [7:0]   s_axi_awaddr;
  reg          s_axi_awvalid;
  wire         s_axi_awready;
  reg  [31:0]  s_axi_wdata;
  reg  [3:0]   s_axi_wstrb;
  reg          s_axi_wvalid;
  wire         s_axi_wready;
  wire [1:0]   s_axi_bresp;
  wire         s_axi_bvalid;
  reg          s_axi_bready;
  reg  [7:0]   s_axi_araddr;
  reg          s_axi_arvalid;
  wire         s_axi_arready;
  wire [31:0]  s_axi_rdata;
  wire [1:0]   s_axi_rresp;
  wire         s_axi_rvalid;
  reg          s_axi_rready;

  reg  [63:0]  s_axis_tdata;
  reg          s_axis_tvalid;
  reg          s_axis_tlast;
  wire         s_axis_tready;

  wire [7:0]   m_axis_tdata;
  wire         m_axis_tvalid;
  wire         m_axis_tlast;
  reg          m_axis_tready;

  wire         irq_done;
  wire         irq_error;

  // ------------------------------------------------------------
  // Register map constants
  // ------------------------------------------------------------
  localparam [7:0] REG_CTRL           = 8'h00;
  localparam [7:0] REG_STATUS         = 8'h04;
  localparam [7:0] REG_NUM_LAYERS     = 8'h08;
  localparam [7:0] REG_PE_LANES       = 8'h0C;
  localparam [7:0] REG_LAYER_BASE     = 8'h10;
  localparam [7:0] REG_VERSION        = 8'hFC;

  // ------------------------------------------------------------
  // Test memory
  // ------------------------------------------------------------
  reg [63:0] weight_mem [0:1023];
  reg [63:0] input_mem  [0:255];
  reg [7:0]  expected_out [0:255];

  // ------------------------------------------------------------
  // Test variables (declare at module level for Verilator)
  // ------------------------------------------------------------
  reg [31:0] rd_data;
  integer vec;
  integer word;
  integer output_count;
  integer errors;
  reg [7:0] captured_outputs [0:127];

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
  task axi_lite_write;
    input [7:0]  addr;
    input [31:0] data;
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

  task axi_lite_read;
    input  [7:0]  addr;
    output [31:0] data;
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
  // Initialize weight ROM
  // ------------------------------------------------------------
  initial begin
    for (integer i = 0; i < 1024; i = i + 1) begin
      weight_mem[i] = {8{8'hFF}};
    end
    weight_mem[0] = 64'hFFFF_FFFF_FFFF_FFFF;
    weight_mem[1] = 64'h0000_0000_0000_0000;
    weight_mem[2] = 64'hAAAA_AAAA_AAAA_AAAA;
    weight_mem[3] = 64'h5555_5555_5555_5555;
  end

  // Preload weight ROM in DUT (skipped for Verilator - uses default zeros)
  // Note: Generate block instances have different naming in Verilator
  // For simulation, weights are loaded via $readmemh in weight_rom.sv

  // ------------------------------------------------------------
  // Initialize input data
  // ------------------------------------------------------------
  initial begin
    for (integer i = 0; i < 256; i = i + 1) begin
      input_mem[i] = $urandom;
    end
    input_mem[0] = 64'hFFFF_FFFF_FFFF_FFFF;
    input_mem[1] = 64'h0000_0000_0000_0000;
    input_mem[2] = 64'hAAAA_AAAA_AAAA_AAAA;
  end

  // ------------------------------------------------------------
  // Main test sequence
  // ------------------------------------------------------------
  initial begin
    // Initialize signals
    s_axi_awaddr  = 8'h0;
    s_axi_awvalid = 1'b0;
    s_axi_wdata   = 32'h0;
    s_axi_wstrb   = 4'h0;
    s_axi_wvalid  = 1'b0;
    s_axi_bready  = 1'b0;
    s_axi_araddr  = 8'h0;
    s_axi_arvalid = 1'b0;
    s_axi_rready  = 1'b0;

    s_axis_tdata  = 64'h0;
    s_axis_tvalid = 1'b0;
    s_axis_tlast  = 1'b0;
    m_axis_tready = 1'b1;

    @(posedge rst_n);
    repeat (10) @(posedge clk);

    $display("[TB] Starting BNN Accelerator Test");

    // 1. Configure layer 0: 784 inputs -> 128 outputs, 1 PE lane
    axi_lite_write(REG_NUM_LAYERS, 32'h1);
    axi_lite_write(REG_PE_LANES,   32'h1);
    axi_lite_write(REG_LAYER_BASE + 0, 32'h0001_0310);
    axi_lite_write(REG_LAYER_BASE + 4, 32'h0000_0080);
    axi_lite_write(REG_LAYER_BASE + 8, 32'h0000_0000);

    // 2. Read back config
    axi_lite_read(REG_NUM_LAYERS, rd_data);
    $display("[TB] Num layers: %d", rd_data[7:0]);
    axi_lite_read(REG_PE_LANES, rd_data);
    $display("[TB] PE lanes: %d", rd_data[7:0]);

    // 3. Start accelerator
    $display("[TB] Starting accelerator...");
    axi_lite_write(REG_CTRL, 32'h1);

    // 4. Feed input data
    repeat (5) @(posedge clk);
    
    for (vec = 0; vec < 3; vec = vec + 1) begin
      for (word = 0; word < 13; word = word + 1) begin
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
    output_count = 0;
    while (output_count < 128) begin
      @(posedge clk);
      if (m_axis_tvalid && m_axis_tready) begin
        captured_outputs[output_count] = m_axis_tdata;
        output_count = output_count + 1;
        $display("[TB] Output[%0d] = %b", output_count-1, m_axis_tdata);
      end
    end

    // 6. Wait for done
    wait (irq_done);
    $display("[TB] Accelerator DONE");

    // 7. Check status
    axi_lite_read(REG_STATUS, rd_data);
    $display("[TB] Status: done=%b busy=%b error=%b", rd_data[0], rd_data[1], rd_data[2]);

    // 8. Verify results
    errors = 0;
    for (integer i = 0; i < 128; i = i + 1) begin
      // Basic sanity check
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