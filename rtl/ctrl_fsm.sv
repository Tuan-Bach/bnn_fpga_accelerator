`ifndef CTRL_FSM_SV
`define CTRL_FSM_SV

module ctrl_fsm #(
  parameter int MAX_LAYERS   = 4,
  parameter int MAX_PE_LANES = 8
)(
  input  logic                    clk,
  input  logic                    rst_n,
  // Configuration
  input  logic                    start,
  input  logic [7:0]              num_layers,
  input  logic [7:0]              num_pe_lanes,
  input  bnn_pkg::layer_cfg_t     layer_cfg [MAX_LAYERS-1:0],
  // Input buffer status
  input  logic                    input_buffer_empty,
  input  logic                    input_buffer_valid,
  // PE array status
  input  logic [MAX_PE_LANES-1:0] pe_valid_out,
  // Weight ROM
  output logic [$clog2(1024)-1:0] weight_addr,
  output logic                    weight_rom_en,
  input  logic [63:0]             weight_rom_data [MAX_PE_LANES-1:0],
  // Output control
  output logic                    layer_done,
  output logic                    all_done,
  output logic                    error,
  // Debug
  output logic [3:0]              current_layer,
  output logic [11:0]             current_neuron,
  output logic [2:0]              fsm_state
);

  import bnn_pkg::*;

  typedef enum logic [2:0] {
    S_IDLE       = 3'd0,
    S_LOAD_CFG   = 3'd1,
    S_WAIT_INPUT = 3'd2,
    S_COMPUTE    = 3'd3,
    S_LAYER_DONE = 3'd4,
    S_ALL_DONE   = 3'd5,
    S_ERROR      = 3'd7
  } state_e;

  state_e current_state, next_state;

  // Layer/neuron counters
  logic [3:0]  layer_idx;
  logic [11:0] neuron_idx;
  logic [11:0] input_feature_idx;
  logic [7:0]  pe_lanes_active;

  // Weight address calculation
  logic [$clog2(1024)-1:0] weight_base_addr;
  logic [$clog2(1024)-1:0] weight_stride;

  // State register
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      current_state <= S_IDLE;
    end else begin
      current_state <= next_state;
    end
  end

  // Next state logic
  always_comb begin
    next_state = current_state;
    case (current_state)
      S_IDLE: begin
        if (start) next_state = S_LOAD_CFG;
      end
      S_LOAD_CFG: begin
        next_state = S_WAIT_INPUT;
      end
      S_WAIT_INPUT: begin
        if (input_buffer_valid) next_state = S_COMPUTE;
      end
      S_COMPUTE: begin
        if (layer_done) next_state = S_LAYER_DONE;
        else if (error) next_state = S_ERROR;
      end
      S_LAYER_DONE: begin
        if (layer_idx == num_layers - 1) next_state = S_ALL_DONE;
        else next_state = S_WAIT_INPUT;
      end
      S_ALL_DONE: begin
        next_state = S_IDLE;
      end
      S_ERROR: begin
        if (!start) next_state = S_IDLE;
      end
      default: next_state = S_ERROR;
    endcase
  end

  // Layer/neuron counters
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      layer_idx          <= '0;
      neuron_idx         <= '0;
      input_feature_idx  <= '0;
      pe_lanes_active    <= '0;
      weight_base_addr   <= '0;
      weight_stride      <= '0;
      weight_addr        <= '0;
      weight_rom_en      <= 1'b0;
      layer_done         <= 1'b0;
      all_done           <= 1'b0;
      error              <= 1'b0;
    end else begin
      layer_done <= 1'b0;
      all_done   <= 1'b0;
      error      <= 1'b0;

      case (current_state)
        S_IDLE: begin
          layer_idx         <= '0;
          neuron_idx        <= '0;
          input_feature_idx <= '0;
        end
        S_LOAD_CFG: begin
          pe_lanes_active  <= num_pe_lanes;
          weight_base_addr <= layer_cfg[0].weight_offset;
          weight_stride    <= layer_cfg[0].input_features / 64; // words per neuron
        end
        S_WAIT_INPUT: begin
          input_feature_idx <= '0;
          neuron_idx        <= '0;
        end
        S_COMPUTE: begin
          if (input_buffer_valid) begin
            // Calculate weight address for current neuron
            weight_addr   <= weight_base_addr + neuron_idx * weight_stride + input_feature_idx;
            weight_rom_en <= 1'b1;

            // Advance input feature index
            if (input_feature_idx == layer_cfg[layer_idx].input_features/64 - 1) begin
              input_feature_idx <= '0;
              neuron_idx <= neuron_idx + 1'b1;
            end else begin
              input_feature_idx <= input_feature_idx + 1'b1;
            end

            // Check if layer complete
            if (neuron_idx == layer_cfg[layer_idx].output_features - 1 && 
                input_feature_idx == layer_cfg[layer_idx].input_features/64 - 1) begin
              layer_done <= 1'b1;
            end
          end else begin
            weight_rom_en <= 1'b0;
          end
        end
        S_LAYER_DONE: begin
          layer_idx <= layer_idx + 1'b1;
          if (layer_idx < num_layers - 1) begin
            weight_base_addr <= layer_cfg[layer_idx+1].weight_offset;
            weight_stride    <= layer_cfg[layer_idx+1].input_features / 64;
          end
          neuron_idx        <= '0;
          input_feature_idx <= '0;
        end
        S_ALL_DONE: begin
          all_done <= 1'b1;
        end
        S_ERROR: begin
          error <= 1'b1;
        end
        default: begin
          // Should not reach here
          error <= 1'b1;
        end
      endcase
    end
  end

  // Debug outputs
  assign current_layer  = layer_idx;
  assign current_neuron = neuron_idx;
  assign fsm_state      = current_state;

endmodule

`endif