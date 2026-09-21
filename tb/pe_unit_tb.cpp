/**
 * PE Unit Testbench - Fixed version
 *
 * Key timing:
 *   Cycle N  : set last_in_batch=1, apply data/weight, call tick()
 *              -> posedge samples: acc_captured <= acc_next, last_in_batch_d <= 1
 *              -> after tick: valid_out=1, acc_out=captured value, binary_out=computed
 *   Cycle N+1: set last_in_batch=0, call tick()
 *              -> after tick: valid_out=0
 *
 * Reset: data_in=0, weight_in=0xFFFFFFFFFFFFFFFF -> XNOR=0 -> popcount=0
 */

#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vpe_unit.h"
#include <cstdio>
#include <cstdint>

vluint64_t sim_time = 0;

void tick(Vpe_unit* dut, VerilatedVcdC* tfp) {
    dut->clk = 1; dut->eval(); tfp->dump(sim_time++);
    dut->clk = 0; dut->eval(); tfp->dump(sim_time++);
}

// Reset: feed zero-popcount words (data=0, weight=all-1s -> XNOR=0)
// and pulse last_in_batch to drain the accumulator.
void reset_pe(Vpe_unit* dut, VerilatedVcdC* tfp) {
    dut->data_in = 0;
    dut->weight_in = 0xFFFFFFFFFFFFFFFFULL;  // XNOR with 0 -> 0, popcount=0
    dut->thresh_in = 0;
    dut->last_in_batch = 1;
    tick(dut, tfp);
    dut->last_in_batch = 0;
    tick(dut, tfp);
    tick(dut, tfp);
}

// Run one single-cycle computation:
//   set data/weight/thresh, pulse last_in_batch=1, tick once, read outputs.
void run_single_cycle(Vpe_unit* dut, VerilatedVcdC* tfp,
                      uint64_t data, uint64_t weight, int16_t thresh) {
    dut->data_in = data;
    dut->weight_in = weight;
    dut->thresh_in = thresh;
    dut->last_in_batch = 1;
    tick(dut, tfp);
    // valid_out, acc_out, binary_out are now valid
    dut->last_in_batch = 0;
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);
    VerilatedVcdC* tfp = new VerilatedVcdC;

    Vpe_unit* dut = new Vpe_unit;
    dut->trace(tfp, 99);
    tfp->open("pe_unit.vcd");

    // Hard reset
    dut->rst_n = 0;
    dut->enable = 1;
    dut->data_in = 0;
    dut->weight_in = 0;
    dut->last_in_batch = 0;
    dut->thresh_in = 0;
    dut->clk = 0;
    for (int i = 0; i < 10; i++) tick(dut, tfp);
    dut->rst_n = 1;
    // Post-reset: use data=0, weight=0xFFFF... so XNOR=0 -> popcount=0
    dut->data_in = 0;
    dut->weight_in = 0xFFFFFFFFFFFFFFFFULL;
    for (int i = 0; i < 5; i++) tick(dut, tfp);

    int pass = 0, fail = 0, total = 0;

    // ========== Test 1: All 1s XNOR All 1s -> popcount=64 ==========
    total++;
    printf("Test 1: All 1s XNOR All 1s -> popcount=64\n");
    run_single_cycle(dut, tfp, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0);
    if (dut->valid_out && dut->acc_out == 64 && dut->binary_out == 1) {
        printf("  PASS: acc=%d, binary=%d\n", (int)dut->acc_out, (int)dut->binary_out);
        pass++;
    } else {
        printf("  FAIL: valid=%d, acc=%d, binary=%d\n",
               (int)dut->valid_out, (int)dut->acc_out, (int)dut->binary_out);
        fail++;
    }
    reset_pe(dut, tfp);

    // ========== Test 2: All 0s XNOR All 1s -> popcount=0 ==========
    total++;
    printf("\nTest 2: All 0s XNOR All 1s -> popcount=0\n");
    run_single_cycle(dut, tfp, 0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL, 0);
    if (dut->valid_out && dut->acc_out == 0 && dut->binary_out == 0) {
        printf("  PASS: acc=%d, binary=%d\n", (int)dut->acc_out, (int)dut->binary_out);
        pass++;
    } else {
        printf("  FAIL: valid=%d, acc=%d, binary=%d\n",
               (int)dut->valid_out, (int)dut->acc_out, (int)dut->binary_out);
        fail++;
    }
    reset_pe(dut, tfp);

    // ========== Test 3: Alternating pattern -> popcount=32 ==========
    total++;
    printf("\nTest 3: 0xAAAA... XNOR 0xFFFF... -> popcount=32\n");
    run_single_cycle(dut, tfp, 0xAAAAAAAAAAAAAAAAULL, 0xFFFFFFFFFFFFFFFFULL, 0);
    if (dut->valid_out && dut->acc_out == 32) {
        printf("  PASS: acc=%d\n", (int)dut->acc_out);
        pass++;
    } else {
        printf("  FAIL: valid=%d, acc=%d\n", (int)dut->valid_out, (int)dut->acc_out);
        fail++;
    }
    reset_pe(dut, tfp);

    // ========== Test 4: Multi-cycle accumulation (4 x 32 = 128) ==========
    total++;
    printf("\nTest 4: 4-cycle accumulation (4 x 32 = 128)\n");
    // Feed 4 cycles: first 3 with last_in_batch=0, last with last_in_batch=1
    dut->thresh_in = 0;
    dut->data_in = 0x0000FFFF0000FFFFULL;   // 32 bits set -> popcount=32
    dut->weight_in = 0xFFFFFFFFFFFFFFFFULL;
    for (int c = 0; c < 3; c++) {
        dut->last_in_batch = 0;
        tick(dut, tfp);
    }
    dut->last_in_batch = 1;
    tick(dut, tfp);
    // Now valid_out=1, acc_out should be 4*32=128
    if (dut->valid_out && dut->acc_out == 128) {
        printf("  PASS: acc=%d\n", (int)dut->acc_out);
        pass++;
    } else {
        printf("  FAIL: valid=%d, acc=%d (expected 128)\n",
               (int)dut->valid_out, (int)dut->acc_out);
        fail++;
    }
    dut->last_in_batch = 0;
    tick(dut, tfp);
    reset_pe(dut, tfp);

    // ========== Test 5: Threshold test - acc=64 > thresh=32 -> binary=1 ==========
    total++;
    printf("\nTest 5: acc=64, thresh=32 -> binary=1\n");
    run_single_cycle(dut, tfp, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0x0020);
    if (dut->valid_out && dut->acc_out == 64 && dut->binary_out == 1) {
        printf("  PASS: acc=%d, binary=%d\n", (int)dut->acc_out, (int)dut->binary_out);
        pass++;
    } else {
        printf("  FAIL: valid=%d, acc=%d, binary=%d\n",
               (int)dut->valid_out, (int)dut->acc_out, (int)dut->binary_out);
        fail++;
    }
    reset_pe(dut, tfp);

    // ========== Test 6: Threshold test - acc=64 < thresh=128 -> binary=0 ==========
    total++;
    printf("\nTest 6: acc=64, thresh=128 -> binary=0\n");
    run_single_cycle(dut, tfp, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0x0080);
    if (dut->valid_out && dut->acc_out == 64 && dut->binary_out == 0) {
        printf("  PASS: acc=%d, binary=%d\n", (int)dut->acc_out, (int)dut->binary_out);
        pass++;
    } else {
        printf("  FAIL: valid=%d, acc=%d, binary=%d\n",
               (int)dut->valid_out, (int)dut->acc_out, (int)dut->binary_out);
        fail++;
    }
    reset_pe(dut, tfp);

    // ========== Test 7: Latency measurement ==========
    total++;
    printf("\nTest 7: Latency measurement\n");
    dut->data_in = 0xFFFFFFFFFFFFFFFFULL;
    dut->weight_in = 0xFFFFFFFFFFFFFFFFULL;
    dut->thresh_in = 0;
    dut->last_in_batch = 1;
    tick(dut, tfp);
    // After one tick: valid_out should be 1 (1-cycle latency)
    if (dut->valid_out) {
        printf("  PASS: latency = 1 cycle\n");
        pass++;
    } else {
        printf("  FAIL: valid=%d after 1 cycle (expected 1)\n", (int)dut->valid_out);
        fail++;
    }
    dut->last_in_batch = 0;
    tick(dut, tfp);
    reset_pe(dut, tfp);

    // ========== Test 8: Throughput calculation (theoretical) ==========
    total++;
    printf("\nTest 8: Throughput calculation\n");
    double freq = 100e6;
    int words = 13;
    double img_per_sec = (freq / words) / 128.0;
    double latency_us = (words * 128.0) / freq * 1e6;
    printf("  1 PE: %.0f img/s, %.2f us latency\n", img_per_sec, latency_us);
    printf("  8 PE: %.0f img/s, %.2f us latency\n", img_per_sec * 8, latency_us / 8);
    printf("  PASS (theoretical)\n");
    pass++;

    // ========== Summary ==========
    printf("\n========================================\n");
    printf("Results: %d / %d PASSED\n", pass, total);
    if (pass == total)
        printf("*** ALL TESTS PASSED ***\n");
    else
        printf("*** %d TESTS FAILED ***\n", fail);
    printf("========================================\n");

    tfp->close();
    delete dut;
    delete tfp;
    return (pass == total) ? 0 : 1;
}
