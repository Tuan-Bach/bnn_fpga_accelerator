/**
 * MNIST BNN Verification Testbench - Deep 4-layer (784->2048->2048->2048->10)
 * Loads real MNIST test images and trained weights,
 * runs binary hidden layers through PE hardware (XNOR-popcount),
 * computes float output layer in software.
 */

#include <verilated.h>
#include "Vpe_unit.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <cmath>

vluint64_t sim_time = 0;

void tick(Vpe_unit* dut) {
    dut->clk = 1; dut->eval(); sim_time++;
    dut->clk = 0; dut->eval(); sim_time++;
}

void reset_pe(Vpe_unit* dut) {
    dut->data_in = 0;
    dut->weight_in = 0xFFFFFFFFFFFFFFFFULL;
    dut->thresh_in = 0;
    dut->last_in_batch = 1;
    tick(dut);
    dut->last_in_batch = 0;
    tick(dut);
    tick(dut);
}

std::vector<uint64_t> load_u64(const char* path) {
    std::vector<uint64_t> d; FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return d; }
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '/' || line[0] == '\n' || line[0] == '\0') continue;
        char* p = line; while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0' || *p == '\n') continue;
        d.push_back(strtoull(p, nullptr, 16));
    }
    fclose(f); return d;
}

std::vector<uint16_t> load_u16(const char* path) {
    std::vector<uint16_t> d; FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return d; }
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '/' || line[0] == '\n' || line[0] == '\0') continue;
        char* p = line; while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0' || *p == '\n') continue;
        d.push_back((uint16_t)strtoul(p, nullptr, 16));
    }
    fclose(f); return d;
}

std::vector<int> load_labels(const char* path) {
    std::vector<int> d; FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return d; }
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '/' || line[0] == '\n' || line[0] == '\0') continue;
        char* p = line; while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0' || *p == '\n') continue;
        d.push_back((int)strtoul(p, nullptr, 16));
    }
    fclose(f); return d;
}

/**
 * Run XNOR-popcount for one neuron through PE hardware.
 * Returns accumulated popcount.
 */
int64_t run_neuron(
    Vpe_unit* pe,
    const uint64_t* inputs, int n_words,
    const uint64_t* weights
) {
    reset_pe(pe);
    for (int w = 0; w < n_words; w++) {
        pe->data_in = inputs[w];
        pe->weight_in = weights[w];
        pe->thresh_in = 0;
        pe->last_in_batch = (w == n_words - 1) ? 1 : 0;
        tick(pe);
    }
    return pe->acc_out;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    const int N_INPUT   = 784;
    const int N_H1      = 2048;
    const int N_H2      = 2048;
    const int N_H3      = 2048;
    const int N_OUTPUT  = 10;

    const int NW_IN  = 13;  // ceil(784/64)
    const int NW_H   = 32;  // ceil(2048/64)

    Vpe_unit* pe = new Vpe_unit;

    // Reset
    pe->rst_n = 0; pe->enable = 1; pe->clk = 0;
    pe->data_in = 0; pe->weight_in = 0; pe->last_in_batch = 0; pe->thresh_in = 0;
    for (int i = 0; i < 10; i++) tick(pe);
    pe->rst_n = 1;
    for (int i = 0; i < 5; i++) tick(pe);

    printf("=== MNIST BNN Verification (Deep: 784->2048->2048->2048->10) ===\n\n");

    // Load weights
    printf("Loading weights...\n");
    auto w1 = load_u64("tb/mnist_data/layer1_weights.mem");
    auto w2 = load_u64("tb/mnist_data/layer2_weights.mem");
    auto w3 = load_u64("tb/mnist_data/layer3_weights.mem");
    auto w4_raw = load_u64("tb/mnist_data/layer4_weights.mem");
    auto thresh1 = load_u16("tb/mnist_data/layer1_threshold.mem");
    auto thresh2 = load_u16("tb/mnist_data/layer2_threshold.mem");
    auto thresh3 = load_u16("tb/mnist_data/layer3_threshold.mem");
    auto bias4 = load_u16("tb/mnist_data/layer4_bias.mem");

    if (w1.empty() || w2.empty() || w3.empty() || w4_raw.empty()) {
        printf("ERROR: Run 'python3 python/export_bnn_deep.py' first\n");
        return 1;
    }

    int w1_words = w1.size() / N_H1;
    int w2_words = w2.size() / N_H2;
    int w3_words = w3.size() / N_H3;
    printf("Layer1: %d->%d (%d words/neuron)\n", N_INPUT, N_H1, w1_words);
    printf("Layer2: %d->%d (%d words/neuron)\n", N_H1, N_H2, w2_words);
    printf("Layer3: %d->%d (%d words/neuron)\n", N_H2, N_H3, w3_words);
    printf("Layer4: %d->%d (float Q4.12)\n", N_H3, N_OUTPUT);

    // Load test data
    printf("Loading test data...\n");
    auto images = load_u64("tb/mnist_data/test_images.mem");
    auto labels = load_labels("tb/mnist_data/test_labels.mem");

    int n_test = labels.size();
    int words_per_img = images.size() / n_test;
    printf("Images: %d (%d words each)\n\n", n_test, words_per_img);

    int correct = 0;
    std::vector<int64_t> acc_h1(N_H1), acc_h2(N_H2), acc_h3(N_H3);
    std::vector<int> out_h1(N_H1), out_h2(N_H2), out_h3(N_H3);
    float scores[N_OUTPUT];

    // Stats for verbose
    int display_count = 0;

    for (int img = 0; img < n_test; img++) {
        const uint64_t* img_data = &images[img * words_per_img];

        uint64_t packed1[NW_H] = {0};
        uint64_t packed2[NW_H] = {0};
        uint64_t packed3[NW_H] = {0};

        // ---- Layer 1: 784 -> 2048 ----
        for (int h = 0; h < N_H1; h++) {
            const uint64_t* nw_w = &w1[h * w1_words];
            acc_h1[h] = run_neuron(pe, img_data, words_per_img, nw_w);
        }
        for (int h = 0; h < N_H1; h++) {
            int16_t thr = (h < (int)thresh1.size()) ? (int16_t)thresh1[h] : 0;
            out_h1[h] = (acc_h1[h] > thr) ? 1 : 0;
        }
        for (int i = 0; i < N_H1; i++)
            if (out_h1[i]) packed1[i / 64] |= (1ULL << (i % 64));

        // ---- Layer 2: 2048 -> 2048 ----
        for (int h = 0; h < N_H2; h++) {
            const uint64_t* nw_w = &w2[h * w2_words];
            acc_h2[h] = run_neuron(pe, packed1, NW_H, nw_w);
        }
        for (int h = 0; h < N_H2; h++) {
            int16_t thr = (h < (int)thresh2.size()) ? (int16_t)thresh2[h] : 0;
            out_h2[h] = (acc_h2[h] > thr) ? 1 : 0;
        }
        for (int i = 0; i < N_H2; i++)
            if (out_h2[i]) packed2[i / 64] |= (1ULL << (i % 64));

        // ---- Layer 3: 2048 -> 2048 ----
        for (int h = 0; h < N_H3; h++) {
            const uint64_t* nw_w = &w3[h * w3_words];
            acc_h3[h] = run_neuron(pe, packed2, NW_H, nw_w);
        }
        for (int h = 0; h < N_H3; h++) {
            int16_t thr = (h < (int)thresh3.size()) ? (int16_t)thresh3[h] : 0;
            out_h3[h] = (acc_h3[h] > thr) ? 1 : 0;
        }
        for (int i = 0; i < N_H3; i++)
            if (out_h3[i]) packed3[i / 64] |= (1ULL << (i % 64));

        // ---- Layer 4: 2048 -> 10 (float Q4.12, software) ----
        int w4_words = w4_raw.size() / N_OUTPUT;  // words per neuron (weights only)
        for (int o = 0; o < N_OUTPUT; o++) {
            float sum = 0;
            for (int k = 0; k < w4_words; k++) {
                uint64_t word = w4_raw[o * w4_words + k];
                for (int j = 0; j < 4; j++) {
                    int idx = k * 4 + j;
                    if (idx >= N_H3) break;
                    int16_t qw = (int16_t)((word >> (j * 16)) & 0xFFFF);
                    sum += out_h3[idx] * (qw / 4096.0f);
                }
            }
            // Add bias (Q4.12)
            if (o < (int)bias4.size()) {
                int16_t qb = (int16_t)bias4[o];
                sum += qb / 4096.0f;
            }
            scores[o] = sum;
        }

        // Argmax
        int pred = 0;
        for (int o = 1; o < N_OUTPUT; o++)
            if (scores[o] > scores[pred]) pred = o;

        if (pred == labels[img]) correct++;

        if (display_count < 20) {
            printf("  [%3d] True:%d Pred:%d %s  scores=[",
                   img, labels[img], pred, pred == labels[img] ? "OK " : "MISS");
            for (int o = 0; o < N_OUTPUT; o++) printf("%.1f ", scores[o]);
            printf("]\n");
            display_count++;
        }
    }

    float acc = (float)correct / n_test * 100.0f;
    printf("\n========================================\n");
    printf("MNIST Verification Results (deep BNN)\n");
    printf("  Correct: %d / %d\n", correct, n_test);
    printf("  Accuracy: %.2f%%\n", acc);
    printf("========================================\n");

    delete pe;
    return (acc > 10.0f) ? 0 : 1;
}