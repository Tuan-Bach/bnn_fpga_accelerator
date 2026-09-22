/**
 * BNN Verification Testbench (dataset-parametric)
 *
 * Runs any exported {in_features}->hidden^3->n_classes BNN through the PE
 * hardware (XNOR-popcount), then computes the float output layer in software.
 *
 * Reads model_info.txt + layer*.mem from a dataset directory given as argv[1]
 * (default: tb/mnist_data). Topology is derived at runtime - no recompile
 * needed when switching datasets.
 *
 * Usage: ./obj_dir_mnist/Vpe_unit [data_dir]
 */

#include <verilated.h>
#include "Vpe_unit.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <string>

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

/** Run XNOR-popcount for one neuron through the PE; returns popcount. */
int64_t run_neuron(Vpe_unit* pe, const uint64_t* inputs, int n_words,
                   const uint64_t* weights) {
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

/** Parse "key value" lines from model_info.txt. */
std::string get_info(const std::string& path, const std::string& key,
                     std::string& value) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return "";
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        std::string s(line);
        if (s.rfind(key, 0) == 0 && s[key.size()] == ' ') {
            value = s.substr(key.size() + 1);
            // trim trailing newline/space
            while (!value.empty() && (value.back() == '\n' || value.back() == '\r'
                                      || value.back() == ' '))
                value.pop_back();
            break;
        }
    }
    fclose(f);
    return value;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    std::string data_dir = (argc > 1) ? argv[1] : "tb/mnist_data";
    std::string info_path = data_dir + "/model_info.txt";

    std::string n_input_s, n_hidden_s, hidden_s, n_output_s, dataset;
    if (get_info(info_path, "dataset", dataset).empty() ||
        get_info(info_path, "n_input", n_input_s).empty() ||
        get_info(info_path, "hidden", hidden_s).empty() ||
        get_info(info_path, "n_output", n_output_s).empty()) {
        printf("ERROR: cannot read model_info.txt from %s\n"
               "Run: python3 python/export_bnn_dataset.py --dataset <name>\n",
               data_dir.c_str());
        return 1;
    }
    std::string junk;
    get_info(info_path, "n_hidden", junk);

    const int N_INPUT  = atoi(n_input_s.c_str());
    const int N_HIDDEN = atoi(hidden_s.c_str());   // all hidden layers equal width
    const int N_OUTPUT = atoi(n_output_s.c_str());
    const int NW_IN = (N_INPUT + 63) / 64;
    const int NW_H  = (N_HIDDEN + 63) / 64;

    Vpe_unit* pe = new Vpe_unit;

    // Reset
    pe->rst_n = 0; pe->enable = 1; pe->clk = 0;
    pe->data_in = 0; pe->weight_in = 0; pe->last_in_batch = 0; pe->thresh_in = 0;
    for (int i = 0; i < 10; i++) tick(pe);
    pe->rst_n = 1;
    for (int i = 0; i < 5; i++) tick(pe);

    printf("=== BNN Verification: %s (%d->%d^3->%d) ===\n\n",
           dataset.c_str(), N_INPUT, N_HIDDEN, N_OUTPUT);

    // ---- Load weights & thresholds ----
    printf("Loading weights...\n");
    std::vector<uint64_t> W[3];
    std::vector<uint16_t> THR[3];
    std::vector<uint64_t> w4_raw;
    std::vector<uint16_t> bias4;
    for (int l = 0; l < 3; l++) {
        char wpath[256], tpath[256];
        snprintf(wpath, sizeof(wpath), "%s/layer%d_weights.mem", data_dir.c_str(), l + 1);
        snprintf(tpath, sizeof(tpath), "%s/layer%d_threshold.mem", data_dir.c_str(), l + 1);
        W[l] = load_u64(wpath);
        THR[l] = load_u16(tpath);
    }
    w4_raw = load_u64((data_dir + "/layer4_weights.mem").c_str());
    bias4 = load_u16((data_dir + "/layer4_bias.mem").c_str());

    if (W[0].empty() || W[1].empty() || W[2].empty() || w4_raw.empty()) {
        printf("ERROR: missing .mem files. Run export first.\n");
        return 1;
    }

    int w_words[3];
    for (int l = 0; l < 3; l++) {
        w_words[l] = (int)(W[l].size() / N_HIDDEN);
        printf("Layer%d: words/neuron = %d (%zu bytes)\n", l + 1,
               w_words[l], W[l].size() * 8);
    }

    // ---- Load test data ----
    printf("Loading test data...\n");
    auto images = load_u64((data_dir + "/test_images.mem").c_str());
    auto labels = load_labels((data_dir + "/test_labels.mem").c_str());
    if (images.empty() || labels.empty()) {
        printf("ERROR: missing test data files\n");
        return 1;
    }

    int n_test = (int)labels.size();
    int words_per_img = (int)(images.size() / n_test);
    printf("Images: %d (%d words each)\n\n", n_test, words_per_img);

    int correct = 0;
    int w4_words = (int)(w4_raw.size() / N_OUTPUT);
    int display_count = 0;

    std::vector<int64_t> acc(N_HIDDEN);
    std::vector<int> out(N_HIDDEN);
    std::vector<float> scores(N_OUTPUT);

    for (int img = 0; img < n_test; img++) {
        const uint64_t* img_data = &images[(size_t)img * words_per_img];

        // current input words: layer1 uses image words, then packed hidden activations
        std::vector<uint64_t> packed(NW_H, 0);
        std::vector<uint64_t> packed_next(NW_H, 0);
        const uint64_t* cur_input = img_data;
        int cur_words = words_per_img;

        // ---- Hidden layers 1..3 through PE hardware ----
        for (int l = 0; l < 3; l++) {
            for (int h = 0; h < N_HIDDEN; h++) {
                const uint64_t* nw_w = &W[l][(size_t)h * w_words[l]];
                acc[h] = run_neuron(pe, cur_input, cur_words, nw_w);
            }
            for (int h = 0; h < N_HIDDEN; h++) {
                int16_t thr = (h < (int)THR[l].size()) ? (int16_t)THR[l][h] : 0;
                out[h] = (acc[h] > thr) ? 1 : 0;
            }
            for (int i = 0; i < N_HIDDEN; i++)
                packed_next[i / 64] = 0;
            for (int i = 0; i < N_HIDDEN; i++)
                if (out[i]) packed_next[i / 64] |= (1ULL << (i % 64));
            // next layer consumes packed activations
            cur_input = packed_next.data();
            cur_words = NW_H;
        }

        // ---- Output layer: hidden -> N_OUTPUT (float Q4.12, software) ----
        for (int o = 0; o < N_OUTPUT; o++) {
            float sum = 0;
            for (int k = 0; k < w4_words; k++) {
                uint64_t word = w4_raw[(size_t)o * w4_words + k];
                for (int j = 0; j < 4; j++) {
                    int idx = k * 4 + j;
                    if (idx >= N_HIDDEN) break;
                    int16_t qw = (int16_t)((word >> (j * 16)) & 0xFFFF);
                    sum += out[idx] * (qw / 4096.0f);
                }
            }
            if (o < (int)bias4.size()) {
                int16_t qb = (int16_t)bias4[o];
                sum += qb / 4096.0f;
            }
            scores[o] = sum;
        }

        int pred = 0;
        for (int o = 1; o < N_OUTPUT; o++)
            if (scores[o] > scores[pred]) pred = o;

        if (pred == labels[img]) correct++;

        if (display_count < 20) {
            printf("  [%3d] True:%d Pred:%d %s", img, labels[img], pred,
                   pred == labels[img] ? "OK " : "MISS");
            if (N_OUTPUT <= 10) {
                printf("  scores=[");
                for (int o = 0; o < N_OUTPUT; o++) printf("%.1f ", scores[o]);
                printf("]");
            }
            printf("\n");
            display_count++;
        }
    }

    float acc_pct = (float)correct / n_test * 100.0f;
    printf("\n========================================\n");
    printf("BNN Verification Results (%s)\n", dataset.c_str());
    printf("  Correct: %d / %d\n", correct, n_test);
    printf("  Accuracy: %.2f%%\n", acc_pct);
    printf("========================================\n");

    delete pe;
    return (acc_pct > 10.0f) ? 0 : 1;
}