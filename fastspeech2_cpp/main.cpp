#include "fastspeech2.h"
#include "op.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

// ============================================================================
// Helper Functions
// ============================================================================

void print_usage(const char* program_name) {
    printf("FastSpeech2 C++ Inference Engine\n");
    printf("\nUsage:\n");
    printf("  %s [OPTIONS]\n\n", program_name);
    printf("Options:\n");
    printf("  --config <path>       Path to config.bin file (required)\n");
    printf("  --weights <dir>       Path to weights directory (required)\n");
    printf("  --input <path>        Path to input phoneme IDs file (.txt or .bin)\n");
    printf("  --phonemes <ids>      Phoneme IDs as comma-separated integers\n");
    printf("  --output <path>       Output mel-spectrogram file (.bin)\n");
    printf("  --help                Show this help message\n");
    printf("\nExample:\n");
    printf("  %s --config weights/config.bin --weights weights/ \\\n", program_name);
    printf("            --phonemes \"23,15,8,32,45,12\" --output output.bin\n");
    printf("\n");
}

std::vector<int> load_phoneme_ids_from_text(const char* path) {
    std::vector<int> ids;
    std::ifstream file(path);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Could not open %s\n", path);
        return ids;
    }

    int id;
    while (file >> id) {
        ids.push_back(id);
    }

    file.close();
    return ids;
}

std::vector<int> load_phoneme_ids_from_binary(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open %s\n", path);
        return std::vector<int>();
    }

    // Get file size
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    int n_ids = file_size / sizeof(int);
    std::vector<int> ids(n_ids);

    size_t read = fread(ids.data(), sizeof(int), n_ids, f);
    fclose(f);

    if (read != (size_t)n_ids) {
        fprintf(stderr, "Error: Read %zu ids but expected %d\n", read, n_ids);
        return std::vector<int>();
    }

    return ids;
}

std::vector<int> parse_phoneme_string(const char* phoneme_str) {
    std::vector<int> ids;
    std::stringstream ss(phoneme_str);
    std::string token;

    while (std::getline(ss, token, ',')) {
        // Trim whitespace
        size_t start = token.find_first_not_of(" \t");
        size_t end = token.find_last_not_of(" \t");
        if (start != std::string::npos && end != std::string::npos) {
            token = token.substr(start, end - start + 1);
        }

        try {
            int id = std::stoi(token);
            ids.push_back(id);
        } catch (...) {
            fprintf(stderr, "Warning: Could not parse '%s' as integer\n", token.c_str());
        }
    }

    return ids;
}

void save_mel_spectrogram(const char* path, float* mel, int mel_len, int n_mels) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Error: Could not open %s for writing\n", path);
        return;
    }

    // Write shape information (metadata)
    int shape[2] = {mel_len, n_mels};
    fwrite(shape, sizeof(int), 2, f);

    // Write mel data
    fwrite(mel, sizeof(float), mel_len * n_mels, f);

    fclose(f);

    printf("✓ Saved mel-spectrogram to %s\n", path);
    printf("  Shape: [%d, %d]\n", mel_len, n_mels);
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char* argv[]) {
    // Default values
    const char* config_path = nullptr;
    const char* weights_dir = nullptr;
    const char* input_path = nullptr;
    const char* phoneme_str = nullptr;
    const char* output_path = "output_mel.bin";

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
        } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            weights_dir = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (strcmp(argv[i], "--phonemes") == 0 && i + 1 < argc) {
            phoneme_str = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate required arguments
    if (!config_path) {
        fprintf(stderr, "Error: --config is required\n");
        print_usage(argv[0]);
        return 1;
    }

    if (!weights_dir) {
        fprintf(stderr, "Error: --weights is required\n");
        print_usage(argv[0]);
        return 1;
    }

    if (!input_path && !phoneme_str) {
        fprintf(stderr, "Error: Either --input or --phonemes is required\n");
        print_usage(argv[0]);
        return 1;
    }

    printf("========================================\n");
    printf("FastSpeech2 C++ Inference\n");
    printf("========================================\n\n");

    // Load configuration
    printf("1. Loading configuration...\n");
    Config config;
    load_config(config_path, &config);
    printf("\n");

    // Create weights and load
    printf("2. Loading model weights...\n");
    Weights* weights = create_weights(&config);
    load_weights(weights_dir, weights, &config);
    printf("\n");

    // Load phoneme IDs
    printf("3. Loading phoneme IDs...\n");
    std::vector<int> phoneme_ids;

    if (phoneme_str) {
        phoneme_ids = parse_phoneme_string(phoneme_str);
        printf("  Parsed %zu phoneme IDs from command line\n", phoneme_ids.size());
    } else if (input_path) {
        // Detect format by extension
        const char* ext = strrchr(input_path, '.');
        if (ext && strcmp(ext, ".bin") == 0) {
            phoneme_ids = load_phoneme_ids_from_binary(input_path);
        } else {
            phoneme_ids = load_phoneme_ids_from_text(input_path);
        }
        printf("  Loaded %zu phoneme IDs from %s\n", phoneme_ids.size(), input_path);
    }

    if (phoneme_ids.empty()) {
        fprintf(stderr, "Error: No phoneme IDs loaded\n");
        return 1;
    }

    // Print phoneme IDs
    printf("  Phoneme IDs: [");
    for (size_t i = 0; i < phoneme_ids.size(); i++) {
        printf("%d", phoneme_ids[i]);
        if (i < phoneme_ids.size() - 1) printf(", ");
    }
    printf("]\n\n");

    // Create runtime state
    printf("4. Initializing runtime state...\n");
    RunState* state = create_run_state(&config);
    printf("  ✓ State initialized\n\n");

    // Run inference
    printf("5. Running inference...\n");
    printf("========================================\n");
    fastspeech2_forward(state, &config, weights, phoneme_ids.data(), phoneme_ids.size());
    printf("========================================\n\n");

    // Save output
    printf("6. Saving output...\n");
    save_mel_spectrogram(output_path, state->postnet_mel_out, state->mel_len, config.n_mels);
    printf("\n");

    // Print statistics
    printf("========================================\n");
    printf("Inference Statistics:\n");
    printf("  Input phonemes:  %zu\n", phoneme_ids.size());
    printf("  Output frames:   %d\n", state->mel_len);
    printf("  Mel bins:        %d\n", config.n_mels);
    printf("  Avg duration:    %.2f frames/phoneme\n",
           (float)state->mel_len / phoneme_ids.size());
    printf("========================================\n");

    // Cleanup
    free_run_state(state);
    free_weights(weights, &config);

    printf("\n✓ Inference complete!\n\n");

    return 0;
}
