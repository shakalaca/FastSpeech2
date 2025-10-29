#include "fastspeech2.h"
#include "op.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <cerrno>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#endif

typedef struct {
    bool enabled;
    std::string dir;
} DumpContext;

static DumpContext g_dump_ctx = {false, ""};

static bool dump_enabled() {
    return g_dump_ctx.enabled;
}

static void ensure_dump_dir() {
    if (!g_dump_ctx.enabled || g_dump_ctx.dir.empty()) return;
#ifdef _WIN32
    _mkdir(g_dump_ctx.dir.c_str());
#else
    if (mkdir(g_dump_ctx.dir.c_str(), 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "Warning: could not create dump directory %s\n", g_dump_ctx.dir.c_str());
    }
#endif
}

static std::string make_dump_path(const char* name) {
    std::string path = g_dump_ctx.dir;
    if (!path.empty() && path.back() != '/' && path.back() != '\\') {
        path += "/";
    }
    path += name;
    return path;
}

static void dump_float_matrix(const char* name, const float* data, int rows, int cols) {
    if (!dump_enabled()) return;
    std::string path = make_dump_path(name);
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "Warning: could not open %s for dumping\n", path.c_str());
        return;
    }
    fwrite(&rows, sizeof(int), 1, f);
    fwrite(&cols, sizeof(int), 1, f);
    fwrite(data, sizeof(float), (size_t)rows * cols, f);
    fclose(f);
}

static void dump_float_vector(const char* name, const float* data, int length) {
    dump_float_matrix(name, data, length, 1);
}

static void dump_int_vector(const char* name, const int* data, int length) {
    if (!dump_enabled()) return;
    std::string path = make_dump_path(name);
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "Warning: could not open %s for dumping\n", path.c_str());
        return;
    }
    fwrite(&length, sizeof(int), 1, f);
    int cols = 1;
    fwrite(&cols, sizeof(int), 1, f);
    fwrite(data, sizeof(int), (size_t)length, f);
    fclose(f);
}

void set_dump_directory(const char* path) {
    if (path && path[0] != '\0') {
        g_dump_ctx.enabled = true;
        g_dump_ctx.dir = path;
        ensure_dump_dir();
    } else {
        g_dump_ctx.enabled = false;
        g_dump_ctx.dir.clear();
    }
}

// ============================================================================
// Initialization and Cleanup Functions
// ============================================================================

void init_config(Config* c) {
    // Initialize with default LJSpeech config values
    c->dim = 256;
    c->n_enc_layers = 4;
    c->n_dec_layers = 6;
    c->n_heads = 2;
    c->head_dim = 128;  // dim / n_heads
    c->ffn_hidden = 1024;

    c->vocab_size = 300;
    c->n_mels = 80;
    c->max_seq_len = 1000;

    c->var_pred_filter_size = 256;
    c->var_pred_kernel_size = 3;
    c->var_pred_dropout = 0.5f;
    c->n_bins = 256;

    c->postnet_embedding_dim = 512;
    c->postnet_kernel_size = 5;
    c->postnet_n_convolutions = 5;

    c->pitch_min = -2.917f;
    c->pitch_max = 11.391f;
    c->energy_min = -1.431f;
    c->energy_max = 8.184f;
}

RunState* create_run_state(Config* c) {
    RunState* s = new RunState;

    int max_frames = c->max_seq_len * 20;  // Assume max duration ~20 frames per phoneme

    // Encoder buffers
    s->encoder_emb = new float[c->max_seq_len * c->dim]();
    s->encoder_out = new float[c->max_seq_len * c->dim]();

    // Variance Adaptor buffers
    s->duration_pred = new float[c->max_seq_len]();
    s->pitch_pred = new float[c->max_seq_len]();
    s->energy_pred = new float[c->max_seq_len]();
    s->durations = new int[c->max_seq_len]();
    s->variance_out = new float[max_frames * c->dim]();
    s->mel_len = 0;

    // Decoder buffers
    s->decoder_out = new float[max_frames * c->dim]();

    // Output buffers
    s->mel_out = new float[max_frames * c->n_mels]();
    s->postnet_mel_out = new float[max_frames * c->n_mels]();

    // Temporary attention/FFN buffers
    s->attn_q = new float[max_frames * c->dim]();
    s->attn_k = new float[max_frames * c->dim]();
    s->attn_v = new float[max_frames * c->dim]();
    s->attn_scores = new float[c->n_heads * max_frames * max_frames]();
    s->attn_out = new float[max_frames * c->dim]();
    s->ffn_hidden = new float[max_frames * c->ffn_hidden]();
    s->ffn_out = new float[max_frames * c->dim]();

    // Conv1D temporary buffers
    s->conv_buf1 = new float[max_frames * c->var_pred_filter_size]();
    s->conv_buf2 = new float[max_frames * c->var_pred_filter_size]();

    return s;
}

void free_run_state(RunState* s) {
    delete[] s->encoder_emb;
    delete[] s->encoder_out;
    delete[] s->duration_pred;
    delete[] s->pitch_pred;
    delete[] s->energy_pred;
    delete[] s->durations;
    delete[] s->variance_out;
    delete[] s->decoder_out;
    delete[] s->mel_out;
    delete[] s->postnet_mel_out;
    delete[] s->attn_q;
    delete[] s->attn_k;
    delete[] s->attn_v;
    delete[] s->attn_scores;
    delete[] s->attn_out;
    delete[] s->ffn_hidden;
    delete[] s->ffn_out;
    delete[] s->conv_buf1;
    delete[] s->conv_buf2;
    delete s;
}

Weights* create_weights(Config* c) {
    Weights* w = new Weights;

    // Encoder
    w->encoder_embedding = nullptr;
    w->encoder_pe = new float[c->max_seq_len * c->dim]();
    w->encoder_layers = new FFTLayer[c->n_enc_layers];
    memset(w->encoder_layers, 0, c->n_enc_layers * sizeof(FFTLayer));

    // Variance Adaptor - initialize structures
    memset(&w->duration_predictor, 0, sizeof(VariancePredictor));
    memset(&w->pitch_predictor, 0, sizeof(VariancePredictor));
    memset(&w->energy_predictor, 0, sizeof(VariancePredictor));
    w->pitch_embedding = nullptr;
    w->energy_embedding = nullptr;

    // Decoder
    w->decoder_pe = new float[c->max_seq_len * c->dim]();
    w->decoder_layers = new FFTLayer[c->n_dec_layers];
    memset(w->decoder_layers, 0, c->n_dec_layers * sizeof(FFTLayer));

    // Mel linear
    w->mel_linear_weight = nullptr;
    w->mel_linear_bias = nullptr;

    // PostNet
    w->postnet_layers = new PostNetLayer[c->postnet_n_convolutions];
    memset(w->postnet_layers, 0, c->postnet_n_convolutions * sizeof(PostNetLayer));

    // Generate positional encodings
    sinusoidal_position_encoding(w->encoder_pe, c->max_seq_len, c->dim);
    sinusoidal_position_encoding(w->decoder_pe, c->max_seq_len, c->dim);

    return w;
}

void free_weights(Weights* w, Config* c) {
    (void)c;
    // Free encoder
    delete[] w->encoder_embedding;
    delete[] w->encoder_pe;
    // Note: FFTLayer weights would need individual freeing if allocated
    delete[] w->encoder_layers;

    // Free variance predictor weights (would need detailed freeing if allocated)
    delete[] w->pitch_embedding;
    delete[] w->energy_embedding;

    // Free decoder
    delete[] w->decoder_pe;
    delete[] w->decoder_layers;

    // Free mel linear
    delete[] w->mel_linear_weight;
    delete[] w->mel_linear_bias;

    // Free PostNet
    delete[] w->postnet_layers;

    delete w;
}

// ============================================================================
// Weight Loading Functions
// ============================================================================

void load_config(const char* config_path, Config* c) {
    FILE* f = fopen(config_path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open config file %s\n", config_path);
        exit(1);
    }

    // Read config (matches the format from convert_weights.py)
    int int_configs[11];
    float float_configs[5];
    int int_configs2[4];

    fread(int_configs, sizeof(int), 11, f);
    fread(float_configs, sizeof(float), 5, f);
    fread(int_configs2, sizeof(int), 4, f);

    c->dim = int_configs[0];
    c->n_enc_layers = int_configs[1];
    c->n_dec_layers = int_configs[2];
    c->n_heads = int_configs[3];
    c->head_dim = int_configs[4];
    c->ffn_hidden = int_configs[5];
    c->vocab_size = int_configs[6];
    c->n_mels = int_configs[7];
    c->max_seq_len = int_configs[8];
    c->var_pred_filter_size = int_configs[9];
    c->var_pred_kernel_size = int_configs[10];

    c->var_pred_dropout = float_configs[0];
    c->pitch_min = float_configs[1];
    c->pitch_max = float_configs[2];
    c->energy_min = float_configs[3];
    c->energy_max = float_configs[4];

    c->n_bins = int_configs2[0];
    c->postnet_embedding_dim = int_configs2[1];
    c->postnet_kernel_size = int_configs2[2];
    c->postnet_n_convolutions = int_configs2[3];

    fclose(f);

    printf("✓ Loaded config: dim=%d, enc_layers=%d, dec_layers=%d\n",
           c->dim, c->n_enc_layers, c->n_dec_layers);
}

float* load_weight_file(const char* weights_dir, const char* rel_path, int expected_size,
                        int* actual_size = nullptr) {
    char full_path[512];
    snprintf(full_path, sizeof(full_path), "%s/%s", weights_dir, rel_path);

    FILE* f = fopen(full_path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open %s\n", full_path);
        return nullptr;
    }

    // Get file size
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    int n_elements = file_size / sizeof(float);
    if (actual_size) {
        *actual_size = n_elements;
    }

    if (expected_size > 0 && n_elements != expected_size) {
        fprintf(stderr, "Warning: %s has %d elements but expected %d\n",
                rel_path, n_elements, expected_size);
    }

    float* data = new float[n_elements];
    size_t read = fread(data, sizeof(float), n_elements, f);
    fclose(f);

    if (read != (size_t)n_elements) {
        fprintf(stderr, "Error: Read %zu elements but expected %d from %s\n",
                read, n_elements, rel_path);
        delete[] data;
        return nullptr;
    }

    return data;
}

void load_variance_predictor(const char* weights_dir, const char* vp_name,
                             VariancePredictor* vp, Config* c) {
    char path[256];
    int dim = c->dim;
    int filter_size = c->var_pred_filter_size;
    int kernel_size = c->var_pred_kernel_size;

    // Conv layers
    snprintf(path, sizeof(path), "variance_adaptor/%s/conv1_weight.bin", vp_name);
    vp->conv1_weight = load_weight_file(weights_dir, path, filter_size * dim * kernel_size);

    snprintf(path, sizeof(path), "variance_adaptor/%s/conv1_bias.bin", vp_name);
    vp->conv1_bias = load_weight_file(weights_dir, path, filter_size);

    snprintf(path, sizeof(path), "variance_adaptor/%s/conv2_weight.bin", vp_name);
    vp->conv2_weight = load_weight_file(weights_dir, path, filter_size * filter_size * kernel_size);

    snprintf(path, sizeof(path), "variance_adaptor/%s/conv2_bias.bin", vp_name);
    vp->conv2_bias = load_weight_file(weights_dir, path, filter_size);

    // Layer norms
    snprintf(path, sizeof(path), "variance_adaptor/%s/ln1_gamma.bin", vp_name);
    vp->ln1_gamma = load_weight_file(weights_dir, path, filter_size);

    snprintf(path, sizeof(path), "variance_adaptor/%s/ln1_beta.bin", vp_name);
    vp->ln1_beta = load_weight_file(weights_dir, path, filter_size);

    snprintf(path, sizeof(path), "variance_adaptor/%s/ln2_gamma.bin", vp_name);
    vp->ln2_gamma = load_weight_file(weights_dir, path, filter_size);

    snprintf(path, sizeof(path), "variance_adaptor/%s/ln2_beta.bin", vp_name);
    vp->ln2_beta = load_weight_file(weights_dir, path, filter_size);

    // Linear projection
    snprintf(path, sizeof(path), "variance_adaptor/%s/linear_weight.bin", vp_name);
    vp->linear_weight = load_weight_file(weights_dir, path, filter_size);

    snprintf(path, sizeof(path), "variance_adaptor/%s/linear_bias.bin", vp_name);
    vp->linear_bias = load_weight_file(weights_dir, path, 1);
}

void load_fft_layer(const char* weights_dir, const char* prefix, int layer_idx,
                    FFTLayer* layer, Config* c) {
    char path[256];
    int dim = c->dim;
    int ffn_hidden = c->ffn_hidden;
    layer->ffn_kernel_size1 = 1;
    layer->ffn_kernel_size2 = 1;

    // Attention weights
    snprintf(path, sizeof(path), "%s/layer_%d/attn_q_weight.bin", prefix, layer_idx);
    layer->attn_q_weight = load_weight_file(weights_dir, path, dim * dim);

    snprintf(path, sizeof(path), "%s/layer_%d/attn_q_bias.bin", prefix, layer_idx);
    layer->attn_q_bias = load_weight_file(weights_dir, path, dim);

    snprintf(path, sizeof(path), "%s/layer_%d/attn_k_weight.bin", prefix, layer_idx);
    layer->attn_k_weight = load_weight_file(weights_dir, path, dim * dim);

    snprintf(path, sizeof(path), "%s/layer_%d/attn_k_bias.bin", prefix, layer_idx);
    layer->attn_k_bias = load_weight_file(weights_dir, path, dim);

    snprintf(path, sizeof(path), "%s/layer_%d/attn_v_weight.bin", prefix, layer_idx);
    layer->attn_v_weight = load_weight_file(weights_dir, path, dim * dim);

    snprintf(path, sizeof(path), "%s/layer_%d/attn_v_bias.bin", prefix, layer_idx);
    layer->attn_v_bias = load_weight_file(weights_dir, path, dim);

    snprintf(path, sizeof(path), "%s/layer_%d/attn_out_weight.bin", prefix, layer_idx);
    layer->attn_out_weight = load_weight_file(weights_dir, path, dim * dim);

    snprintf(path, sizeof(path), "%s/layer_%d/attn_out_bias.bin", prefix, layer_idx);
    layer->attn_out_bias = load_weight_file(weights_dir, path, dim);

    snprintf(path, sizeof(path), "%s/layer_%d/attn_norm_gamma.bin", prefix, layer_idx);
    layer->attn_norm_gamma = load_weight_file(weights_dir, path, dim);

    snprintf(path, sizeof(path), "%s/layer_%d/attn_norm_beta.bin", prefix, layer_idx);
    layer->attn_norm_beta = load_weight_file(weights_dir, path, dim);

    // Feed-forward weights
    snprintf(path, sizeof(path), "%s/layer_%d/ffn_w1.bin", prefix, layer_idx);
    int ffn_w1_size = 0;
    layer->ffn_w1 = load_weight_file(weights_dir, path, 0, &ffn_w1_size);
    if (layer->ffn_w1 && ffn_w1_size > 0) {
        int expected_base = dim * ffn_hidden;
        if (ffn_w1_size % expected_base != 0) {
            fprintf(stderr,
                    "Error: %s has %d elements which is incompatible with dim=%d and ffn_hidden=%d\n",
                    path, ffn_w1_size, dim, ffn_hidden);
            exit(1);
        }
        layer->ffn_kernel_size1 = ffn_w1_size / expected_base;
    }

    snprintf(path, sizeof(path), "%s/layer_%d/ffn_b1.bin", prefix, layer_idx);
    layer->ffn_b1 = load_weight_file(weights_dir, path, ffn_hidden);

    snprintf(path, sizeof(path), "%s/layer_%d/ffn_w2.bin", prefix, layer_idx);
    int ffn_w2_size = 0;
    layer->ffn_w2 = load_weight_file(weights_dir, path, 0, &ffn_w2_size);
    if (layer->ffn_w2 && ffn_w2_size > 0) {
        int expected_base = ffn_hidden * dim;
        if (ffn_w2_size % expected_base != 0) {
            fprintf(stderr,
                    "Error: %s has %d elements which is incompatible with dim=%d and ffn_hidden=%d\n",
                    path, ffn_w2_size, dim, ffn_hidden);
            exit(1);
        }
        layer->ffn_kernel_size2 = ffn_w2_size / expected_base;
    }

    snprintf(path, sizeof(path), "%s/layer_%d/ffn_b2.bin", prefix, layer_idx);
    layer->ffn_b2 = load_weight_file(weights_dir, path, dim);

    snprintf(path, sizeof(path), "%s/layer_%d/ffn_norm_gamma.bin", prefix, layer_idx);
    layer->ffn_norm_gamma = load_weight_file(weights_dir, path, dim);

    snprintf(path, sizeof(path), "%s/layer_%d/ffn_norm_beta.bin", prefix, layer_idx);
    layer->ffn_norm_beta = load_weight_file(weights_dir, path, dim);
}

void load_postnet_layer(const char* weights_dir, int layer_idx, PostNetLayer* layer,
                       Config* c, int in_channels, int out_channels) {
    char path[256];
    int kernel_size = c->postnet_kernel_size;

    snprintf(path, sizeof(path), "postnet/layer_%d/conv_weight.bin", layer_idx);
    layer->conv_weight = load_weight_file(weights_dir, path, out_channels * in_channels * kernel_size);

    snprintf(path, sizeof(path), "postnet/layer_%d/conv_bias.bin", layer_idx);
    layer->conv_bias = load_weight_file(weights_dir, path, out_channels);

    snprintf(path, sizeof(path), "postnet/layer_%d/bn_gamma.bin", layer_idx);
    layer->bn_gamma = load_weight_file(weights_dir, path, out_channels);

    snprintf(path, sizeof(path), "postnet/layer_%d/bn_beta.bin", layer_idx);
    layer->bn_beta = load_weight_file(weights_dir, path, out_channels);

    snprintf(path, sizeof(path), "postnet/layer_%d/bn_mean.bin", layer_idx);
    layer->bn_mean = load_weight_file(weights_dir, path, out_channels);

    snprintf(path, sizeof(path), "postnet/layer_%d/bn_var.bin", layer_idx);
    layer->bn_var = load_weight_file(weights_dir, path, out_channels);
}

void load_weights(const char* weights_dir, Weights* w, Config* c) {
    printf("Loading weights from %s...\n", weights_dir);

    // Load encoder embedding
    printf("  Loading encoder...\n");
    w->encoder_embedding = load_weight_file(weights_dir, "encoder/embedding.bin",
                                           c->vocab_size * c->dim);
    if (!w->encoder_embedding) {
        fprintf(stderr, "Failed to load encoder embedding\n");
        return;
    }

    // Load encoder layers
    for (int i = 0; i < c->n_enc_layers; i++) {
        printf("    Layer %d...\n", i);
        load_fft_layer(weights_dir, "encoder", i, &w->encoder_layers[i], c);
    }

    // Load variance adaptor
    printf("  Loading variance adaptor...\n");
    load_variance_predictor(weights_dir, "duration_predictor", &w->duration_predictor, c);
    load_variance_predictor(weights_dir, "pitch_predictor", &w->pitch_predictor, c);
    load_variance_predictor(weights_dir, "energy_predictor", &w->energy_predictor, c);

    w->pitch_embedding = load_weight_file(weights_dir, "variance_adaptor/pitch_embedding.bin",
                                         c->n_bins * c->dim);
    w->energy_embedding = load_weight_file(weights_dir, "variance_adaptor/energy_embedding.bin",
                                          c->n_bins * c->dim);

    // Load decoder layers
    printf("  Loading decoder...\n");
    for (int i = 0; i < c->n_dec_layers; i++) {
        printf("    Layer %d...\n", i);
        load_fft_layer(weights_dir, "decoder", i, &w->decoder_layers[i], c);
    }

    // Load mel linear projection
    printf("  Loading mel linear...\n");
    w->mel_linear_weight = load_weight_file(weights_dir, "mel_linear_weight.bin",
                                           c->dim * c->n_mels);
    w->mel_linear_bias = load_weight_file(weights_dir, "mel_linear_bias.bin", c->n_mels);

    // Load PostNet layers
    printf("  Loading postnet...\n");
    int postnet_dim = c->postnet_embedding_dim;
    load_postnet_layer(weights_dir, 0, &w->postnet_layers[0], c, c->n_mels, postnet_dim);
    for (int i = 1; i < c->postnet_n_convolutions - 1; i++) {
        load_postnet_layer(weights_dir, i, &w->postnet_layers[i], c, postnet_dim, postnet_dim);
    }
    load_postnet_layer(weights_dir, c->postnet_n_convolutions - 1,
                      &w->postnet_layers[c->postnet_n_convolutions - 1], c, postnet_dim, c->n_mels);

    printf("✓ Weights loaded successfully!\n");
}

// ============================================================================
// Utility Functions
// ============================================================================

int quantize_value(float value, float min_val, float max_val, int n_bins) {
    // Quantize a continuous value to bin index
    float normalized = (value - min_val) / (max_val - min_val);
    normalized = fmaxf(0.0f, fminf(1.0f, normalized));  // Clip to [0, 1]
    int bin = (int)(normalized * (n_bins - 1));
    return bin;
}

// ============================================================================
// Encoder Implementation
// ============================================================================

void encoder_layer_forward(RunState* s, Config* c, FFTLayer* layer, float* x, int seq_len) {
    // Multi-Head Self-Attention
    multi_head_attention(s, c, layer, x, x, x, seq_len, seq_len);

    // Residual connection + Layer Norm
    for (int i = 0; i < seq_len * c->dim; i++) {
        s->attn_out[i] += x[i];
    }
    layer_norm(s->attn_out, s->attn_out, layer->attn_norm_gamma, layer->attn_norm_beta, seq_len, c->dim);

    // Feed-Forward Network
    feed_forward(s, c, layer, s->attn_out, seq_len);

    // Residual connection + Layer Norm
    for (int i = 0; i < seq_len * c->dim; i++) {
        s->ffn_out[i] += s->attn_out[i];
    }
    layer_norm(s->attn_out, s->ffn_out, layer->ffn_norm_gamma, layer->ffn_norm_beta, seq_len, c->dim);
}

void encoder_forward(RunState* s, Config* c, Weights* w, int* phoneme_ids, int n_phonemes) {
    // 1. Embedding lookup
    for (int i = 0; i < n_phonemes; i++) {
        int ph_id = phoneme_ids[i];
        if (ph_id < 0 || ph_id >= c->vocab_size) {
            fprintf(stderr, "Warning: Invalid phoneme ID %d at position %d\n", ph_id, i);
            ph_id = 0;  // Use padding token
        }
        memcpy(s->encoder_emb + i * c->dim,
               w->encoder_embedding + ph_id * c->dim,
               c->dim * sizeof(float));
    }

    // 2. Add positional encoding
    for (int i = 0; i < n_phonemes; i++) {
        for (int j = 0; j < c->dim; j++) {
            s->encoder_emb[i * c->dim + j] += w->encoder_pe[i * c->dim + j];
        }
    }

    if (dump_enabled()) {
        dump_float_matrix("encoder_embedding.bin", s->encoder_emb, n_phonemes, c->dim);
    }

    // 3. Pass through encoder layers
    float* x = s->encoder_emb;
    for (int layer = 0; layer < c->n_enc_layers; layer++) {
        encoder_layer_forward(s, c, &w->encoder_layers[layer], x, n_phonemes);
        if (dump_enabled()) {
            char filename[64];
            snprintf(filename, sizeof(filename), "encoder_layer_%d.bin", layer);
            printf("Dumping %s\n", filename);
            dump_float_matrix(filename, s->attn_out, n_phonemes, c->dim);
        }
        x = s->attn_out;  // Output becomes input to next layer
    }

    // 4. Copy final output to encoder_out
    memcpy(s->encoder_out, s->attn_out, n_phonemes * c->dim * sizeof(float));

    if (dump_enabled()) {
        dump_float_matrix("encoder_output.bin", s->encoder_out, n_phonemes, c->dim);
    }
}

// ============================================================================
// Variance Adaptor Implementation
// ============================================================================

void variance_predictor_forward(RunState* s, Config* c, VariancePredictor* vp,
                                float* x, int seq_len, float* output) {
    // Variance predictor: 2 Conv1D layers + LayerNorm + Linear projection
    int filter_size = c->var_pred_filter_size;
    int kernel_size = c->var_pred_kernel_size;
    int padding = kernel_size / 2;

    // First Conv1D layer
    // Input: [1, dim, seq_len] -> [1, filter_size, seq_len]
    // Note: PyTorch conv1d expects [batch, channels, length]
    // We need to transpose x from [seq_len, dim] to [1, dim, seq_len]

    // For simplicity, we'll work with [batch=1, channels, seq_len] format
    conv1d(s->conv_buf1, x, vp->conv1_weight, vp->conv1_bias,
           1, c->dim, filter_size, seq_len, kernel_size, padding);

    // ReLU activation
    relu(s->conv_buf1, seq_len * filter_size);

    // Layer Norm 1
    layer_norm(s->conv_buf1, s->conv_buf1, vp->ln1_gamma, vp->ln1_beta, seq_len, filter_size);

    // Second Conv1D layer
    conv1d(s->conv_buf2, s->conv_buf1, vp->conv2_weight, vp->conv2_bias,
           1, filter_size, filter_size, seq_len, kernel_size, padding);

    // ReLU activation
    relu(s->conv_buf2, seq_len * filter_size);

    // Layer Norm 2
    layer_norm(s->conv_buf2, s->conv_buf2, vp->ln2_gamma, vp->ln2_beta, seq_len, filter_size);

    // Linear projection: [seq_len, filter_size] -> [seq_len, 1]
    for (int i = 0; i < seq_len; i++) {
        float sum = 0.0f;
        for (int j = 0; j < filter_size; j++) {
            sum += s->conv_buf2[i * filter_size + j] * vp->linear_weight[j];
        }
        output[i] = sum + vp->linear_bias[0];
    }
}

void variance_adaptor_forward(RunState* s, Config* c, Weights* w, int n_phonemes) {
    // 1. Predict duration, pitch, energy
    variance_predictor_forward(s, c, &w->duration_predictor, s->encoder_out, n_phonemes, s->duration_pred);
    variance_predictor_forward(s, c, &w->pitch_predictor, s->encoder_out, n_phonemes, s->pitch_pred);
    variance_predictor_forward(s, c, &w->energy_predictor, s->encoder_out, n_phonemes, s->energy_pred);

    std::vector<float> log_duration_values;
    if (dump_enabled()) {
        dump_float_vector("pitch_prediction.bin", s->pitch_pred, n_phonemes);
        dump_float_vector("energy_prediction.bin", s->energy_pred, n_phonemes);
        log_duration_values.assign(s->duration_pred, s->duration_pred + n_phonemes);
    }

    // 2. Quantize and embed pitch/energy (phoneme-level)
    for (int i = 0; i < n_phonemes; i++) {
        // Quantize predictions to bin indices
        int pitch_bin = quantize_value(s->pitch_pred[i], c->pitch_min, c->pitch_max, c->n_bins);
        int energy_bin = quantize_value(s->energy_pred[i], c->energy_min, c->energy_max, c->n_bins);

        // Add pitch and energy embeddings to encoder output
        for (int j = 0; j < c->dim; j++) {
            s->encoder_out[i * c->dim + j] += w->pitch_embedding[pitch_bin * c->dim + j];
            s->encoder_out[i * c->dim + j] += w->energy_embedding[energy_bin * c->dim + j];
        }
    }

    // 3. Length Regulation - expand phoneme sequence to mel frames
    s->mel_len = 0;
    for (int i = 0; i < n_phonemes; i++) {
        // Duration predictor outputs log(duration + 1)
        float duration = expf(s->duration_pred[i]) - 1.0f;
        if (duration < 0.0f) duration = 0.0f;

        s->duration_pred[i] = duration;

        // Round to nearest integer number of frames
        int rounded = (int)lrintf(duration);
        if (rounded < 0) {
            rounded = 0;
        }
        s->durations[i] = rounded;

        // Repeat phoneme representation duration[i] times
        for (int d = 0; d < s->durations[i]; d++) {
            memcpy(s->variance_out + s->mel_len * c->dim,
                   s->encoder_out + i * c->dim,
                   c->dim * sizeof(float));
            s->mel_len++;
        }
    }

    if (s->mel_len == 0) {
        // Ensure at least one frame to avoid downstream issues
        memcpy(s->variance_out, s->encoder_out, c->dim * sizeof(float));
        s->mel_len = 1;
    }

    const char* debug_env = getenv("FS2_DEBUG_DURATIONS");
    if (debug_env) {
        printf("  Durations (rounded): [");
        for (int i = 0; i < n_phonemes; i++) {
            printf("%d", s->durations[i]);
            if (i != n_phonemes - 1) printf(", ");
        }
        printf("]\n");
        printf("  Durations (float): [");
        for (int i = 0; i < n_phonemes; i++) {
            printf("%.4f", s->duration_pred[i]);
            if (i != n_phonemes - 1) printf(", ");
        }
        printf("]\n");
    }

    if (dump_enabled()) {
        if (!log_duration_values.empty()) {
            dump_float_vector("log_duration_prediction.bin", log_duration_values.data(), n_phonemes);
        }
        dump_float_vector("duration_prediction.bin", s->duration_pred, n_phonemes);
        dump_int_vector("duration_rounded.bin", s->durations, n_phonemes);
        dump_float_matrix("variance_output.bin", s->variance_out, s->mel_len, c->dim);
    }
}

// ============================================================================
// Decoder Implementation
// ============================================================================

void decoder_layer_forward(RunState* s, Config* c, FFTLayer* layer, float* x, int seq_len) {
    // Same structure as encoder layer
    encoder_layer_forward(s, c, layer, x, seq_len);
}

void decoder_forward(RunState* s, Config* c, Weights* w, int mel_len) {
    // 1. Add positional encoding to variance adaptor output
    for (int i = 0; i < mel_len; i++) {
        for (int j = 0; j < c->dim; j++) {
            s->variance_out[i * c->dim + j] += w->decoder_pe[i * c->dim + j];
        }
    }

    // 2. Pass through decoder layers
    float* x = s->variance_out;
    for (int layer = 0; layer < c->n_dec_layers; layer++) {
        decoder_layer_forward(s, c, &w->decoder_layers[layer], x, mel_len);
        x = s->attn_out;  // Output becomes input to next layer
    }

    // 3. Copy final output to decoder_out
    memcpy(s->decoder_out, s->attn_out, mel_len * c->dim * sizeof(float));
    if (dump_enabled()) {
        dump_float_matrix("decoder_output.bin", s->decoder_out, mel_len, c->dim);
    }

    // 4. Project to mel-spectrogram: [mel_len, dim] -> [mel_len, n_mels]
    matmul_transposed(s->mel_out, s->decoder_out, w->mel_linear_weight, mel_len, c->dim, c->n_mels);

    // Add bias
    for (int i = 0; i < mel_len; i++) {
        for (int j = 0; j < c->n_mels; j++) {
            s->mel_out[i * c->n_mels + j] += w->mel_linear_bias[j];
        }
    }

    if (dump_enabled()) {
        dump_float_matrix("mel_before.bin", s->mel_out, mel_len, c->n_mels);
    }
}

// ============================================================================
// PostNet Implementation
// ============================================================================

void postnet_forward(RunState* s, Config* c, Weights* w, int mel_len) {
    // PostNet: 5 Conv1D layers with BatchNorm
    int n_convs = c->postnet_n_convolutions;
    int embedding_dim = c->postnet_embedding_dim;
    int kernel_size = c->postnet_kernel_size;
    int padding = kernel_size / 2;

    // First layer: [mel_len, n_mels] -> [mel_len, postnet_embedding_dim]
    conv1d(s->conv_buf1, s->mel_out, w->postnet_layers[0].conv_weight, w->postnet_layers[0].conv_bias,
           1, c->n_mels, embedding_dim, mel_len, kernel_size, padding);

    batch_norm(s->conv_buf1, s->conv_buf1,
               w->postnet_layers[0].bn_gamma, w->postnet_layers[0].bn_beta,
               w->postnet_layers[0].bn_mean, w->postnet_layers[0].bn_var,
               mel_len, embedding_dim);

    tanh_activation(s->conv_buf1, mel_len * embedding_dim);

    // Middle layers: [mel_len, postnet_embedding_dim] -> [mel_len, postnet_embedding_dim]
    for (int i = 1; i < n_convs - 1; i++) {
        float* in_buf = (i % 2 == 1) ? s->conv_buf1 : s->conv_buf2;
        float* out_buf = (i % 2 == 1) ? s->conv_buf2 : s->conv_buf1;

        conv1d(out_buf, in_buf, w->postnet_layers[i].conv_weight, w->postnet_layers[i].conv_bias,
               1, embedding_dim, embedding_dim, mel_len, kernel_size, padding);

        batch_norm(out_buf, out_buf,
                   w->postnet_layers[i].bn_gamma, w->postnet_layers[i].bn_beta,
                   w->postnet_layers[i].bn_mean, w->postnet_layers[i].bn_var,
                   mel_len, embedding_dim);

        tanh_activation(out_buf, mel_len * embedding_dim);
    }

    // Last layer: [mel_len, postnet_embedding_dim] -> [mel_len, n_mels]
    float* last_in = ((n_convs - 2) % 2 == 1) ? s->conv_buf2 : s->conv_buf1;
    float* temp_out = s->conv_buf2;  // Use conv_buf2 as temporary

    conv1d(temp_out, last_in, w->postnet_layers[n_convs - 1].conv_weight,
           w->postnet_layers[n_convs - 1].conv_bias,
           1, embedding_dim, c->n_mels, mel_len, kernel_size, padding);

    batch_norm(temp_out, temp_out,
               w->postnet_layers[n_convs - 1].bn_gamma, w->postnet_layers[n_convs - 1].bn_beta,
               w->postnet_layers[n_convs - 1].bn_mean, w->postnet_layers[n_convs - 1].bn_var,
               mel_len, c->n_mels);

    // No tanh for the last layer

    // Add residual connection: postnet_out = mel_out + postnet_output
    for (int i = 0; i < mel_len * c->n_mels; i++) {
        s->postnet_mel_out[i] = s->mel_out[i] + temp_out[i];
    }

    if (dump_enabled()) {
        dump_float_matrix("mel_after.bin", s->postnet_mel_out, mel_len, c->n_mels);
    }
}

// ============================================================================
// Main Inference Function
// ============================================================================

void fastspeech2_forward(RunState* s, Config* c, Weights* w,
                        int* phoneme_ids, int n_phonemes) {
    printf("Running FastSpeech2 inference...\n");
    printf("  Input: %d phonemes\n", n_phonemes);

    // 1. Encoder
    encoder_forward(s, c, w, phoneme_ids, n_phonemes);
    printf("  ✓ Encoder complete\n");

    // 2. Variance Adaptor (includes Length Regulation)
    variance_adaptor_forward(s, c, w, n_phonemes);
    printf("  ✓ Variance Adaptor complete (mel_len=%d)\n", s->mel_len);

    // 3. Decoder
    decoder_forward(s, c, w, s->mel_len);
    printf("  ✓ Decoder complete\n");

    // 4. PostNet
    postnet_forward(s, c, w, s->mel_len);
    printf("  ✓ PostNet complete\n");

    printf("FastSpeech2 inference complete! Output shape: [%d, %d]\n",
           s->mel_len, c->n_mels);
}
