#ifndef FASTSPEECH2_H
#define FASTSPEECH2_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// Configuration structure for FastSpeech2 model
typedef struct {
    // Model dimensions
    int dim;                    // 256 - hidden dimension
    int n_enc_layers;           // 4 - encoder FFT blocks
    int n_dec_layers;           // 6 - decoder FFT blocks
    int n_heads;                // 2 - attention heads
    int head_dim;               // 128 - dimension per head (dim / n_heads)
    int ffn_hidden;             // 1024 - FFN intermediate dimension

    // Vocabulary & I/O
    int vocab_size;             // 76 - phoneme vocabulary size
    int n_mels;                 // 80 - mel-spectrogram bins
    int max_seq_len;            // 1000 - maximum phoneme sequence length

    // Variance predictor
    int var_pred_filter_size;   // 256
    int var_pred_kernel_size;   // 3
    float var_pred_dropout;     // 0.5 (not used in inference)
    int n_bins;                 // 256 - pitch/energy quantization bins

    // PostNet
    int postnet_embedding_dim;  // 512
    int postnet_kernel_size;    // 5
    int postnet_n_convolutions; // 5

    // Pitch/Energy stats for normalization
    float pitch_min;
    float pitch_max;
    float energy_min;
    float energy_max;
} Config;

// Variance Predictor structure (for Duration, Pitch, Energy)
typedef struct {
    // Conv layers
    float* conv1_weight;        // [filter_size, dim, kernel_size]
    float* conv1_bias;          // [filter_size]
    float* conv2_weight;        // [filter_size, filter_size, kernel_size]
    float* conv2_bias;          // [filter_size]

    // Layer norms
    float* ln1_gamma;           // [filter_size]
    float* ln1_beta;            // [filter_size]
    float* ln2_gamma;           // [filter_size]
    float* ln2_beta;            // [filter_size]

    // Linear projection
    float* linear_weight;       // [filter_size, 1]
    float* linear_bias;         // [1]
} VariancePredictor;

// Encoder/Decoder layer structure (FFT block)
typedef struct {
    // Multi-Head Attention
    float* attn_q_weight;       // [dim, dim]
    float* attn_q_bias;         // [dim]
    float* attn_k_weight;       // [dim, dim]
    float* attn_k_bias;         // [dim]
    float* attn_v_weight;       // [dim, dim]
    float* attn_v_bias;         // [dim]
    float* attn_out_weight;     // [dim, dim]
    float* attn_out_bias;       // [dim]
    float* attn_norm_gamma;     // [dim]
    float* attn_norm_beta;      // [dim]

    // Position-wise Feed-Forward
    float* ffn_w1;              // [ffn_hidden, dim, kernel]
    float* ffn_b1;              // [ffn_hidden]
    float* ffn_w2;              // [dim, ffn_hidden, kernel]
    float* ffn_b2;              // [dim]
    float* ffn_norm_gamma;      // [dim]
    float* ffn_norm_beta;       // [dim]
    int ffn_kernel_size1;       // kernel size for first FFN conv
    int ffn_kernel_size2;       // kernel size for second FFN conv
} FFTLayer;

// PostNet layer structure
typedef struct {
    float* conv_weight;         // [out_ch, in_ch, kernel_size]
    float* conv_bias;           // [out_ch]
    float* bn_gamma;            // [out_ch]
    float* bn_beta;             // [out_ch]
    float* bn_mean;             // [out_ch] - running mean
    float* bn_var;              // [out_ch] - running variance
} PostNetLayer;

// Model weights structure
typedef struct {
    // Encoder
    float* encoder_embedding;   // [vocab_size, dim]
    float* encoder_pe;          // [max_seq_len, dim] - positional encoding
    FFTLayer* encoder_layers;   // Array of n_enc_layers

    // Variance Adaptor
    VariancePredictor duration_predictor;
    VariancePredictor pitch_predictor;
    VariancePredictor energy_predictor;
    float* pitch_embedding;     // [n_bins, dim]
    float* energy_embedding;    // [n_bins, dim]

    // Decoder
    float* decoder_pe;          // [max_seq_len, dim] - positional encoding
    FFTLayer* decoder_layers;   // Array of n_dec_layers

    // Mel Linear projection
    float* mel_linear_weight;   // [dim, n_mels]
    float* mel_linear_bias;     // [n_mels]

    // PostNet
    PostNetLayer* postnet_layers; // Array of postnet_n_convolutions
} Weights;

// Runtime state buffers
typedef struct {
    // Encoder buffers
    float* encoder_emb;         // [max_seq_len, dim]
    float* encoder_out;         // [max_seq_len, dim]

    // Variance Adaptor buffers
    float* duration_pred;       // [max_seq_len]
    float* pitch_pred;          // [max_seq_len]
    float* energy_pred;         // [max_seq_len]
    int* durations;             // [max_seq_len] - rounded durations
    float* variance_out;        // [max_seq_len * max_duration, dim]
    int mel_len;                // Actual mel frame count after length regulation

    // Decoder buffers
    float* decoder_out;         // [max_seq_len * max_duration, dim]

    // Output buffers
    float* mel_out;             // [max_seq_len * max_duration, n_mels]
    float* postnet_mel_out;     // [max_seq_len * max_duration, n_mels]

    // Temporary attention/FFN buffers (reused across layers)
    float* attn_q;              // [max_seq_len * max_duration, dim]
    float* attn_k;              // [max_seq_len * max_duration, dim]
    float* attn_v;              // [max_seq_len * max_duration, dim]
    float* attn_scores;         // [n_heads, max_seq_len * max_duration, max_seq_len * max_duration]
    float* attn_out;            // [max_seq_len * max_duration, dim]
    float* ffn_hidden;          // [max_seq_len * max_duration, ffn_hidden]
    float* ffn_out;             // [max_seq_len * max_duration, dim]

    // Conv1D temporary buffers
    float* conv_buf1;           // [max_seq_len * max_duration, filter_size]
    float* conv_buf2;           // [max_seq_len * max_duration, filter_size]
} RunState;

// Function declarations

// Initialization and cleanup
void init_config(Config* c);
RunState* create_run_state(Config* c);
void free_run_state(RunState* s);
Weights* create_weights(Config* c);
void free_weights(Weights* w, Config* c);

// Weight loading
void load_config(const char* config_path, Config* c);
void load_weights(const char* weights_path, Weights* w, Config* c);

// Model components
void encoder_forward(RunState* s, Config* c, Weights* w, int* phoneme_ids, int n_phonemes);
void encoder_layer_forward(RunState* s, Config* c, FFTLayer* layer, float* x, int seq_len);

void variance_predictor_forward(RunState* s, Config* c, VariancePredictor* vp,
                                float* x, int seq_len, float* output);
void variance_adaptor_forward(RunState* s, Config* c, Weights* w, int n_phonemes);

void decoder_forward(RunState* s, Config* c, Weights* w, int mel_len);
void decoder_layer_forward(RunState* s, Config* c, FFTLayer* layer, float* x, int seq_len);

void postnet_forward(RunState* s, Config* c, Weights* w, int mel_len);

// Main inference function
void fastspeech2_forward(RunState* s, Config* c, Weights* w,
                        int* phoneme_ids, int n_phonemes);

// Utilities
int quantize_value(float value, float min_val, float max_val, int n_bins);

#endif // FASTSPEECH2_H
