#include "op.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <string>

// ============================================================================
// Basic Linear Algebra Operations
// ============================================================================

void matmul(float* C, float* A, float* B, int M, int K, int N) {
    // C[M, N] = A[M, K] @ B[K, N]
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void matmul_transposed(float* C, float* A, float* B, int M, int K, int N) {
    // C[M, N] = A[M, K] @ B^T, where B is stored as [N, K]
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}

void vec_add(float* out, float* a, float* b, int n) {
    // out = a + b (element-wise)
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

void vec_scale(float* out, float* x, float scale, int n) {
    // out = x * scale
    for (int i = 0; i < n; i++) {
        out[i] = x[i] * scale;
    }
}

// ============================================================================
// Normalization Operations
// ============================================================================

void layer_norm(float* out, float* x, float* gamma, float* beta, int seq_len, int dim) {
    // Layer normalization: out = gamma * (x - mean) / sqrt(var + eps) + beta
    const float eps = 1e-5f;

    for (int i = 0; i < seq_len; i++) {
        float* x_row = x + i * dim;
        float* out_row = out + i * dim;

        // Calculate mean
        float mean = 0.0f;
        for (int j = 0; j < dim; j++) {
            mean += x_row[j];
        }
        mean /= dim;

        // Calculate variance
        float var = 0.0f;
        for (int j = 0; j < dim; j++) {
            float diff = x_row[j] - mean;
            var += diff * diff;
        }
        var /= dim;

        // Normalize and apply affine transformation
        float std = sqrtf(var + eps);
        for (int j = 0; j < dim; j++) {
            out_row[j] = gamma[j] * (x_row[j] - mean) / std + beta[j];
        }
    }
}

void batch_norm(float* out, float* x, float* gamma, float* beta,
                float* running_mean, float* running_var, int frames, int channels) {
    // Batch normalization (inference mode): out = gamma * (x - mean) / sqrt(var + eps) + beta
    const float eps = 1e-5f;

    for (int i = 0; i < frames; i++) {
        for (int c = 0; c < channels; c++) {
            int idx = i * channels + c;
            float normalized = (x[idx] - running_mean[c]) / sqrtf(running_var[c] + eps);
            out[idx] = gamma[c] * normalized + beta[c];
        }
    }
}

// ============================================================================
// Activation Functions
// ============================================================================

void gelu(float* x, int n) {
    // GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/pi)
    const float coef = 0.044715f;

    for (int i = 0; i < n; i++) {
        float x_val = x[i];
        float x_cubed = x_val * x_val * x_val;
        float tanh_arg = sqrt_2_over_pi * (x_val + coef * x_cubed);
        x[i] = 0.5f * x_val * (1.0f + tanhf(tanh_arg));
    }
}

void relu(float* x, int n) {
    // ReLU activation: max(0, x)
    for (int i = 0; i < n; i++) {
        x[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

void tanh_activation(float* x, int n) {
    // Tanh activation
    for (int i = 0; i < n; i++) {
        x[i] = tanhf(x[i]);
    }
}

void softmax(float* x, int batch_size, int seq_len) {
    // Softmax along the last dimension
    for (int b = 0; b < batch_size; b++) {
        float* x_batch = x + b * seq_len;

        // Find max for numerical stability
        float max_val = x_batch[0];
        for (int i = 1; i < seq_len; i++) {
            if (x_batch[i] > max_val) {
                max_val = x_batch[i];
            }
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            x_batch[i] = expf(x_batch[i] - max_val);
            sum += x_batch[i];
        }

        // Normalize
        for (int i = 0; i < seq_len; i++) {
            x_batch[i] /= sum;
        }
    }
}

// ============================================================================
// Attention Mechanism
// ============================================================================

void multi_head_attention(RunState* s, Config* c, FFTLayer* layer,
                         float* q_input, float* k_input, float* v_input,
                         int q_len, int kv_len) {
    // Multi-head self-attention
    int dim = c->dim;
    int n_heads = c->n_heads;
    int head_dim = c->head_dim;

    // Linear projections: Q, K, V
    matmul_transposed(s->attn_q, q_input, layer->attn_q_weight, q_len, dim, dim);
    matmul_transposed(s->attn_k, k_input, layer->attn_k_weight, kv_len, dim, dim);
    matmul_transposed(s->attn_v, v_input, layer->attn_v_weight, kv_len, dim, dim);

    // Add bias
    for (int i = 0; i < q_len; i++) {
        for (int j = 0; j < dim; j++) {
            s->attn_q[i * dim + j] += layer->attn_q_bias[j];
        }
    }
    for (int i = 0; i < kv_len; i++) {
        for (int j = 0; j < dim; j++) {
            s->attn_k[i * dim + j] += layer->attn_k_bias[j];
            s->attn_v[i * dim + j] += layer->attn_v_bias[j];
        }
    }

    // Reshape to [batch=1, n_heads, seq_len, head_dim] and compute attention
    // For simplicity, we process each head separately
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < n_heads; h++) {
        // Extract Q, K, V for this head
        // Q: [q_len, head_dim], K: [kv_len, head_dim], V: [kv_len, head_dim]

        // Compute attention scores: Q @ K^T / sqrt(head_dim)
        // scores: [q_len, kv_len]
        float* scores = s->attn_scores + h * q_len * kv_len;

        for (int i = 0; i < q_len; i++) {
            for (int j = 0; j < kv_len; j++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    float q_val = s->attn_q[i * dim + h * head_dim + d];
                    float k_val = s->attn_k[j * dim + h * head_dim + d];
                    score += q_val * k_val;
                }
                scores[i * kv_len + j] = score * scale;
            }
        }

        // Apply softmax to scores
        softmax(scores, q_len, kv_len);

        // Compute weighted sum: scores @ V
        // output: [q_len, head_dim]
        for (int i = 0; i < q_len; i++) {
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int j = 0; j < kv_len; j++) {
                    float v_val = s->attn_v[j * dim + h * head_dim + d];
                    sum += scores[i * kv_len + j] * v_val;
                }
                // Store in output buffer at correct position
                s->attn_out[i * dim + h * head_dim + d] = sum;
            }
        }
    }

    // Output projection
    float* temp = s->ffn_out;  // Reuse buffer
    memcpy(temp, s->attn_out, q_len * dim * sizeof(float));
    matmul_transposed(s->attn_out, temp, layer->attn_out_weight, q_len, dim, dim);

    // Add bias
    for (int i = 0; i < q_len; i++) {
        for (int j = 0; j < dim; j++) {
            s->attn_out[i * dim + j] += layer->attn_out_bias[j];
        }
    }
}

// ============================================================================
// Feed-Forward Network
// ============================================================================

void feed_forward(RunState* s, Config* c, FFTLayer* layer, float* x, int seq_len, int layer_idx) {
    // Two-layer position-wise feed-forward network implemented as conv1d ops.
    int dim = c->dim;
    int ffn_hidden = c->ffn_hidden;
    int k1 = layer->ffn_kernel_size1 > 0 ? layer->ffn_kernel_size1 : 1;
    int k2 = layer->ffn_kernel_size2 > 0 ? layer->ffn_kernel_size2 : 1;
    int pad1 = k1 / 2;
    int pad2 = k2 / 2;

    conv1d(s->ffn_hidden, x, layer->ffn_w1, layer->ffn_b1,
           1, dim, ffn_hidden, seq_len, k1, pad1);
    bool should_dump = is_dump_enabled() && layer_idx >= 0;

    if (should_dump) {
        std::string prefix = "encoder_layer_" + std::to_string(layer_idx) + "_";
        dump_float_matrix((prefix + "ffn_pre_relu.bin").c_str(), s->ffn_hidden, seq_len, ffn_hidden);
    }
    relu(s->ffn_hidden, seq_len * ffn_hidden);
    if (should_dump) {
        std::string prefix = "encoder_layer_" + std::to_string(layer_idx) + "_";
        dump_float_matrix((prefix + "ffn_post_relu.bin").c_str(), s->ffn_hidden, seq_len, ffn_hidden);
    }
    conv1d(s->ffn_out, s->ffn_hidden, layer->ffn_w2, layer->ffn_b2,
           1, ffn_hidden, dim, seq_len, k2, pad2);
    if (should_dump) {
        std::string prefix = "encoder_layer_" + std::to_string(layer_idx) + "_";
        dump_float_matrix((prefix + "ffn_out.bin").c_str(), s->ffn_out, seq_len, dim);
    }
}

// ============================================================================
// Convolution Operations
// ============================================================================

void conv1d(float* out, float* in, float* weight, float* bias,
           int batch, int in_channels, int out_channels, int seq_len,
           int kernel_size, int padding) {
    // 1D convolution using [batch, seq_len, channels] memory layout.
    // weight: [out_channels, in_channels, kernel_size]
    // out: [batch, seq_len, out_channels]

    for (int b = 0; b < batch; b++) {
        float* in_batch = in + b * seq_len * in_channels;
        float* out_batch = out + b * seq_len * out_channels;

        for (int t = 0; t < seq_len; t++) {
            for (int oc = 0; oc < out_channels; oc++) {
                float sum = bias ? bias[oc] : 0.0f;

                for (int ic = 0; ic < in_channels; ic++) {
                    for (int k = 0; k < kernel_size; k++) {
                        int in_pos = t + k - padding;
                        if (in_pos < 0 || in_pos >= seq_len) {
                            continue;
                        }
                        int in_idx = in_pos * in_channels + ic;
                        int weight_idx = oc * in_channels * kernel_size + ic * kernel_size + k;
                        sum += in_batch[in_idx] * weight[weight_idx];
                    }
                }

                out_batch[t * out_channels + oc] = sum;
            }
        }
    }
}

// ============================================================================
// Positional Encoding
// ============================================================================

void sinusoidal_position_encoding(float* pe, int max_len, int dim) {
    // Generate sinusoidal positional encoding
    // pe: [max_len, dim]

    for (int pos = 0; pos < max_len; pos++) {
        for (int i = 0; i < dim / 2; i++) {
            float angle = pos / powf(10000.0f, 2.0f * i / (float)dim);
            pe[pos * dim + 2 * i] = sinf(angle);
            pe[pos * dim + 2 * i + 1] = cosf(angle);
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

void dump_tensor(const char* name, float* tensor, int rows, int cols) {
    // Dump tensor to binary file for debugging
    char filename[256];
    snprintf(filename, sizeof(filename), "debug/%s.bin", name);

    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Warning: Could not open %s for writing\n", filename);
        return;
    }

    fwrite(tensor, sizeof(float), rows * cols, f);
    fclose(f);

    printf("Dumped tensor: %s [%d, %d]\n", name, rows, cols);
}

float* load_tensor_file(const char* path, int* size) {
    // Load tensor from binary file
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open %s\n", path);
        return nullptr;
    }

    // Get file size
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    *size = file_size / sizeof(float);

    float* tensor = new float[*size];
    size_t read = fread(tensor, sizeof(float), *size, f);
    fclose(f);

    if (read != (size_t)*size) {
        fprintf(stderr, "Error: Read %zu elements but expected %d\n", read, *size);
        delete[] tensor;
        return nullptr;
    }

    return tensor;
}
