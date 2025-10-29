#ifndef OP_H
#define OP_H

#include "fastspeech2.h"

// Basic linear algebra operations
void matmul(float* C, float* A, float* B, int M, int K, int N);
void matmul_transposed(float* C, float* A, float* B, int M, int K, int N);
void vec_add(float* out, float* a, float* b, int n);
void vec_scale(float* out, float* x, float scale, int n);

// Normalization operations
void layer_norm(float* out, float* x, float* gamma, float* beta, int seq_len, int dim);
void batch_norm(float* out, float* x, float* gamma, float* beta,
                float* running_mean, float* running_var, int frames, int channels);

// Activation functions
void gelu(float* x, int n);
void relu(float* x, int n);
void tanh_activation(float* x, int n);
void softmax(float* x, int batch_size, int seq_len);

// Attention mechanism
void multi_head_attention(RunState* s, Config* c, FFTLayer* layer,
                         float* q_input, float* k_input, float* v_input,
                         int q_len, int kv_len);

// Feed-forward network
void feed_forward(RunState* s, Config* c, FFTLayer* layer, float* x, int seq_len, int layer_idx);

// Convolution operations
void conv1d(float* out, float* in, float* weight, float* bias,
           int batch, int in_channels, int out_channels, int seq_len,
           int kernel_size, int padding);

// Positional encoding
void sinusoidal_position_encoding(float* pe, int max_len, int dim);

// Utility functions
void dump_tensor(const char* name, float* tensor, int rows, int cols);
float* load_tensor_file(const char* path, int* size);

#endif // OP_H
