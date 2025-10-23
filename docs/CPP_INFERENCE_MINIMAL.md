# FastSpeech2.c - 極簡 C++ 推理引擎規劃

**風格參考**: llama2.c
**理念**: 輕量、簡潔、零依賴、易讀

---

## 設計理念

### llama2.c 風格特點

```
✅ 單文件實現（或最多 2-3 個文件）
✅ 零外部依賴，只用標準庫
✅ 手寫所有算子
✅ 使用原始數組和指針
✅ mmap 映射權重文件
✅ 簡單的數據結構
✅ 直接、易讀的代碼
```

### FastSpeech2.c 設計目標

```cpp
// 使用方式應該極其簡單
./fastspeech2 model.bin "Hello world" output.mel

// 或者
./fastspeech2 model.bin < input.txt > output.mel
```

---

## 文件結構（極簡版）

```
fastspeech2.c/
├── fastspeech2.cpp        # 主程序 (~1000-1500 行)
├── fastspeech2.h          # 數據結構定義 (~200 行)
├── convert_weights.py     # PyTorch → 二進制轉換
├── CMakeLists.txt         # 可選，也可以直接 g++ 編譯
└── README.md
```

**編譯**:
```bash
# 最簡單的編譯方式
g++ -O3 -march=native -o fastspeech2 fastspeech2.cpp -lm

# 或使用 CMake
mkdir build && cd build
cmake .. && make
```

---

## 核心數據結構

### 1. 權重存儲（全部用原始數組）

```cpp
// fastspeech2.h

typedef struct {
    int dim;      // hidden dimension (256)
    int n_layers; // encoder/decoder layers (4/6)
    int n_heads;  // attention heads (2)
    int n_mels;   // mel channels (80)
    int vocab_size; // phoneme vocab size (76)
    int max_seq_len; // max sequence length (1000)
} Config;

typedef struct {
    // Encoder
    float* encoder_tok_emb;        // [vocab_size, dim]
    float* encoder_pos_emb;        // [max_seq_len, dim]

    // Encoder layers (4 layers)
    float* encoder_attn_q[4];      // [dim, dim] per layer
    float* encoder_attn_k[4];
    float* encoder_attn_v[4];
    float* encoder_attn_o[4];
    float* encoder_attn_norm_w[4]; // [dim]
    float* encoder_attn_norm_b[4];

    float* encoder_ffn_w1[4];      // [dim, 1024]
    float* encoder_ffn_w2[4];      // [1024, dim]
    float* encoder_ffn_norm_w[4];
    float* encoder_ffn_norm_b[4];

    // Variance Adaptor
    float* duration_conv1_w;       // [256, 256, 3]
    float* duration_conv1_b;
    float* duration_conv2_w;
    float* duration_conv2_b;
    float* duration_linear_w;      // [256, 1]

    float* pitch_conv1_w;
    float* pitch_conv1_b;
    float* pitch_conv2_w;
    float* pitch_conv2_b;
    float* pitch_linear_w;
    float* pitch_emb;              // [256, dim]
    float* pitch_bins;             // [255]

    float* energy_conv1_w;
    float* energy_conv1_b;
    float* energy_conv2_w;
    float* energy_conv2_b;
    float* energy_linear_w;
    float* energy_emb;             // [256, dim]
    float* energy_bins;            // [255]

    // Decoder layers (6 layers)
    float* decoder_pos_emb;        // [max_seq_len, dim]
    float* decoder_attn_q[6];
    float* decoder_attn_k[6];
    float* decoder_attn_v[6];
    float* decoder_attn_o[6];
    float* decoder_attn_norm_w[6];
    float* decoder_attn_norm_b[6];
    float* decoder_ffn_w1[6];
    float* decoder_ffn_w2[6];
    float* decoder_ffn_norm_w[6];
    float* decoder_ffn_norm_b[6];

    // Mel Linear
    float* mel_linear_w;           // [dim, n_mels]
    float* mel_linear_b;           // [n_mels]

    // PostNet
    float* postnet_conv_w[5];      // 5 layers
    float* postnet_conv_b[5];
    float* postnet_bn_w[5];
    float* postnet_bn_b[5];
    float* postnet_bn_mean[5];
    float* postnet_bn_var[5];

} Weights;

typedef struct {
    // 運行時緩衝區（activation buffers）
    float* x;           // [max_seq_len, dim] - 通用緩衝
    float* xb;          // [max_seq_len, dim] - 備用緩衝
    float* q;           // [max_seq_len, dim] - query
    float* k;           // [max_seq_len, dim] - key
    float* v;           // [max_seq_len, dim] - value
    float* att;         // [n_heads, max_seq_len, max_seq_len] - attention scores
    float* logits;      // [max_seq_len] - variance predictions
    float* mel_output;  // [max_mel_len, n_mels] - final mel output
} RunState;
```

**設計理念**:
- 使用原始 `float*` 指針，而非封裝的類
- 所有權重在初始化時一次性分配
- 使用 `mmap` 零拷貝加載權重
- 簡單的命名，易於理解

### 2. 權重文件格式

```
FastSpeech2 Binary Format (.fs2)

Header (256 bytes):
  - Magic: "FS2\0" (4 bytes)
  - Version: uint32_t (4 bytes)
  - Config: Config struct (248 bytes)

Weights (sequential floats):
  - encoder_tok_emb
  - encoder_pos_emb
  - encoder_attn_q[0..3]
  - ... (按順序排列)
  - postnet_bn_var[4]

Total size: ~30-40 MB (float32)
```

---

## 手寫算子實現

### 1. 基礎算子（純 C++ 標準庫）

```cpp
// fastspeech2.cpp

// ----------------------------------------------------------------------------
// 基礎數學函數

inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

inline float tanh_approx(float x) {
    // 快速 tanh 近似（可選）
    if (x < -3.0f) return -1.0f;
    if (x > 3.0f) return 1.0f;
    return tanhf(x);
}

void softmax(float* x, int size) {
    // 找最大值（數值穩定性）
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    // exp 和求和
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // 歸一化
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// ----------------------------------------------------------------------------
// 矩陣運算

void matmul(float* out, float* x, float* w, int n, int d) {
    // out = x @ w^T
    // x: [n, d], w: [d, d] (stored row-major)
    // out: [n, d]

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            float val = 0.0f;
            for (int k = 0; k < d; k++) {
                val += x[i * d + k] * w[j * d + k];
            }
            out[i * d + j] = val;
        }
    }
}

void add_bias(float* out, float* bias, int n, int d) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            out[i * d + j] += bias[j];
        }
    }
}

void layer_norm(float* out, float* x, float* w, float* b, int n, int d) {
    // Layer normalization
    float eps = 1e-5f;

    for (int i = 0; i < n; i++) {
        // 計算均值
        float mean = 0.0f;
        for (int j = 0; j < d; j++) {
            mean += x[i * d + j];
        }
        mean /= d;

        // 計算方差
        float var = 0.0f;
        for (int j = 0; j < d; j++) {
            float diff = x[i * d + j] - mean;
            var += diff * diff;
        }
        var /= d;

        // 歸一化
        float scale = 1.0f / sqrtf(var + eps);
        for (int j = 0; j < d; j++) {
            out[i * d + j] = (x[i * d + j] - mean) * scale * w[j] + b[j];
        }
    }
}

void conv1d(float* out, float* x, float* w, float* b,
            int seq_len, int in_ch, int out_ch, int kernel_size) {
    // 簡化版 1D 卷積（padding = kernel_size // 2）
    int padding = kernel_size / 2;

    for (int t = 0; t < seq_len; t++) {
        for (int oc = 0; oc < out_ch; oc++) {
            float sum = 0.0f;

            for (int k = 0; k < kernel_size; k++) {
                int t_in = t - padding + k;
                if (t_in >= 0 && t_in < seq_len) {
                    for (int ic = 0; ic < in_ch; ic++) {
                        sum += x[t_in * in_ch + ic] *
                               w[oc * in_ch * kernel_size + ic * kernel_size + k];
                    }
                }
            }

            out[t * out_ch + oc] = sum + b[oc];
        }
    }
}

// ----------------------------------------------------------------------------
// Multi-Head Attention

void multi_head_attention(
    float* out,      // output [seq_len, dim]
    float* q_proj,   // Q projection weight [dim, dim]
    float* k_proj,   // K projection weight [dim, dim]
    float* v_proj,   // V projection weight [dim, dim]
    float* o_proj,   // output projection [dim, dim]
    float* x,        // input [seq_len, dim]
    float* q_buf,    // buffer for Q [seq_len, dim]
    float* k_buf,    // buffer for K [seq_len, dim]
    float* v_buf,    // buffer for V [seq_len, dim]
    float* att,      // attention buffer [n_heads, seq_len, seq_len]
    int seq_len,
    int dim,
    int n_heads
) {
    int head_dim = dim / n_heads;

    // 1. Linear projections Q = x @ Wq^T
    matmul(q_buf, x, q_proj, seq_len, dim);
    matmul(k_buf, x, k_proj, seq_len, dim);
    matmul(v_buf, x, v_proj, seq_len, dim);

    // 2. Scaled dot-product attention for each head
    for (int h = 0; h < n_heads; h++) {
        // Compute attention scores
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    float q_val = q_buf[i * dim + h * head_dim + d];
                    float k_val = k_buf[j * dim + h * head_dim + d];
                    score += q_val * k_val;
                }
                score /= sqrtf((float)head_dim);
                att[h * seq_len * seq_len + i * seq_len + j] = score;
            }

            // Softmax over keys
            softmax(&att[h * seq_len * seq_len + i * seq_len], seq_len);
        }

        // Weighted sum of values
        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < head_dim; d++) {
                float val = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    float att_weight = att[h * seq_len * seq_len + i * seq_len + j];
                    val += att_weight * v_buf[j * dim + h * head_dim + d];
                }
                out[i * dim + h * head_dim + d] = val;
            }
        }
    }

    // 3. Output projection
    matmul(q_buf, out, o_proj, seq_len, dim);  // reuse q_buf
    memcpy(out, q_buf, seq_len * dim * sizeof(float));
}

// ----------------------------------------------------------------------------
// Variance Predictor

void variance_predictor(
    float* out,      // output [seq_len]
    float* x,        // input [seq_len, dim]
    float* conv1_w,  // [dim, dim, 3]
    float* conv1_b,
    float* conv2_w,
    float* conv2_b,
    float* linear_w, // [dim, 1]
    float* norm1_w,
    float* norm1_b,
    float* norm2_w,
    float* norm2_b,
    float* buf,      // buffer [seq_len, dim]
    int seq_len,
    int dim
) {
    // Conv1D + ReLU + LayerNorm
    conv1d(buf, x, conv1_w, conv1_b, seq_len, dim, dim, 3);
    for (int i = 0; i < seq_len * dim; i++) {
        buf[i] = relu(buf[i]);
    }
    layer_norm(buf, buf, norm1_w, norm1_b, seq_len, dim);

    // Conv1D + ReLU + LayerNorm
    conv1d(x, buf, conv2_w, conv2_b, seq_len, dim, dim, 3);
    for (int i = 0; i < seq_len * dim; i++) {
        x[i] = relu(x[i]);
    }
    layer_norm(x, x, norm2_w, norm2_b, seq_len, dim);

    // Linear projection to scalar
    for (int i = 0; i < seq_len; i++) {
        float val = 0.0f;
        for (int j = 0; j < dim; j++) {
            val += x[i * dim + j] * linear_w[j];
        }
        out[i] = val;
    }
}

// ----------------------------------------------------------------------------
// Length Regulator

int length_regulate(
    float* out,          // output [total_frames, dim]
    float* x,            // input [seq_len, dim]
    int* durations,      // duration for each phoneme [seq_len]
    int seq_len,
    int dim
) {
    int total_frames = 0;

    // 計算總幀數
    for (int i = 0; i < seq_len; i++) {
        total_frames += durations[i];
    }

    // 擴展序列
    int out_idx = 0;
    for (int i = 0; i < seq_len; i++) {
        for (int d = 0; d < durations[i]; d++) {
            memcpy(&out[out_idx * dim], &x[i * dim], dim * sizeof(float));
            out_idx++;
        }
    }

    return total_frames;
}
```

---

## 主程序結構

```cpp
// fastspeech2.cpp (main function)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

// ... (所有算子實現在上面)

// ----------------------------------------------------------------------------
// 權重加載（使用 mmap）

void load_weights(Weights* w, Config* config, const char* checkpoint_path) {
    int fd = open(checkpoint_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Failed to open %s\n", checkpoint_path);
        exit(1);
    }

    // 獲取文件大小
    off_t file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);

    // mmap 映射整個文件
    void* data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        fprintf(stderr, "mmap failed\n");
        exit(1);
    }

    // 讀取 header
    char* ptr = (char*)data;
    memcpy(config, ptr + 4, sizeof(Config));  // skip magic
    ptr += 256;  // skip header

    // 將所有權重指針指向 mmap 區域
    float* weights_ptr = (float*)ptr;

    int offset = 0;
    w->encoder_tok_emb = weights_ptr + offset;
    offset += config->vocab_size * config->dim;

    w->encoder_pos_emb = weights_ptr + offset;
    offset += config->max_seq_len * config->dim;

    // ... (按順序分配所有權重指針)

    close(fd);
}

// ----------------------------------------------------------------------------
// 完整推理流程

void forward(
    float* mel_output,      // output [mel_len, n_mels]
    int* mel_len_out,       // output mel length
    int* phoneme_ids,       // input [seq_len]
    int seq_len,
    Config* config,
    Weights* weights,
    RunState* state,
    float pitch_control,
    float energy_control,
    float duration_control
) {
    int dim = config->dim;
    int n_heads = config->n_heads;

    // 1. Encoder
    // Token embedding
    for (int i = 0; i < seq_len; i++) {
        memcpy(&state->x[i * dim],
               &weights->encoder_tok_emb[phoneme_ids[i] * dim],
               dim * sizeof(float));
    }

    // Add positional encoding
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < dim; j++) {
            state->x[i * dim + j] += weights->encoder_pos_emb[i * dim + j];
        }
    }

    // Encoder layers
    for (int l = 0; l < config->n_layers; l++) {
        // Self-attention
        multi_head_attention(
            state->xb,
            weights->encoder_attn_q[l],
            weights->encoder_attn_k[l],
            weights->encoder_attn_v[l],
            weights->encoder_attn_o[l],
            state->x,
            state->q, state->k, state->v, state->att,
            seq_len, dim, n_heads
        );

        // Residual + LayerNorm
        for (int i = 0; i < seq_len * dim; i++) {
            state->x[i] += state->xb[i];
        }
        layer_norm(state->x, state->x,
                   weights->encoder_attn_norm_w[l],
                   weights->encoder_attn_norm_b[l],
                   seq_len, dim);

        // FFN (simplified: Linear + ReLU + Linear)
        matmul(state->xb, state->x, weights->encoder_ffn_w1[l], seq_len, dim);
        for (int i = 0; i < seq_len * dim; i++) {
            state->xb[i] = relu(state->xb[i]);
        }
        matmul(state->xb, state->xb, weights->encoder_ffn_w2[l], seq_len, dim);

        // Residual + LayerNorm
        for (int i = 0; i < seq_len * dim; i++) {
            state->x[i] += state->xb[i];
        }
        layer_norm(state->x, state->x,
                   weights->encoder_ffn_norm_w[l],
                   weights->encoder_ffn_norm_b[l],
                   seq_len, dim);
    }

    // 2. Variance Adaptor
    // Duration prediction
    variance_predictor(
        state->logits,
        state->x,
        weights->duration_conv1_w, weights->duration_conv1_b,
        weights->duration_conv2_w, weights->duration_conv2_b,
        weights->duration_linear_w,
        weights->duration_norm1_w, weights->duration_norm1_b,
        weights->duration_norm2_w, weights->duration_norm2_b,
        state->xb,
        seq_len, dim
    );

    // Convert log duration to actual duration
    int* durations = (int*)malloc(seq_len * sizeof(int));
    for (int i = 0; i < seq_len; i++) {
        float log_dur = state->logits[i];
        int dur = (int)(expf(log_dur) - 1.0f) * duration_control;
        durations[i] = dur > 0 ? dur : 1;
    }

    // Pitch prediction + embedding (simplified)
    // ... (similar to duration)

    // Energy prediction + embedding
    // ... (similar to duration)

    // Length regulate
    int mel_len = length_regulate(state->xb, state->x, durations, seq_len, dim);
    memcpy(state->x, state->xb, mel_len * dim * sizeof(float));

    // 3. Decoder
    // Add positional encoding
    for (int i = 0; i < mel_len; i++) {
        for (int j = 0; j < dim; j++) {
            state->x[i * dim + j] += weights->decoder_pos_emb[i * dim + j];
        }
    }

    // Decoder layers (similar to encoder)
    for (int l = 0; l < 6; l++) {  // 6 decoder layers
        // ... (similar structure)
    }

    // 4. Mel Linear
    int n_mels = config->n_mels;
    for (int i = 0; i < mel_len; i++) {
        for (int j = 0; j < n_mels; j++) {
            float val = 0.0f;
            for (int k = 0; k < dim; k++) {
                val += state->x[i * dim + k] * weights->mel_linear_w[j * dim + k];
            }
            mel_output[i * n_mels + j] = val + weights->mel_linear_b[j];
        }
    }

    // 5. PostNet (optional, can skip for simplicity)
    // ...

    *mel_len_out = mel_len;
    free(durations);
}

// ----------------------------------------------------------------------------
// Main

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.bin> <phoneme_ids>\n", argv[0]);
        return 1;
    }

    const char* checkpoint = argv[1];

    // Parse phoneme IDs from command line or stdin
    // Example: "23,15,8,32" -> [23, 15, 8, 32]

    // Load model
    Config config;
    Weights weights;
    load_weights(&weights, &config, checkpoint);

    // Allocate run state
    RunState state;
    state.x = (float*)malloc(config.max_seq_len * config.dim * sizeof(float));
    state.xb = (float*)malloc(config.max_seq_len * config.dim * sizeof(float));
    state.q = (float*)malloc(config.max_seq_len * config.dim * sizeof(float));
    state.k = (float*)malloc(config.max_seq_len * config.dim * sizeof(float));
    state.v = (float*)malloc(config.max_seq_len * config.dim * sizeof(float));
    state.att = (float*)malloc(config.n_heads * config.max_seq_len *
                               config.max_seq_len * sizeof(float));
    state.mel_output = (float*)malloc(2000 * config.n_mels * sizeof(float));

    // Run inference
    int phoneme_ids[] = {23, 15, 8, 32};  // Example: "Hello"
    int seq_len = 4;
    int mel_len;

    forward(
        state.mel_output,
        &mel_len,
        phoneme_ids,
        seq_len,
        &config,
        &weights,
        &state,
        1.0f, 1.0f, 1.0f  // pitch, energy, duration control
    );

    // Save mel output
    FILE* f = fopen("output.mel", "wb");
    fwrite(&mel_len, sizeof(int), 1, f);
    fwrite(&config.n_mels, sizeof(int), 1, f);
    fwrite(state.mel_output, sizeof(float), mel_len * config.n_mels, f);
    fclose(f);

    printf("Generated %d frames\n", mel_len);

    // Cleanup
    free(state.x);
    free(state.xb);
    free(state.q);
    free(state.k);
    free(state.v);
    free(state.att);
    free(state.mel_output);

    return 0;
}
```

---

## 權重轉換工具

```python
# convert_weights.py

import torch
import struct
import numpy as np

def convert_pytorch_to_binary(checkpoint_path, output_path):
    """轉換 PyTorch checkpoint 為二進制格式"""

    # 載入 PyTorch checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']

    # 從 config 讀取參數
    config = {
        'dim': 256,
        'n_layers': 4,
        'n_heads': 2,
        'n_mels': 80,
        'vocab_size': 76,
        'max_seq_len': 1000,
    }

    with open(output_path, 'wb') as f:
        # 寫入 header
        f.write(b'FS2\x00')  # Magic
        f.write(struct.pack('I', 1))  # Version

        # 寫入 config
        f.write(struct.pack('I', config['dim']))
        f.write(struct.pack('I', config['n_layers']))
        f.write(struct.pack('I', config['n_heads']))
        f.write(struct.pack('I', config['n_mels']))
        f.write(struct.pack('I', config['vocab_size']))
        f.write(struct.pack('I', config['max_seq_len']))

        # Padding to 256 bytes
        f.write(b'\x00' * (256 - 4 - 4 - 4*6))

        # 寫入權重（按順序）
        def write_tensor(name):
            tensor = state_dict[name].numpy().astype(np.float32)
            f.write(tensor.tobytes())

        # Encoder
        write_tensor('encoder.src_word_emb.weight')
        write_tensor('encoder.position_enc')

        # Encoder layers
        for i in range(4):
            write_tensor(f'encoder.layer_stack.{i}.slf_attn.w_qs.weight')
            write_tensor(f'encoder.layer_stack.{i}.slf_attn.w_ks.weight')
            write_tensor(f'encoder.layer_stack.{i}.slf_attn.w_vs.weight')
            write_tensor(f'encoder.layer_stack.{i}.slf_attn.fc.weight')
            # ... 繼續寫入其他權重

        # Variance Adaptor
        # ...

        # Decoder
        # ...

        # Mel Linear
        # ...

        # PostNet
        # ...

    print(f"Converted to {output_path}")

if __name__ == '__main__':
    import sys
    convert_pytorch_to_binary(sys.argv[1], sys.argv[2])
```

---

## 簡化開發路線圖

### Week 1: 基礎架構
- [ ] 定義數據結構（Config, Weights, RunState）
- [ ] 實現權重加載（mmap）
- [ ] 實現基礎算子（matmul, layer_norm, softmax）

### Week 2: 核心算子
- [ ] 實現 Conv1D
- [ ] 實現 Multi-Head Attention
- [ ] 單元測試（與 NumPy 對比）

### Week 3: Encoder
- [ ] 實現 Encoder forward
- [ ] 與 PyTorch 輸出對比

### Week 4: Variance Adaptor
- [ ] 實現 Variance Predictor
- [ ] 實現 Length Regulator
- [ ] 測試

### Week 5: Decoder & Output
- [ ] 實現 Decoder
- [ ] 實現 Mel Linear
- [ ] 端到端測試

### Week 6: 優化 & 工具
- [ ] 性能優化（SIMD hints）
- [ ] 權重轉換工具
- [ ] 文檔和示例

---

## 性能優化技巧

```cpp
// 1. 編譯器優化
// g++ -O3 -march=native -ffast-math

// 2. Loop unrolling
for (int i = 0; i < n; i += 4) {
    out[i+0] = x[i+0] * w[i+0];
    out[i+1] = x[i+1] * w[i+1];
    out[i+2] = x[i+2] * w[i+2];
    out[i+3] = x[i+3] * w[i+3];
}

// 3. 使用 restrict 關鍵字
void matmul(float* restrict out, float* restrict x,
            float* restrict w, int n, int d) {
    // 告訴編譯器指針不會重疊
}

// 4. OpenMP 並行（可選）
#pragma omp parallel for
for (int i = 0; i < n; i++) {
    // parallel loop
}
```

---

## 優勢

✅ **極簡**: 單文件 ~1500 行
✅ **零依賴**: 只用標準庫
✅ **快速編譯**: < 5 秒
✅ **小型二進制**: < 100 KB
✅ **易讀**: 直接、沒有抽象
✅ **可移植**: 跨平台
✅ **易調試**: 簡單的代碼結構

---

## 下一步

1. **審視規劃** - 確認這個極簡風格符合需求
2. **開始實現** - 從基礎算子開始
3. **逐步測試** - 每個算子都與 PyTorch 對比

這個設計是否符合您的需求？我們可以開始實現第一部分嗎？
