# FastSpeech2.cpp - 最終實現計劃

**目標**: 輕量、簡潔、架構清晰的 C++ 推理引擎
**風格**: llama2.c 極簡風格 + 模塊化設計

---

## 項目結構

```
fastspeech2_inference/
├── src/
│   ├── fastspeech2.h          # 數據結構定義 (~150 行)
│   ├── op.h                   # 算子聲明 (~100 行)
│   ├── op.cpp                 # 算子實現 (~800 行)
│   └── fastspeech2.cpp        # 主程序 (~600 行)
│
├── tools/
│   ├── convert_weights.py     # PyTorch → 二進制權重
│   ├── text_to_phonemes.py    # 文字 → 音素 ID（前處理）
│   ├── mel_to_wav.py          # Mel → 音訊（後處理，Vocoder）
│   └── verify_output.py       # 驗證正確性
│
├── tests/
│   ├── test_ops.cpp           # 算子單元測試
│   └── test_inference.cpp     # 端到端測試
│
├── examples/
│   ├── phoneme_ids.txt        # 示例輸入
│   └── run_inference.sh       # 完整流程示例
│
├── CMakeLists.txt             # 構建配置
├── Makefile                   # 簡單構建（可選）
└── README.md                  # 使用說明
```

---

## 文件職責說明

### 核心文件

| 文件 | 行數 | 職責 | 主要內容 |
|------|------|------|----------|
| **fastspeech2.h** | ~150 | 數據結構定義 | Config, Weights, RunState, Input/Output |
| **op.h** | ~100 | 算子聲明 | 所有算子的函數聲明 |
| **op.cpp** | ~800 | 算子實現 | matmul, layer_norm, attention, conv1d 等 |
| **fastspeech2.cpp** | ~600 | 主程序 | 權重加載、推理流程、main() |

**總代碼量**: ~1650 行（核心部分）

### 工具文件

| 工具 | 用途 | 輸入 | 輸出 |
|------|------|------|------|
| **convert_weights.py** | 權重轉換 | PyTorch .pth.tar | Binary .bin |
| **text_to_phonemes.py** | 文字預處理 | "Hello world" | phoneme_ids.txt |
| **mel_to_wav.py** | 音訊生成 | output.mel + HiFi-GAN | output.wav |
| **verify_output.py** | 驗證正確性 | C++ output + PyTorch model | MSE report |

---

## 完整使用流程

```bash
# ========== 步驟 1: 轉換權重 ==========
python tools/convert_weights.py \
    output/ckpt/LJSpeech/900000.pth.tar \
    model.bin

# ========== 步驟 2: 編譯 C++ 程序 ==========
make clean && make
# 或使用 CMake:
# mkdir build && cd build && cmake .. && make

# ========== 步驟 3: 文字轉音素（前處理） ==========
python tools/text_to_phonemes.py "Hello world, how are you?"
# 輸出: phoneme_ids.txt

# ========== 步驟 4: C++ 推理 ==========
./fastspeech2 model.bin phoneme_ids.txt
# 輸出: output.mel (二進制格式)

# ========== 步驟 5: Mel 轉音訊（後處理） ==========
python tools/mel_to_wav.py \
    output.mel \
    output.wav \
    hifigan/generator.pth

# ========== 步驟 6: 播放音訊 ==========
play output.wav  # 或使用其他播放器

# ========== 驗證正確性（可選） ==========
python tools/verify_output.py
```

---

## 數據結構設計

### 1. Config - 模型配置

```cpp
typedef struct {
    int dim;              // hidden dimension (256)
    int n_enc_layers;     // encoder layers (4)
    int n_dec_layers;     // decoder layers (6)
    int n_heads;          // attention heads (2)
    int n_mels;           // mel channels (80)
    int vocab_size;       // phoneme vocabulary (76)
    int max_seq_len;      // max sequence length (1000)
    int ffn_hidden;       // FFN hidden size (1024)
    int var_filter_size;  // variance predictor filter (256)
    int n_bins;           // quantization bins (256)
} Config;
```

### 2. Weights - 模型權重

所有權重使用原始 `float*` 指針，通過 **mmap** 直接映射到文件：

```cpp
typedef struct {
    // Encoder
    float* encoder_tok_emb;        // [vocab_size, dim]
    float* encoder_pos_emb;        // [max_seq_len, dim]
    float* enc_attn_q_w[4];        // [dim, dim] × 4 layers
    // ... (所有 encoder 權重)

    // Variance Adaptor
    float* dur_conv1_w;            // Duration predictor
    float* pitch_conv1_w;          // Pitch predictor
    float* energy_conv1_w;         // Energy predictor
    float* pitch_bins;             // [n_bins-1]
    float* pitch_emb;              // [n_bins, dim]
    // ... (所有 variance 權重)

    // Decoder
    float* decoder_pos_emb;
    float* dec_attn_q_w[6];        // [dim, dim] × 6 layers
    // ... (所有 decoder 權重)

    // Output
    float* mel_linear_w;           // [n_mels, dim]
    float* postnet_conv_w[5];      // PostNet 5 layers
    // ... (PostNet 權重)
} Weights;
```

### 3. RunState - 運行時緩衝

所有中間激活值的緩衝區：

```cpp
typedef struct {
    // 通用緩衝
    float* x;              // [max_seq_len, dim] - 主緩衝
    float* xb;             // [max_seq_len, dim] - 備用緩衝
    float* xb2;            // [max_seq_len, dim] - 第二備用

    // Attention 緩衝
    float* q, *k, *v;      // Query, Key, Value
    float* att;            // [n_heads, seq_len, seq_len]
    float* att_out;        // [seq_len, dim]

    // Variance 緩衝
    float* logits;         // [seq_len] - 預測輸出
    int* durations;        // [seq_len] - duration 值

    // 輸出緩衝
    float* mel_output;     // [max_mel_len, n_mels]
    float* mel_postnet;    // [max_mel_len, n_mels]
} RunState;
```

### 4. InferenceInput/Output

```cpp
typedef struct {
    int* phoneme_ids;      // 輸入音素 ID
    int seq_len;           // 序列長度
    float pitch_control;   // 音調控制 (1.0 = 正常)
    float energy_control;  // 音量控制
    float duration_control;// 語速控制
} InferenceInput;

typedef struct {
    float* mel;            // 輸出 mel-spectrogram
    int mel_len;           // mel 幀數
    float* durations;      // 預測的 duration（調試用）
    float* pitches;        // 預測的 pitch
    float* energies;       // 預測的 energy
} InferenceOutput;
```

---

## 算子層設計 (op.cpp)

### 基礎算子

```cpp
// 數學函數
float relu(float x);
void softmax(float* x, int size);
void tanh_inplace(float* x, int size);

// 矩陣運算
void matmul(float* out, const float* x, const float* w,
            int n, int d_in, int d_out);
void add_bias(float* out, const float* bias, int n, int d);
void add(float* out, const float* a, const float* b, int size);

// 歸一化
void layer_norm(float* out, const float* x, const float* w,
                const float* b, int n, int d);
void batch_norm_1d(float* out, const float* x, const float* weight,
                   const float* bias, const float* mean,
                   const float* var, int n, int c);

// 卷積
void conv1d(float* out, const float* x, const float* w,
            const float* b, int seq_len, int in_ch,
            int out_ch, int kernel_size);
```

### 複合算子

```cpp
// Multi-Head Attention
void multi_head_attention(
    float* out,          // [seq_len, dim]
    const float* x,      // [seq_len, dim]
    const float* q_w, const float* q_b,
    const float* k_w, const float* k_b,
    const float* v_w, const float* v_b,
    const float* o_w, const float* o_b,
    float* q_buf, float* k_buf, float* v_buf, float* att,
    int seq_len, int dim, int n_heads
);

// Variance Predictor
void variance_predictor(
    float* out,          // [seq_len]
    const float* x,      // [seq_len, dim]
    const float* conv1_w, const float* conv1_b,
    const float* norm1_w, const float* norm1_b,
    const float* conv2_w, const float* conv2_b,
    const float* norm2_w, const float* norm2_b,
    const float* linear_w, const float* linear_b,
    float* buf,
    int seq_len, int dim, int filter_size
);

// Length Regulator
int length_regulate(float* out, const float* x,
                    const int* durations, int seq_len, int dim);

// PostNet
void postnet_forward(float* out, const float* x, const Weights* w,
                     float* buf, int mel_len, int n_mels);
```

---

## 推理流程 (fastspeech2.cpp::forward)

```cpp
void forward(InferenceOutput* output, InferenceInput* input,
             Config* config, Weights* weights, RunState* state) {

    // ========== 1. Encoder ==========
    // Token embedding + positional encoding
    embedding(state->x, input->phoneme_ids, weights->encoder_tok_emb, ...);
    add(state->x, state->x, weights->encoder_pos_emb, ...);

    // 4 encoder layers
    for (int l = 0; l < 4; l++) {
        // Self-attention
        multi_head_attention(...);

        // Residual + LayerNorm
        add(...);
        layer_norm(...);

        // FFN (Linear + ReLU + Linear)
        matmul(...);
        relu_inplace(...);
        matmul(...);

        // Residual + LayerNorm
        add(...);
        layer_norm(...);
    }

    // ========== 2. Variance Adaptor ==========
    // Duration prediction
    variance_predictor(state->logits, state->x, weights->dur_*, ...);

    // Convert log duration to frames
    for (int i = 0; i < seq_len; i++) {
        state->durations[i] = (int)(exp(state->logits[i]) - 1.0)
                             * input->duration_control;
    }

    // Pitch prediction + quantization + embedding
    variance_predictor(state->logits, state->x, weights->pitch_*, ...);
    quantize(pitch_indices, state->logits, weights->pitch_bins, ...);
    embedding(pitch_emb, pitch_indices, weights->pitch_emb, ...);
    add(state->x, state->x, pitch_emb, ...);

    // Energy prediction (similar)
    // ...

    // Length regulate (expand to frame-level)
    int mel_len = length_regulate(state->xb, state->x, state->durations, ...);
    memcpy(state->x, state->xb, ...);

    // ========== 3. Decoder ==========
    // Add positional encoding
    add(state->x, state->x, weights->decoder_pos_emb, ...);

    // 6 decoder layers (similar to encoder)
    for (int l = 0; l < 6; l++) {
        // ... (same structure as encoder)
    }

    // ========== 4. Mel Linear ==========
    matmul(state->mel_output, state->x, weights->mel_linear_w, ...);
    add_bias(state->mel_output, weights->mel_linear_b, ...);

    // ========== 5. PostNet ==========
    postnet_forward(state->mel_postnet, state->mel_output, weights, ...);

    // ========== Output ==========
    output->mel = state->mel_postnet;
    output->mel_len = mel_len;
}
```

---

## 權重文件格式

```
二進制格式 (.bin):

[Header - 256 bytes]
  - Magic: "FS2\0" (4 bytes)
  - Version: uint32_t (4 bytes)
  - Config struct (40 bytes)
  - Padding (212 bytes)

[Weights - sequential floats]
  - encoder_tok_emb [vocab_size × dim]
  - encoder_pos_emb [max_seq_len × dim]
  - enc_attn_q_w[0] [dim × dim]
  - enc_attn_q_b[0] [dim]
  - ... (按順序排列所有權重)
  - postnet_bn_var[4] [最後一個權重]

Total size: ~35-40 MB (float32)
```

### 加載方式（mmap）

```cpp
void load_weights(const char* path, Config* config, Weights* weights) {
    int fd = open(path, O_RDONLY);
    void* data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

    // 讀取 header
    char* ptr = (char*)data;
    memcpy(config, ptr + 8, sizeof(Config));

    // 權重指針直接指向 mmap 區域（零拷貝）
    float* weights_ptr = (float*)(ptr + 256);
    weights->encoder_tok_emb = weights_ptr;
    weights_ptr += vocab_size * dim;
    weights->encoder_pos_emb = weights_ptr;
    // ...
}
```

---

## 工具實現摘要

### 1. convert_weights.py

```python
def convert_pytorch_to_binary(pth_path, bin_path):
    checkpoint = torch.load(pth_path)
    state_dict = checkpoint['model']

    with open(bin_path, 'wb') as f:
        # Write header
        f.write(b'FS2\x00')
        f.write(struct.pack('I', 1))  # version
        # Write config
        for v in config.values():
            f.write(struct.pack('I', v))
        # Padding
        f.write(b'\x00' * padding)

        # Write weights in order
        write_tensor('encoder.src_word_emb.weight')
        write_tensor('encoder.position_enc')
        # ... (all tensors)
```

### 2. text_to_phonemes.py

```python
def text_to_phoneme_ids(text):
    g2p = G2p()
    phones = g2p(text)

    # Convert to IDs
    phone_ids = [symbols.index(p) for p in phones if p in symbols]

    # Save
    with open('phoneme_ids.txt', 'w') as f:
        for pid in phone_ids:
            f.write(f"{pid}\n")
```

### 3. mel_to_wav.py

```python
def mel_to_wav(mel_path, wav_path, vocoder_path):
    # Load mel
    mel = load_mel_binary(mel_path)  # [mel_len, 80]

    # Load HiFi-GAN
    vocoder = load_hifigan(vocoder_path)

    # Generate audio
    audio = vocoder.infer(mel)

    # Save WAV
    wavfile.write(wav_path, 22050, audio)
```

### 4. verify_output.py

```python
def verify(cpp_mel, pytorch_model, phoneme_ids):
    # Load C++ output
    cpp_mel = load_mel_binary(cpp_mel)

    # Run PyTorch
    pytorch_mel = run_pytorch_inference(pytorch_model, phoneme_ids)

    # Compare
    mse = np.mean((cpp_mel - pytorch_mel) ** 2)
    print(f"MSE: {mse:.6f}")

    if mse < 1e-4:
        print("✓ Pass")
    else:
        print("✗ Fail")
```

---

## 開發路線圖（6 週）

### Week 1: 基礎設施
- [x] 項目結構創建
- [ ] fastspeech2.h 定義
- [ ] 基礎算子實現
  - [ ] matmul
  - [ ] layer_norm
  - [ ] softmax
  - [ ] add, add_bias
- [ ] test_ops.cpp 測試框架
- [ ] convert_weights.py 基本版本

**里程碑**: 基礎算子通過測試

### Week 2: 核心算子
- [ ] Conv1D 實現
- [ ] Multi-Head Attention 實現
- [ ] BatchNorm 實現
- [ ] 對比 PyTorch 輸出（逐層測試）

**里程碑**: 核心算子與 PyTorch 輸出一致 (MSE < 1e-4)

### Week 3: Encoder
- [ ] Encoder forward 實現
- [ ] Token embedding + positional encoding
- [ ] 4 層 encoder layers
- [ ] 端到端測試 encoder 輸出

**里程碑**: Encoder 完整運行，輸出正確

### Week 4: Variance Adaptor
- [ ] Variance Predictor 實現
- [ ] Duration/Pitch/Energy prediction
- [ ] Quantization & Embedding
- [ ] Length Regulator 實現
- [ ] text_to_phonemes.py 工具

**里程碑**: Variance Adaptor 運行，duration 擴展正確

### Week 5: Decoder & Output
- [ ] Decoder 實現 (6 layers)
- [ ] Mel Linear
- [ ] PostNet 實現
- [ ] 權重加載完整實現
- [ ] 端到端推理測試

**里程碑**: 完整推理流程運行，輸出 mel-spectrogram

### Week 6: 優化 & 工具
- [ ] mel_to_wav.py (Vocoder)
- [ ] verify_output.py (驗證)
- [ ] 性能優化（編譯器選項）
- [ ] 完整文檔
- [ ] 示例和教程

**里程碑**: 完整工具鏈可用，從文字到音訊

---

## 性能目標

| 指標 | 目標 | 測試條件 |
|------|------|----------|
| **推理延遲** | < 100ms | 單句，Intel i7 |
| **內存占用** | < 150MB | 模型載入後 |
| **準確性** | MSE < 1e-4 | vs PyTorch |
| **編譯時間** | < 10 秒 | make clean && make |
| **二進制大小** | < 150 KB | 可執行文件 |

---

## 編譯選項

```bash
# 開發版本（調試）
g++ -g -O0 -std=c++17 -o fastspeech2 \
    src/fastspeech2.cpp src/op.cpp

# 發布版本（優化）
g++ -O3 -march=native -ffast-math -std=c++17 \
    -o fastspeech2 src/fastspeech2.cpp src/op.cpp

# 帶 OpenMP（可選）
g++ -O3 -march=native -fopenmp -std=c++17 \
    -o fastspeech2 src/fastspeech2.cpp src/op.cpp
```

---

## 測試策略

### 1. 單元測試 (tests/test_ops.cpp)

```cpp
// 測試每個算子的正確性
TEST(MatMul) {
    float x[2*3] = {1,2,3,4,5,6};
    float w[4*3] = {1,0,0,0,1,0,0,0,1,1,1,1};
    float out[2*4];

    matmul(out, x, w, 2, 3, 4);

    // 驗證結果
    assert(fabs(out[0] - 1.0) < 1e-5);
    // ...
}
```

### 2. 對比測試

每個組件都與 PyTorch 輸出對比：

```python
# 提取 PyTorch 中間層輸出
pytorch_encoder_out = model.encoder(x)

# 與 C++ 對比
cpp_encoder_out = load_binary("encoder_out.bin")
mse = np.mean((pytorch_encoder_out - cpp_encoder_out) ** 2)
assert mse < 1e-4
```

### 3. 端到端測試

```bash
# 完整流程
./run_full_pipeline.sh input.txt output.wav

# 聽感測試
play output.wav
```

---

## FAQ

### Q1: 為什麼不用 Eigen？
**A**: 遵循 llama2.c 極簡哲學，零依賴，便於理解和移植。

### Q2: 性能會不會很差？
**A**: 編譯器優化 (-O3 -march=native) 可以達到接近 BLAS 的性能，對於推理場景足夠。

### Q3: 如何添加 GPU 支持？
**A**: 可以後期使用 CUDA 重寫部分算子，但初期專注 CPU。

### Q4: 支持批次推理嗎？
**A**: 初期不支持，後續可擴展。

### Q5: 如何調試數值錯誤？
**A**: 使用 verify_output.py 逐層對比，定位問題層。

---

## 下一步

準備好開始實現了嗎？我可以：

1. ✅ 生成完整的 fastspeech2.h
2. ✅ 生成 op.h 和 op.cpp 骨架
3. ✅ 生成 fastspeech2.cpp 主程序
4. ✅ 生成 convert_weights.py
5. ✅ 生成 Makefile 和 CMakeLists.txt

請告訴我是否開始！🚀
