# FastSpeech2 C++ 推理引擎規劃文檔

**版本**: 1.0
**日期**: 2025-10-23
**目標**: 將 FastSpeech2 從 PyTorch 轉換為純 C++ 推理引擎

---

## 目錄

1. [項目概述](#項目概述)
2. [技術棧選擇](#技術棧選擇)
3. [系統架構設計](#系統架構設計)
4. [核心資料結構](#核心資料結構)
5. [組件模塊設計](#組件模塊設計)
6. [最小可執行單位 (MVP)](#最小可執行單位-mvp)
7. [開發路線圖](#開發路線圖)
8. [風險評估](#風險評估)

---

## 項目概述

### 目標

構建一個高性能、跨平台的 FastSpeech2 純 C++ 推理引擎，用於文字轉語音 (TTS) 任務。

### 範圍

**包含**:
- ✅ 文字預處理 (Phoneme 轉換)
- ✅ FastSpeech2 模型推理 (Encoder → Variance Adaptor → Decoder → PostNet)
- ✅ 權重加載 (支持 PyTorch .pth 或 GGUF 格式)
- ✅ Mel-spectrogram 生成
- ✅ 基礎推理 API

**不包含** (後續擴展):
- ❌ Vocoder (HiFi-GAN/MelGAN) - 階段二
- ❌ 訓練功能
- ❌ Python Binding - 階段三
- ❌ GPU 加速 - 階段四

### 性能目標

- **推理延遲**: < 100ms (CPU, 單句推理)
- **內存占用**: < 200MB (模型載入後)
- **準確性**: 與 PyTorch 版本輸出 MSE < 1e-4

---

## 技術棧選擇

### 依賴庫評估

| 庫 | 用途 | 優點 | 缺點 | 選擇 |
|---|------|------|------|------|
| **Eigen** | 線性代數 | 輕量、純 header、快速 | 無 GPU 支持 | ✅ **推薦** |
| **xtensor** | 多維數組 | NumPy 風格 API | 較大依賴 | ⚠️ 備選 |
| **OpenBLAS** | BLAS 加速 | 高性能矩陣運算 | 需要外部庫 | ✅ **可選** |
| **nlohmann/json** | JSON 解析 | 易用、現代 C++ | - | ✅ **推薦** |
| **yaml-cpp** | YAML 解析 | 讀取配置文件 | - | ✅ **推薦** |
| **spdlog** | 日誌 | 快速、異步 | - | ✅ **推薦** |
| **libtorch** | PyTorch C++ | 完整 PyTorch 功能 | 依賴巨大 (1GB+) | ❌ **不用** |
| **ONNX Runtime** | ONNX 推理 | 跨平台、優化好 | 需要轉 ONNX | ⚠️ 備選方案 |

### 最終技術棧

```yaml
核心依賴:
  - Eigen 3.4+          # 矩陣運算
  - nlohmann/json 3.11+ # JSON 解析
  - yaml-cpp 0.7+       # 配置文件
  - spdlog 1.12+        # 日誌

可選依賴:
  - OpenBLAS            # BLAS 加速
  - Intel MKL           # Intel 平台加速

構建工具:
  - CMake 3.15+
  - C++17 或更高

測試:
  - Google Test
  - Google Benchmark
```

### 為何選擇 Eigen？

1. **輕量級**: Header-only，無需額外鏈接
2. **性能**: 編譯期優化，自動向量化 (SIMD)
3. **API 友好**: 類似 NumPy/PyTorch 的操作
4. **成熟度**: 廣泛應用於機器學習項目

```cpp
// Eigen 示例
Eigen::MatrixXf A(3, 3);
Eigen::VectorXf b(3);
Eigen::VectorXf x = A.colPivHouseholderQr().solve(b);
```

---

## 系統架構設計

### 分層架構

```
┌─────────────────────────────────────────────────────┐
│  應用層 (Application Layer)                          │
│  - CLI 工具                                          │
│  - C API (可選)                                      │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  推理層 (Inference Layer)                            │
│  - FastSpeech2Inference                              │
│  - Text Preprocessor                                 │
│  - Model Runner                                      │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  模型層 (Model Layer)                                │
│  - Encoder                                           │
│  - VarianceAdaptor                                   │
│  - Decoder                                           │
│  - PostNet                                           │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  算子層 (Operator Layer)                             │
│  - Embedding, Linear, Conv1D                         │
│  - MultiHeadAttention                                │
│  - LayerNorm, BatchNorm                              │
│  - Activation (ReLU, Tanh, Softmax)                  │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  核心層 (Core Layer)                                 │
│  - Tensor (基於 Eigen)                               │
│  - WeightLoader (加載 .pth / GGUF)                   │
│  - Config (配置管理)                                 │
└─────────────────────────────────────────────────────┘
```

### 目錄結構

```
fastspeech2_cpp/
├── CMakeLists.txt
├── README.md
├── docs/
│   └── architecture.md
├── include/
│   └── fastspeech2/
│       ├── core/
│       │   ├── tensor.h           # Tensor 封裝
│       │   ├── config.h           # 配置管理
│       │   └── weight_loader.h    # 權重加載
│       ├── ops/
│       │   ├── embedding.h        # Embedding 層
│       │   ├── linear.h           # 全連接層
│       │   ├── conv1d.h           # 1D 卷積
│       │   ├── attention.h        # Multi-Head Attention
│       │   ├── layer_norm.h       # LayerNorm
│       │   ├── batch_norm.h       # BatchNorm
│       │   └── activation.h       # 激活函數
│       ├── layers/
│       │   ├── encoder.h          # Encoder
│       │   ├── variance_adaptor.h # Variance Adaptor
│       │   ├── decoder.h          # Decoder
│       │   ├── postnet.h          # PostNet
│       │   └── fft_block.h        # FFT Block
│       ├── text/
│       │   ├── phonemizer.h       # 文字轉音素
│       │   └── symbols.h          # 音素符號表
│       └── inference.h            # 推理接口
├── src/
│   ├── core/
│   │   ├── tensor.cpp
│   │   ├── config.cpp
│   │   └── weight_loader.cpp
│   ├── ops/
│   │   ├── embedding.cpp
│   │   ├── linear.cpp
│   │   ├── conv1d.cpp
│   │   ├── attention.cpp
│   │   ├── layer_norm.cpp
│   │   ├── batch_norm.cpp
│   │   └── activation.cpp
│   ├── layers/
│   │   ├── encoder.cpp
│   │   ├── variance_adaptor.cpp
│   │   ├── decoder.cpp
│   │   ├── postnet.cpp
│   │   └── fft_block.cpp
│   ├── text/
│   │   ├── phonemizer.cpp
│   │   └── symbols.cpp
│   └── inference.cpp
├── tools/
│   ├── convert_weights.py        # PyTorch -> C++ 權重轉換
│   └── benchmark.cpp              # 性能測試
├── tests/
│   ├── test_ops.cpp               # 算子測試
│   ├── test_layers.cpp            # 層測試
│   └── test_inference.cpp         # 推理測試
└── examples/
    └── simple_tts.cpp             # 示例程序
```

---

## 核心資料結構

### 1. Tensor 類

```cpp
namespace fs2 {

class Tensor {
public:
    // 構造函數
    Tensor();
    Tensor(const std::vector<int64_t>& shape, float fill_value = 0.0f);
    Tensor(Eigen::MatrixXf data);

    // 基本操作
    int64_t size(int dim) const;
    int64_t numel() const;
    int64_t ndim() const;
    const std::vector<int64_t>& shape() const;

    // 數據訪問
    float* data();
    const float* data() const;
    Eigen::Map<Eigen::MatrixXf> matrix();  // 2D view
    Eigen::Map<Eigen::VectorXf> vector();  // 1D view

    // Reshape / View
    Tensor reshape(const std::vector<int64_t>& new_shape) const;
    Tensor view(const std::vector<int64_t>& new_shape) const;
    Tensor transpose(int dim0, int dim1) const;
    Tensor squeeze(int dim = -1) const;
    Tensor unsqueeze(int dim) const;

    // 數學運算
    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;  // Element-wise
    Tensor matmul(const Tensor& other) const;     // Matrix multiplication
    Tensor masked_fill(const Tensor& mask, float value) const;

    // 激活函數
    Tensor relu() const;
    Tensor tanh() const;
    Tensor softmax(int dim = -1) const;

    // 工具函數
    void print(const std::string& name = "") const;
    void save(const std::string& path) const;
    static Tensor load(const std::string& path);

private:
    std::vector<int64_t> shape_;
    std::vector<float> data_;

    // Helper: 將多維索引轉為一維
    int64_t offset(const std::vector<int64_t>& indices) const;
};

} // namespace fs2
```

**設計考量**:
- 內部使用 `std::vector<float>` 存儲，兼容 Eigen
- 提供 Eigen::Map 接口，零拷貝訪問
- 支持動態形狀
- 初期只支持 float32，未來可模板化

### 2. Config 類

```cpp
namespace fs2 {

struct ModelConfig {
    // Transformer
    int encoder_layers = 4;
    int encoder_hidden = 256;
    int encoder_heads = 2;
    int decoder_layers = 6;
    int decoder_hidden = 256;
    int decoder_heads = 2;
    int conv_filter_size = 1024;
    std::vector<int> conv_kernel_size = {9, 1};
    float encoder_dropout = 0.2f;
    float decoder_dropout = 0.2f;

    // Variance Predictor
    int variance_filter_size = 256;
    int variance_kernel_size = 3;
    float variance_dropout = 0.5f;
    int n_bins = 256;

    // Audio
    int n_mel_channels = 80;
    int max_seq_len = 1000;
    bool multi_speaker = false;

    // 從 YAML 加載
    static ModelConfig from_yaml(const std::string& path);
};

struct PreprocessConfig {
    std::string dataset;
    std::string language;
    int sampling_rate = 22050;
    int hop_length = 256;
    int n_mel_channels = 80;

    static PreprocessConfig from_yaml(const std::string& path);
};

} // namespace fs2
```

### 3. 權重存儲

```cpp
namespace fs2 {

// 權重字典
using WeightDict = std::unordered_map<std::string, Tensor>;

class WeightLoader {
public:
    // 從 PyTorch .pth 加載 (需要先轉換為中間格式)
    static WeightDict load_from_pth(const std::string& path);

    // 從 GGUF 加載 (階段二)
    static WeightDict load_from_gguf(const std::string& path);

    // 從自定義二進制格式加載
    static WeightDict load_from_binary(const std::string& path);

    // 保存為自定義格式
    static void save_to_binary(const WeightDict& weights,
                               const std::string& path);
};

} // namespace fs2
```

**權重存儲格式** (自定義二進制):

```
Header:
  - Magic number: 0x46533246 ("FS2F")
  - Version: uint32
  - Num tensors: uint32

For each tensor:
  - Name length: uint32
  - Name: char[name_length]
  - Ndim: uint32
  - Shape: int64[ndim]
  - Data type: uint8 (0=float32, 1=float16)
  - Data: float[numel] or half[numel]
```

### 4. 推理輸入/輸出

```cpp
namespace fs2 {

struct InferenceInput {
    std::vector<int> phoneme_ids;  // 音素 ID 序列
    int speaker_id = 0;             // 說話者 ID (多說話者模式)
    float pitch_control = 1.0f;     // 音調控制
    float energy_control = 1.0f;    // 能量控制
    float duration_control = 1.0f;  // 語速控制
};

struct InferenceOutput {
    Tensor mel_spectrogram;         // (T_mel, 80)
    std::vector<float> durations;   // 預測的 duration
    std::vector<float> pitches;     // 預測的 pitch
    std::vector<float> energies;    // 預測的 energy
    int num_frames;                 // T_mel
};

} // namespace fs2
```

---

## 組件模塊設計

### 算子層 (Ops)

#### 1. Linear Layer

```cpp
namespace fs2 {

class Linear {
public:
    Linear(int in_features, int out_features, bool bias = true);

    // Forward
    Tensor forward(const Tensor& input) const;
    // input: (B, *, in_features)
    // output: (B, *, out_features)

    // 加載權重
    void load_weights(const Tensor& weight, const Tensor& bias = Tensor());

private:
    int in_features_;
    int out_features_;
    bool has_bias_;
    Tensor weight_;  // (out_features, in_features)
    Tensor bias_;    // (out_features,)
};

} // namespace fs2
```

**實現細節**:
```cpp
Tensor Linear::forward(const Tensor& input) const {
    // input: (B, T, in_features)
    // weight: (out_features, in_features)
    // output: (B, T, out_features)

    auto input_2d = input.view({-1, in_features_});  // (B*T, in_features)
    auto output_2d = input_2d.matmul(weight_.transpose(0, 1));  // (B*T, out_features)

    if (has_bias_) {
        output_2d = output_2d + bias_.unsqueeze(0);
    }

    return output_2d.view({input.size(0), input.size(1), out_features_});
}
```

#### 2. Conv1D Layer

```cpp
namespace fs2 {

class Conv1D {
public:
    Conv1D(int in_channels, int out_channels,
           int kernel_size, int padding = 0, int stride = 1);

    Tensor forward(const Tensor& input) const;
    // input: (B, T, in_channels)
    // output: (B, T_out, out_channels)

    void load_weights(const Tensor& weight, const Tensor& bias = Tensor());

private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int padding_;
    int stride_;
    Tensor weight_;  // (out_channels, in_channels, kernel_size)
    Tensor bias_;    // (out_channels,)
};

} // namespace fs2
```

#### 3. Multi-Head Attention

```cpp
namespace fs2 {

class MultiHeadAttention {
public:
    MultiHeadAttention(int d_model, int n_head, float dropout = 0.0f);

    Tensor forward(const Tensor& q, const Tensor& k, const Tensor& v,
                   const Tensor& mask = Tensor()) const;
    // q, k, v: (B, T, d_model)
    // mask: (B, T, T)
    // output: (B, T, d_model)

    void load_weights(const Tensor& w_q, const Tensor& w_k,
                      const Tensor& w_v, const Tensor& w_o,
                      const Tensor& bias_q = Tensor(),
                      const Tensor& bias_k = Tensor(),
                      const Tensor& bias_v = Tensor(),
                      const Tensor& bias_o = Tensor());

private:
    int d_model_;
    int n_head_;
    int d_k_;  // d_model / n_head
    float dropout_;

    Linear w_q_, w_k_, w_v_, w_o_;

    // Scaled dot-product attention
    Tensor scaled_dot_product_attention(
        const Tensor& q, const Tensor& k, const Tensor& v,
        const Tensor& mask = Tensor()) const;
};

} // namespace fs2
```

#### 4. LayerNorm

```cpp
namespace fs2 {

class LayerNorm {
public:
    LayerNorm(int normalized_shape, float eps = 1e-5f);

    Tensor forward(const Tensor& input) const;
    // input: (B, T, D)
    // output: (B, T, D)

    void load_weights(const Tensor& weight, const Tensor& bias);

private:
    int normalized_shape_;
    float eps_;
    Tensor weight_;  // (normalized_shape,)
    Tensor bias_;    // (normalized_shape,)
};

} // namespace fs2
```

### 層級組件 (Layers)

#### 1. FFT Block

```cpp
namespace fs2 {

class FFTBlock {
public:
    FFTBlock(const ModelConfig& config);

    Tensor forward(const Tensor& input, const Tensor& mask) const;
    // input: (B, T, d_model)
    // mask: (B, T)
    // output: (B, T, d_model)

    void load_weights(const WeightDict& weights, const std::string& prefix);

private:
    std::unique_ptr<MultiHeadAttention> attn_;
    std::unique_ptr<LayerNorm> attn_norm_;
    std::unique_ptr<Conv1D> ffn_conv1_;
    std::unique_ptr<Conv1D> ffn_conv2_;
    std::unique_ptr<LayerNorm> ffn_norm_;
};

} // namespace fs2
```

#### 2. Encoder

```cpp
namespace fs2 {

class Encoder {
public:
    Encoder(const ModelConfig& config);

    Tensor forward(const Tensor& input_ids, const Tensor& mask) const;
    // input_ids: (B, T)
    // mask: (B, T)
    // output: (B, T, d_model)

    void load_weights(const WeightDict& weights);

private:
    ModelConfig config_;
    std::unique_ptr<Embedding> token_embedding_;
    Tensor positional_encoding_;  // (max_seq_len, d_model)
    std::vector<std::unique_ptr<FFTBlock>> layers_;
};

} // namespace fs2
```

#### 3. Variance Adaptor

```cpp
namespace fs2 {

class VariancePredictor {
public:
    VariancePredictor(const ModelConfig& config);

    Tensor forward(const Tensor& input, const Tensor& mask) const;
    // input: (B, T, d_model)
    // output: (B, T)

    void load_weights(const WeightDict& weights, const std::string& prefix);

private:
    std::unique_ptr<Conv1D> conv1_;
    std::unique_ptr<LayerNorm> norm1_;
    std::unique_ptr<Conv1D> conv2_;
    std::unique_ptr<LayerNorm> norm2_;
    std::unique_ptr<Linear> linear_;
};

class VarianceAdaptor {
public:
    VarianceAdaptor(const ModelConfig& config,
                    const PreprocessConfig& preprocess_config);

    struct Output {
        Tensor encoder_output;  // (B, T_mel, d_model)
        Tensor pitch_pred;      // (B, T_text)
        Tensor energy_pred;     // (B, T_text)
        Tensor duration_pred;   // (B, T_text)
        std::vector<int> mel_lens;
    };

    Output forward(const Tensor& encoder_output, const Tensor& src_mask,
                   float p_control = 1.0f, float e_control = 1.0f,
                   float d_control = 1.0f) const;

    void load_weights(const WeightDict& weights);

private:
    std::unique_ptr<VariancePredictor> duration_predictor_;
    std::unique_ptr<VariancePredictor> pitch_predictor_;
    std::unique_ptr<VariancePredictor> energy_predictor_;
    std::unique_ptr<Embedding> pitch_embedding_;
    std::unique_ptr<Embedding> energy_embedding_;

    Tensor pitch_bins_;   // (n_bins-1,)
    Tensor energy_bins_;  // (n_bins-1,)

    // Length Regulator
    Tensor length_regulate(const Tensor& input,
                           const std::vector<int>& durations) const;
};

} // namespace fs2
```

### 推理接口

```cpp
namespace fs2 {

class FastSpeech2Inference {
public:
    FastSpeech2Inference(const std::string& model_config_path,
                         const std::string& preprocess_config_path,
                         const std::string& weight_path);

    // 推理接口
    InferenceOutput synthesize(const InferenceInput& input);

    // 批次推理
    std::vector<InferenceOutput> synthesize_batch(
        const std::vector<InferenceInput>& inputs);

private:
    ModelConfig model_config_;
    PreprocessConfig preprocess_config_;

    std::unique_ptr<Encoder> encoder_;
    std::unique_ptr<VarianceAdaptor> variance_adaptor_;
    std::unique_ptr<Decoder> decoder_;
    std::unique_ptr<Linear> mel_linear_;
    std::unique_ptr<PostNet> postnet_;
    std::unique_ptr<Embedding> speaker_embedding_;  // 多說話者模式
};

} // namespace fs2
```

---

## 最小可執行單位 (MVP)

### MVP 定義

最小可執行單位應該能夠：
1. ✅ 加載預訓練權重
2. ✅ 接受音素序列輸入
3. ✅ 輸出 Mel-spectrogram
4. ✅ 輸出與 PyTorch 版本數值接近 (誤差 < 1e-3)

### MVP 範圍

**階段 0: 核心基礎設施** (優先級: 最高)
- [ ] Tensor 類實現
- [ ] Config 加載
- [ ] 權重加載器 (簡化版)
- [ ] 基礎算子: Linear, LayerNorm, ReLU

**階段 1: 簡化推理** (優先級: 高)
- [ ] Encoder (不含 positional encoding)
- [ ] 簡化 Variance Adaptor (固定 duration)
- [ ] Decoder
- [ ] Mel Linear
- [ ] 跳過 PostNet

**階段 2: 完整推理** (優先級: 中)
- [ ] 完整 Variance Adaptor
- [ ] PostNet
- [ ] Positional Encoding
- [ ] Multi-Speaker 支持

**階段 3: 優化與工具** (優先級: 低)
- [ ] SIMD 優化
- [ ] 多線程
- [ ] 權重量化
- [ ] Benchmark 工具

### MVP 示例代碼

```cpp
#include "fastspeech2/inference.h"
#include <iostream>

int main() {
    // 初始化
    fs2::FastSpeech2Inference tts(
        "config/model.yaml",
        "config/preprocess.yaml",
        "weights/model.bin"
    );

    // 準備輸入 (音素序列)
    fs2::InferenceInput input;
    input.phoneme_ids = {23, 15, 8, 32};  // "Hello" 的音素 IDs
    input.pitch_control = 1.0f;
    input.energy_control = 1.0f;
    input.duration_control = 1.0f;

    // 推理
    auto output = tts.synthesize(input);

    // 輸出
    std::cout << "Generated mel shape: ("
              << output.num_frames << ", 80)" << std::endl;

    // 保存 mel-spectrogram
    output.mel_spectrogram.save("output.mel");

    return 0;
}
```

**編譯**:
```bash
mkdir build && cd build
cmake ..
make
./fastspeech2_example
```

---

## 開發路線圖

### 優先順序排列

#### 🔴 P0 - 核心基礎 (第 1-2 週)

| 任務 | 工作量 | 依賴 | 輸出 |
|------|--------|------|------|
| **1. 項目初始化** | 1 天 | - | CMakeLists.txt, 目錄結構 |
| **2. Tensor 類** | 3 天 | - | tensor.h/cpp, 單元測試 |
| **3. Config 加載** | 2 天 | - | config.h/cpp, YAML 解析 |
| **4. 權重加載器** | 3 天 | Tensor | weight_loader.h/cpp |
| **5. 基礎算子** | 3 天 | Tensor | Linear, LayerNorm, Embedding |

**里程碑**: 能夠加載權重並執行矩陣運算

#### 🟠 P1 - 核心推理組件 (第 3-5 週)

| 任務 | 工作量 | 依賴 | 輸出 |
|------|--------|------|------|
| **6. Conv1D** | 2 天 | Tensor | conv1d.h/cpp |
| **7. Activation** | 1 天 | Tensor | activation.h/cpp (ReLU, Tanh, Softmax) |
| **8. Multi-Head Attention** | 4 天 | Linear, LayerNorm | attention.h/cpp |
| **9. FFT Block** | 2 天 | Attention, Conv1D | fft_block.h/cpp |
| **10. Encoder** | 3 天 | FFT Block, Embedding | encoder.h/cpp |
| **11. Decoder** | 2 天 | FFT Block | decoder.h/cpp |

**里程碑**: Encoder 和 Decoder 可以獨立測試

#### 🟡 P2 - Variance Adaptor (第 6-7 週)

| 任務 | 工作量 | 依賴 | 輸出 |
|------|--------|------|------|
| **12. Variance Predictor** | 3 天 | Conv1D, Linear | variance_predictor.cpp |
| **13. Length Regulator** | 3 天 | Tensor | length_regulator.cpp |
| **14. Quantization & Embedding** | 2 天 | Embedding | pitch/energy embedding |
| **15. Variance Adaptor** | 3 天 | 12-14 | variance_adaptor.h/cpp |

**里程碑**: 完整的 Variance Adaptor 可運行

#### 🟢 P3 - 完整推理流程 (第 8-9 週)

| 任務 | 工作量 | 依賴 | 輸出 |
|------|--------|------|------|
| **16. Mel Linear** | 1 天 | Linear | 簡單包裝 |
| **17. PostNet** | 3 天 | Conv1D, BatchNorm | postnet.h/cpp |
| **18. 推理接口** | 3 天 | 所有組件 | inference.h/cpp |
| **19. 文字預處理** | 3 天 | - | phonemizer.h/cpp |
| **20. 端到端測試** | 2 天 | 所有組件 | 與 PyTorch 對比 |

**里程碑**: 完整推理流程可運行，輸出正確

#### 🔵 P4 - 優化與工具 (第 10+ 週)

| 任務 | 工作量 | 依賴 | 輸出 |
|------|--------|------|------|
| **21. 性能優化** | 5 天 | 完整推理 | SIMD, 多線程 |
| **22. 內存優化** | 3 天 | 完整推理 | In-place 操作 |
| **23. 權重轉換工具** | 2 天 | - | convert_weights.py |
| **24. Benchmark** | 2 天 | 完整推理 | benchmark.cpp |
| **25. 文檔** | 3 天 | - | API 文檔, 教程 |

**里程碑**: 性能達標，可投入生產

### Gantt 圖 (簡化版)

```
Week 1-2:  [項目初始化][Tensor][Config][權重加載][基礎算子]
Week 3-5:  [Conv1D][Activation][Attention][FFT][Encoder][Decoder]
Week 6-7:  [Variance Predictor][Length Regulator][Variance Adaptor]
Week 8-9:  [Mel Linear][PostNet][推理接口][文字預處理][測試]
Week 10+:  [性能優化][工具開發][文檔]
```

### 詳細任務分解

#### 任務 1: 項目初始化

```bash
# 工作內容
1. 創建 Git 倉庫
2. 設置 CMake 構建系統
3. 配置依賴管理 (vcpkg / Conan)
4. 設置 CI/CD (GitHub Actions)
5. 創建目錄結構

# 產出
- CMakeLists.txt
- .github/workflows/ci.yml
- README.md
- docs/build.md
```

#### 任務 2: Tensor 類

```cpp
// 工作內容
1. 實現基本構造函數
2. 實現 reshape, view, transpose
3. 實現基本數學運算 (+, -, *, matmul)
4. 實現 masked_fill
5. 編寫單元測試 (Google Test)

// 測試用例
TEST(TensorTest, MatMul) {
    Tensor A({2, 3}, 1.0f);
    Tensor B({3, 4}, 2.0f);
    Tensor C = A.matmul(B);
    EXPECT_EQ(C.shape(), std::vector<int64_t>({2, 4}));
    EXPECT_FLOAT_EQ(C.data()[0], 6.0f);  // 1*2 + 1*2 + 1*2
}
```

#### 任務 8: Multi-Head Attention

```cpp
// 工作內容
1. 實現 scaled dot-product attention
2. 實現 multi-head 分割與合併
3. 實現 mask 機制
4. 與 PyTorch 輸出對比測試

// 偽代碼
Tensor MultiHeadAttention::forward(q, k, v, mask) {
    // 1. Linear projection
    Q = w_q.forward(q);  // (B, T, d_model)
    K = w_k.forward(k);
    V = w_v.forward(v);

    // 2. Split heads
    Q = split_heads(Q);  // (B, n_head, T, d_k)
    K = split_heads(K);
    V = split_heads(V);

    // 3. Scaled dot-product
    scores = Q.matmul(K.transpose(-2, -1)) / sqrt(d_k);
    scores = scores.masked_fill(mask, -1e9);
    attn = scores.softmax(-1);

    // 4. Weighted sum
    output = attn.matmul(V);  // (B, n_head, T, d_v)

    // 5. Concatenate heads
    output = merge_heads(output);  // (B, T, d_model)
    output = w_o.forward(output);

    return output;
}
```

---

## 風險評估

### 技術風險

| 風險 | 概率 | 影響 | 緩解措施 |
|------|------|------|----------|
| **浮點誤差累積** | 高 | 中 | 每層對比 PyTorch，使用 double 調試 |
| **Conv1D 實現複雜** | 中 | 中 | 參考 ONNX Runtime 實現 |
| **內存管理錯誤** | 中 | 高 | 使用智能指針，ASAN 檢測 |
| **性能不達標** | 中 | 中 | 使用 OpenBLAS/MKL 加速 |
| **權重加載失敗** | 低 | 高 | 提供多種格式支持 |

### 進度風險

| 風險 | 概率 | 影響 | 緩解措施 |
|------|------|------|----------|
| **時間估算不準** | 高 | 中 | 採用迭代開發，MVP 優先 |
| **依賴庫問題** | 中 | 低 | 鎖定版本，提供 Docker |
| **測試不充分** | 中 | 高 | 每個組件獨立測試 |

---

## 性能考量

### 性能優化策略

#### 1. 編譯期優化

```cmake
# CMakeLists.txt
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
add_compile_options(-ffast-math -ftree-vectorize)
```

#### 2. Eigen 優化

```cpp
// 啟用 Eigen 並行化
#define EIGEN_USE_THREADS
#include <Eigen/Core>

// 使用 Eigen 的向量化操作
output = (input.array() * weight.array()).matrix();
```

#### 3. 內存優化

```cpp
// In-place 操作
void relu_inplace(Tensor& t) {
    std::transform(t.data(), t.data() + t.numel(), t.data(),
                   [](float x) { return std::max(0.0f, x); });
}

// 對象池 (避免頻繁分配)
class TensorPool {
    std::vector<Tensor> pool_;
public:
    Tensor acquire(const std::vector<int64_t>& shape);
    void release(Tensor&& t);
};
```

#### 4. 多線程

```cpp
// Encoder layers 並行 (如果 batch size > 1)
#pragma omp parallel for
for (int b = 0; b < batch_size; ++b) {
    output[b] = encoder.forward(input[b]);
}
```

### 性能目標

| 配置 | 目標延遲 | 目標吞吐 |
|------|----------|----------|
| **CPU (Intel i7)** | < 100ms | > 10 句/秒 |
| **CPU (ARM M1)** | < 80ms | > 12 句/秒 |
| **優化後** | < 50ms | > 20 句/秒 |

---

## 測試策略

### 單元測試

```cpp
// tests/test_ops.cpp
TEST(LinearTest, Forward) {
    Linear linear(256, 512);
    // 加載固定權重
    linear.load_weights(weight, bias);

    Tensor input({2, 10, 256});  // (B, T, in)
    Tensor output = linear.forward(input);

    EXPECT_EQ(output.shape(), std::vector<int64_t>({2, 10, 512}));

    // 與 PyTorch 輸出對比
    Tensor expected = load_reference("linear_output.npy");
    EXPECT_TRUE(allclose(output, expected, 1e-4));
}
```

### 集成測試

```cpp
// tests/test_inference.cpp
TEST(InferenceTest, EndToEnd) {
    FastSpeech2Inference tts("config/model.yaml",
                             "config/preprocess.yaml",
                             "weights/test.bin");

    InferenceInput input;
    input.phoneme_ids = {23, 15, 8, 32};

    auto output = tts.synthesize(input);

    // 與 PyTorch 版本對比
    auto expected_mel = load_reference("expected_mel.npy");
    EXPECT_TRUE(allclose(output.mel_spectrogram, expected_mel, 1e-3));
}
```

---

## 下一步行動

### 立即可執行

1. **審視規劃** - 與團隊討論，確認技術選型
2. **環境準備** - 安裝 Eigen, CMake, Google Test
3. **創建倉庫** - 初始化項目結構
4. **實現 Tensor** - 從核心數據結構開始

### 需要決策的問題

1. **權重格式**: 使用 PyTorch .pth 還是 GGUF？還是自定義格式？
2. **文字預處理**: 集成在 C++ 中還是作為預處理步驟？
3. **BLAS 庫**: 是否使用 OpenBLAS/MKL？
4. **構建系統**: vcpkg, Conan, 還是手動管理依賴？

---

## 附錄

### A. 依賴庫安裝

```bash
# Ubuntu
sudo apt install libeigen3-dev libyaml-cpp-dev

# macOS
brew install eigen yaml-cpp

# vcpkg
vcpkg install eigen3 yaml-cpp nlohmann-json spdlog
```

### B. 參考實現

- [ONNX Runtime](https://github.com/microsoft/onnxruntime) - 算子實現參考
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - 權重加載、量化
- [Kaldi](https://github.com/kaldi-asr/kaldi) - 語音處理工具

### C. 性能 Benchmark 參考

| 模型 | 平台 | 推理時間 |
|------|------|----------|
| Tacotron2 (PyTorch) | CPU | ~500ms |
| FastSpeech2 (PyTorch) | CPU | ~150ms |
| FastSpeech2 (ONNX) | CPU | ~80ms |
| **目標 (C++)** | **CPU** | **< 100ms** |

---

**版本歷史**:
- v1.0 (2025-10-23): 初始規劃文檔
