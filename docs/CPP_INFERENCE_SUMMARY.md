# FastSpeech2 C++ 推理引擎 - 執行摘要

**目標**: 將 FastSpeech2 從 PyTorch 轉為高性能 C++ 推理引擎

---

## 🎯 核心目標

- ✅ 純 C++ 實現，無 Python 依賴
- ✅ 推理延遲 < 100ms (CPU)
- ✅ 輸出精度與 PyTorch 版本 MSE < 1e-4
- ✅ 跨平台支持 (Linux, macOS, Windows)

---

## 📚 技術選型

### 核心依賴

| 庫 | 用途 | 為何選擇 |
|---|------|----------|
| **Eigen 3.4+** | 線性代數 | 輕量、Header-only、性能優異 |
| **yaml-cpp** | 配置文件 | 讀取模型配置 |
| **nlohmann/json** | JSON 解析 | 權重元數據 |
| **spdlog** | 日誌 | 快速、異步 |

### 可選加速

- OpenBLAS / Intel MKL - BLAS 加速
- OpenMP - 多線程並行

### 不使用的方案

- ❌ **libtorch** - 依賴過大 (1GB+)
- ❌ **ONNX Runtime** - 需要轉換模型

---

## 🏗️ 系統架構

```
應用層 (CLI 工具)
    ↓
推理層 (FastSpeech2Inference)
    ↓
模型層 (Encoder, VarianceAdaptor, Decoder, PostNet)
    ↓
算子層 (Linear, Conv1D, Attention, LayerNorm)
    ↓
核心層 (Tensor, WeightLoader, Config)
```

---

## 📦 目錄結構

```
fastspeech2_cpp/
├── include/fastspeech2/
│   ├── core/          # Tensor, Config, WeightLoader
│   ├── ops/           # Linear, Conv1D, Attention, LayerNorm
│   ├── layers/        # Encoder, Decoder, VarianceAdaptor, PostNet
│   ├── text/          # Phonemizer
│   └── inference.h    # 推理 API
├── src/               # 實現文件
├── tests/             # 單元測試
├── tools/             # 權重轉換工具
└── examples/          # 示例程序
```

---

## 🔑 核心資料結構

### 1. Tensor 類

```cpp
class Tensor {
public:
    Tensor(const std::vector<int64_t>& shape);

    // 形狀操作
    Tensor reshape(const std::vector<int64_t>& new_shape);
    Tensor transpose(int dim0, int dim1);

    // 數學運算
    Tensor operator+(const Tensor& other);
    Tensor matmul(const Tensor& other);
    Tensor masked_fill(const Tensor& mask, float value);

    // Eigen 接口
    Eigen::Map<Eigen::MatrixXf> matrix();

private:
    std::vector<int64_t> shape_;
    std::vector<float> data_;
};
```

### 2. 推理接口

```cpp
struct InferenceInput {
    std::vector<int> phoneme_ids;  // 音素序列
    int speaker_id = 0;
    float pitch_control = 1.0f;
    float energy_control = 1.0f;
    float duration_control = 1.0f;
};

struct InferenceOutput {
    Tensor mel_spectrogram;  // (T_mel, 80)
    std::vector<float> durations;
    std::vector<float> pitches;
    std::vector<float> energies;
};

class FastSpeech2Inference {
public:
    FastSpeech2Inference(const std::string& model_config,
                         const std::string& preprocess_config,
                         const std::string& weights);

    InferenceOutput synthesize(const InferenceInput& input);
};
```

---

## 🚀 最小可執行單位 (MVP)

### MVP 示例

```cpp
#include "fastspeech2/inference.h"

int main() {
    // 初始化模型
    fs2::FastSpeech2Inference tts(
        "config/model.yaml",
        "config/preprocess.yaml",
        "weights/model.bin"
    );

    // 準備輸入
    fs2::InferenceInput input;
    input.phoneme_ids = {23, 15, 8, 32};  // "Hello"

    // 推理
    auto output = tts.synthesize(input);

    // 保存結果
    output.mel_spectrogram.save("output.mel");

    return 0;
}
```

### MVP 範圍

**階段 0 - 核心基礎** (1-2週)
- Tensor 類
- Config 加載
- 權重加載器
- 基礎算子 (Linear, LayerNorm)

**階段 1 - 簡化推理** (3-5週)
- Encoder
- 簡化 Variance Adaptor (固定 duration)
- Decoder
- Mel Linear

**階段 2 - 完整推理** (6-9週)
- 完整 Variance Adaptor
- PostNet
- 文字預處理
- 端到端測試

**階段 3 - 優化** (10+週)
- 性能優化 (SIMD, 多線程)
- 工具開發
- 文檔

---

## 📋 開發任務清單 (按優先順序)

### 🔴 P0 - 核心基礎 (必須完成)

| # | 任務 | 工作量 | 狀態 |
|---|------|--------|------|
| 1 | 項目初始化 (CMake, 目錄結構) | 1天 | ⬜ |
| 2 | Tensor 類實現 | 3天 | ⬜ |
| 3 | Config 加載 (YAML) | 2天 | ⬜ |
| 4 | 權重加載器 | 3天 | ⬜ |
| 5 | Linear 層 | 2天 | ⬜ |
| 6 | LayerNorm | 1天 | ⬜ |
| 7 | Embedding | 1天 | ⬜ |

**里程碑**: 能夠加載權重並執行基本矩陣運算

### 🟠 P1 - 核心推理組件 (高優先級)

| # | 任務 | 工作量 | 狀態 |
|---|------|--------|------|
| 8 | Conv1D 層 | 2天 | ⬜ |
| 9 | Activation (ReLU, Tanh, Softmax) | 1天 | ⬜ |
| 10 | Multi-Head Attention | 4天 | ⬜ |
| 11 | FFT Block | 2天 | ⬜ |
| 12 | Encoder | 3天 | ⬜ |
| 13 | Decoder | 2天 | ⬜ |

**里程碑**: Encoder 和 Decoder 可獨立運行

### 🟡 P2 - Variance Adaptor (中優先級)

| # | 任務 | 工作量 | 狀態 |
|---|------|--------|------|
| 14 | Variance Predictor | 3天 | ⬜ |
| 15 | Length Regulator | 3天 | ⬜ |
| 16 | Pitch/Energy Quantization & Embedding | 2天 | ⬜ |
| 17 | Variance Adaptor 整合 | 3天 | ⬜ |

**里程碑**: Variance Adaptor 可運行

### 🟢 P3 - 完整流程 (中優先級)

| # | 任務 | 工作量 | 狀態 |
|---|------|--------|------|
| 18 | Mel Linear | 1天 | ⬜ |
| 19 | PostNet | 3天 | ⬜ |
| 20 | 推理接口整合 | 3天 | ⬜ |
| 21 | 文字預處理 (Phonemizer) | 3天 | ⬜ |
| 22 | 端到端測試與 PyTorch 對比 | 2天 | ⬜ |

**里程碑**: 完整推理流程，輸出正確

### 🔵 P4 - 優化與工具 (低優先級)

| # | 任務 | 工作量 | 狀態 |
|---|------|--------|------|
| 23 | 性能優化 (SIMD, 多線程) | 5天 | ⬜ |
| 24 | 內存優化 | 3天 | ⬜ |
| 25 | 權重轉換工具 (PyTorch → Binary) | 2天 | ⬜ |
| 26 | Benchmark 工具 | 2天 | ⬜ |
| 27 | 文檔與示例 | 3天 | ⬜ |

**里程碑**: 性能達標，可投入生產

---

## 📊 時間規劃

```
Week 1-2:   ████████░░  P0 核心基礎
Week 3-5:   ████████░░  P1 核心推理組件
Week 6-7:   ████░░░░░░  P2 Variance Adaptor
Week 8-9:   ████░░░░░░  P3 完整流程
Week 10+:   ████░░░░░░  P4 優化與工具
```

**預計總工作量**: 9-12 週

---

## 🎯 關鍵決策點

在開始開發前，需要決定以下問題：

### 1. 權重格式

**選項**:
- **A. PyTorch .pth** - 需要先轉為中間格式
- **B. GGUF** - 統一格式，支持量化
- **C. 自定義二進制** - 最輕量

**建議**: 選擇 **C (自定義二進制)**，初期簡單，後期可擴展支持 GGUF

### 2. 文字預處理

**選項**:
- **A. 集成在 C++ 中** - 完整解決方案
- **B. Python 預處理** - 開發快速
- **C. 提供 C API** - 靈活

**建議**: 選擇 **A (集成在 C++ 中)**，實現完全獨立

### 3. BLAS 庫

**選項**:
- **A. 僅 Eigen** - 最簡單
- **B. Eigen + OpenBLAS** - 性能提升
- **C. Eigen + Intel MKL** - Intel 平台最優

**建議**: 選擇 **A (僅 Eigen)**，初期簡單，後期可選擇性鏈接 BLAS

### 4. 依賴管理

**選項**:
- **A. vcpkg** - 跨平台，Microsoft 維護
- **B. Conan** - 功能強大
- **C. 手動** - 最大控制

**建議**: 選擇 **A (vcpkg)**，跨平台支持好

---

## ⚠️ 風險與挑戰

### 技術風險

| 風險 | 緩解措施 |
|------|----------|
| **浮點誤差累積** | 每層對比 PyTorch，逐層調試 |
| **Conv1D 實現複雜** | 參考 ONNX Runtime 實現 |
| **性能不達標** | 使用 BLAS 加速，SIMD 優化 |

### 進度風險

| 風險 | 緩解措施 |
|------|----------|
| **時間估算不準** | 採用迭代開發，MVP 優先 |
| **測試不充分** | 每個組件獨立測試 |

---

## 📈 性能目標

| 指標 | 目標 | 測試環境 |
|------|------|----------|
| **推理延遲** | < 100ms | Intel i7, 單核 |
| **內存占用** | < 200MB | 模型載入後 |
| **準確性** | MSE < 1e-4 | 與 PyTorch 對比 |
| **吞吐量** | > 10 句/秒 | 批次推理 |

---

## 🚦 下一步行動

### 立即執行 (本週)

1. ✅ **審視此規劃** - 與團隊討論技術選型
2. ✅ **環境準備** - 安裝 Eigen, CMake, vcpkg
3. ✅ **創建倉庫** - 初始化項目結構
4. ✅ **實現 Tensor** - 從核心數據結構開始

### 第一個里程碑 (2週內)

- 完成 P0 所有任務
- 能夠加載配置和權重
- 基本矩陣運算測試通過

### 第二個里程碑 (5週內)

- 完成 P1 所有任務
- Encoder 和 Decoder 可獨立運行
- 與 PyTorch 輸出對比測試

### 第三個里程碑 (9週內)

- 完成 P2 和 P3
- 端到端推理流程可運行
- 輸出精度達標

---

## 📞 需要討論的問題

1. **權重格式**: 選擇哪種格式？需要支持量化嗎？
2. **文字預處理**: 是否需要支持多語言（英文、中文）？
3. **部署場景**:
   - 是否需要 Python Binding？
   - 是否需要 C API？
   - 是否需要 GPU 支持？
4. **性能要求**:
   - 實際延遲要求是多少？
   - 是否需要批次推理？
   - 是否需要流式推理？

---

## 📚 參考資料

- [完整規劃文檔](./CPP_INFERENCE_PLAN.md) - 詳細技術設計
- [架構流程文檔](./ARCHITECTURE_FLOW.md) - FastSpeech2 原理
- [Eigen 文檔](https://eigen.tuxfamily.org/) - 線性代數庫
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) - 算子實現參考
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - 純 C++ 推理參考

---

**準備好開始了嗎？** 讓我們從任務 #1 開始！🚀
