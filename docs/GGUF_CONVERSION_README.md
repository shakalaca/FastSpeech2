# FastSpeech2 GGUF 轉換指南

## 概述

本指南說明如何將訓練好的 FastSpeech2 模型轉換為 GGUF (GPT-Generated Unified Format) 格式。

## 先決條件

### 安裝依賴

```bash
pip install gguf numpy torch pyyaml
```

### 準備文件

確保您有以下文件：
1. 訓練好的模型 checkpoint (`.pth.tar`)
2. 模型配置文件 (`model.yaml`)
3. 預處理配置文件 (`preprocess.yaml`)

## 快速開始

### 基本用法

```bash
python convert_to_gguf.py \
    --checkpoint output/ckpt/LJSpeech/900000.pth.tar \
    --model_config config/LJSpeech/model.yaml \
    --preprocess_config config/LJSpeech/preprocess.yaml \
    --output models/fastspeech2_ljspeech_f16.gguf \
    --file_type f16
```

### 參數說明

- `--checkpoint`: PyTorch checkpoint 路徑
- `--model_config`: 模型配置 YAML 文件路徑
- `--preprocess_config`: 預處理配置 YAML 文件路徑
- `--output`: 輸出 GGUF 文件路徑
- `--file_type`: 輸出數據類型
  - `f32`: 完整精度 (Float32)
  - `f16`: 半精度 (Float16) - **推薦**，節省 50% 空間

## 不同數據集的轉換示例

### LJSpeech (單說話者英文)

```bash
python convert_to_gguf.py \
    --checkpoint output/ckpt/LJSpeech/900000.pth.tar \
    --model_config config/LJSpeech/model.yaml \
    --preprocess_config config/LJSpeech/preprocess.yaml \
    --output models/fastspeech2_ljspeech.gguf \
    --file_type f16
```

### LibriTTS (多說話者英文)

```bash
python convert_to_gguf.py \
    --checkpoint output/ckpt/LibriTTS/800000.pth.tar \
    --model_config config/LibriTTS/model.yaml \
    --preprocess_config config/LibriTTS/preprocess.yaml \
    --output models/fastspeech2_libritts.gguf \
    --file_type f16
```

### AISHELL3 (多說話者中文)

```bash
python convert_to_gguf.py \
    --checkpoint output/ckpt/AISHELL3/600000.pth.tar \
    --model_config config/AISHELL3/model.yaml \
    --preprocess_config config/AISHELL3/preprocess.yaml \
    --output models/fastspeech2_aishell3.gguf \
    --file_type f16
```

## 轉換後的驗證

### 1. 檢查文件大小

```bash
ls -lh models/*.gguf
```

預期大小（F16 格式）：
- LJSpeech: ~30-40 MB
- LibriTTS: ~30-40 MB
- AISHELL3: ~30-40 MB

### 2. 使用 gguf-dump 檢查 metadata

如果安裝了 gguf 工具：

```bash
# 查看所有 metadata
gguf-dump models/fastspeech2_ljspeech.gguf --no-tensors

# 查看特定 metadata
gguf-dump models/fastspeech2_ljspeech.gguf --metadata | grep fastspeech2
```

### 3. 檢查 tensor 列表

```bash
# 列出所有 tensors
gguf-dump models/fastspeech2_ljspeech.gguf --tensors-only
```

## GGUF 文件結構

轉換後的 GGUF 文件包含：

### Metadata (約 40 個鍵值對)

```
general.*                    # 通用信息
fastspeech2.encoder.*        # Encoder 配置
fastspeech2.decoder.*        # Decoder 配置
fastspeech2.variance.*       # Variance Adaptor 配置
fastspeech2.audio.*          # 音頻處理參數
fastspeech2.postnet.*        # PostNet 配置
fastspeech2.vocab.*          # 詞彙表信息
```

### Tensors (約 186-187 個)

```
encoder.*                    # Encoder 權重 (50 tensors)
variance.*                   # Variance Adaptor 權重 (31 tensors)
decoder.*                    # Decoder 權重 (73 tensors)
mel_linear.*                 # Mel Linear 層 (2 tensors)
postnet.*                    # PostNet 權重 (30 tensors)
speaker_emb.*               # Speaker Embedding (1 tensor, 多說話者模式)
```

## GGUF vs PyTorch 模型比較

| 特性 | PyTorch (.pth.tar) | GGUF (.gguf) |
|------|-------------------|--------------|
| 文件大小 (F32) | ~60-80 MB | ~60-80 MB |
| 文件大小 (F16) | N/A | ~30-40 MB |
| 包含訓練狀態 | 是 (optimizer, scheduler) | 否 (僅權重) |
| 跨平台 | 需要 PyTorch | 獨立格式 |
| 量化支持 | 需要額外工具 | 原生支持 |
| Metadata | 有限 | 豐富 |
| 加載速度 | 慢 | 快 (mmap 支持) |

## 文件大小優化

### F32 vs F16 比較

| 格式 | 大小 | 精度損失 | 推薦用途 |
|------|------|---------|---------|
| F32 | 100% | 無 | 研究、對照實驗 |
| F16 | 50% | 極小 | **生產環境推薦** |

### F16 精度測試

F16 格式對 TTS 模型的影響通常可以忽略：
- Mel-spectrogram MSE < 1e-4
- 主觀音質無明顯差異
- 推理速度可能提升

## 常見問題

### Q1: 轉換時出現 "No mapping for XXX" 警告

**A:** 這通常是因為：
1. Checkpoint 包含額外的訓練狀態（optimizer 等）- 可以忽略
2. 模型結構與預期不符 - 檢查配置文件是否正確

### Q2: 轉換後文件很大

**A:**
- 使用 `--file_type f16` 而非 `f32`
- 檢查是否包含不必要的 BatchNorm running stats

### Q3: 如何驗證轉換正確性

**A:**
```python
# 1. 加載原始 PyTorch 模型
import torch
checkpoint = torch.load("checkpoint.pth.tar")

# 2. 檢查 tensor 數量
print(f"PyTorch tensors: {len(checkpoint['model'])}")

# 3. 檢查 GGUF 文件（需要 gguf 庫）
from gguf import GGUFReader
reader = GGUFReader("model.gguf")
print(f"GGUF tensors: {len(reader.tensors)}")
```

### Q4: 能否量化到 Q4/Q8 格式

**A:** 目前腳本僅支持 F32/F16。量化到 Q4/Q8 需要：
1. 實現量化算法
2. 測試音質影響
3. 可能需要針對 TTS 模型調整量化策略

計劃在未來版本中添加。

### Q5: GGUF 模型如何使用

**A:** 需要實現 GGUF 推理引擎，或使用：
1. **ggml** (C/C++ 推理庫)
2. **llama.cpp** 生態系統的工具（需要適配 TTS）
3. 自定義推理引擎

## 進階用法

### 自定義 Tensor 命名

修改 `convert_to_gguf.py` 中的 `_init_name_mapping()` 方法：

```python
def _init_name_mapping(self):
    self.name_map = {
        "your_pytorch_name": "your_gguf_name",
        # ...
    }
```

### 添加自定義 Metadata

修改 `add_metadata()` 方法：

```python
def add_metadata(self, writer):
    # ... 現有代碼 ...

    # 添加自定義 metadata
    writer.add_string("custom.my_field", "my_value")
    writer.add_uint32("custom.my_number", 42)
```

### 選擇性量化

修改 `_should_use_f32()` 和 `_convert_tensor()` 方法以實現：
- 對不同層使用不同精度
- 保持關鍵層為 F32
- 對大型權重矩陣使用量化

## 技術細節

### Tensor 數量統計

完整模型包含：
- **Encoder**: 50 tensors
  - Token embedding: 1
  - Positional encoding: 1
  - 4 layers × 12 tensors = 48

- **Variance Adaptor**: 31 tensors
  - Duration predictor: 9
  - Pitch predictor + bins + emb: 11
  - Energy predictor + bins + emb: 11

- **Decoder**: 73 tensors
  - Positional encoding: 1
  - 6 layers × 12 tensors = 72

- **Output Layers**: 32 tensors
  - Mel linear: 2
  - PostNet (5 layers × 6): 30

- **Speaker Embedding**: 1 tensor (僅多說話者)

**總計**: 186 (單說話者) / 187 (多說話者)

### 權重大小估算

| 層類型 | 參數量估算 | F32 大小 | F16 大小 |
|--------|-----------|---------|---------|
| Encoder | ~8M | ~32 MB | ~16 MB |
| Variance | ~2M | ~8 MB | ~4 MB |
| Decoder | ~12M | ~48 MB | ~24 MB |
| PostNet | ~3M | ~12 MB | ~6 MB |
| **總計** | **~25M** | **~100 MB** | **~50 MB** |

## 相關資源

- [GGUF 格式規範](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [gguf-py 文檔](https://github.com/ggerganov/llama.cpp/tree/master/gguf-py)
- [FastSpeech2 論文](https://arxiv.org/abs/2006.04558)
- [詳細轉換計劃](./gguf_conversion_plan.md)

## 貢獻

如果您發現問題或有改進建議，歡迎提交 Issue 或 Pull Request。

## 許可證

本轉換腳本遵循 FastSpeech2 項目的 MIT 許可證。
