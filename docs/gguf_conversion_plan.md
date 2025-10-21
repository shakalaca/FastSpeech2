# FastSpeech2 to GGUF Conversion Plan

## 概述

本文檔規劃如何將 FastSpeech2 TTS 模型轉換為 GGUF (GPT-Generated Unified Format) 格式，以實現高效存儲、量化和推理。

## GGUF 格式結構

GGUF 文件由四個主要部分組成：

```
┌─────────────────────────────────┐
│  1. Header                       │  <- Magic number, version, metadata count, tensor count
├─────────────────────────────────┤
│  2. Metadata (Key-Value pairs)  │  <- Architecture info, hyperparameters, tokenizer info
├─────────────────────────────────┤
│  3. Tensor Info                  │  <- Tensor names, dimensions, types, offsets
├─────────────────────────────────┤
│  4. Tensor Data (aligned)        │  <- Actual weight data (可量化)
└─────────────────────────────────┘
```

### 1. Header 結構

```python
# Header format (固定部分)
magic: uint32 = 0x46554747  # "GGUF" in ASCII
version: uint32 = 3          # GGUF version 3
tensor_count: uint64         # 總 tensor 數量
metadata_count: uint64       # Metadata KV 對數量
```

### 2. Metadata Key-Value Pairs

Metadata 採用 key-value 結構，每個條目包含：
- key (string): metadata 名稱
- value_type (uint32): 數據類型 (uint8=0, int8=1, uint16=2, int16=3, uint32=4, int32=5, float32=6, bool=7, string=8, array=9)
- value: 實際值

---

## FastSpeech2 GGUF Metadata 設計

### General Metadata (通用元數據)

```yaml
# 架構識別
general.architecture: "fastspeech2"
general.name: "FastSpeech2-LJSpeech-v1"
general.version: "1.0"
general.description: "FastSpeech2 Text-to-Speech Model"
general.file_type: 1  # 0=F32, 1=F16, 2=Q4_0, etc.
general.alignment: 32  # 對齊字節數

# 授權和來源
general.license: "MIT"
general.author: "ming024"
general.url: "https://github.com/ming024/FastSpeech2"
```

### Architecture Specific Metadata (架構特定元數據)

#### Encoder 配置
```yaml
fastspeech2.encoder.layers: 4
fastspeech2.encoder.hidden_size: 256
fastspeech2.encoder.attention.heads: 2
fastspeech2.encoder.attention.head_dim: 128  # hidden_size / heads
fastspeech2.encoder.ffn.hidden_size: 1024
fastspeech2.encoder.ffn.kernel_size: [9, 1]
fastspeech2.encoder.dropout: 0.2
fastspeech2.encoder.max_seq_len: 1000
```

#### Decoder 配置
```yaml
fastspeech2.decoder.layers: 6
fastspeech2.decoder.hidden_size: 256
fastspeech2.decoder.attention.heads: 2
fastspeech2.decoder.attention.head_dim: 128
fastspeech2.decoder.ffn.hidden_size: 1024
fastspeech2.decoder.ffn.kernel_size: [9, 1]
fastspeech2.decoder.dropout: 0.2
```

#### Variance Adaptor 配置
```yaml
# Duration Predictor
fastspeech2.variance.duration.filter_size: 256
fastspeech2.variance.duration.kernel_size: 3
fastspeech2.variance.duration.dropout: 0.5

# Pitch Predictor
fastspeech2.variance.pitch.filter_size: 256
fastspeech2.variance.pitch.kernel_size: 3
fastspeech2.variance.pitch.dropout: 0.5
fastspeech2.variance.pitch.quantization: "linear"
fastspeech2.variance.pitch.n_bins: 256
fastspeech2.variance.pitch.feature_level: "phoneme_level"
fastspeech2.variance.pitch.normalization: true

# Energy Predictor
fastspeech2.variance.energy.filter_size: 256
fastspeech2.variance.energy.kernel_size: 3
fastspeech2.variance.energy.dropout: 0.5
fastspeech2.variance.energy.quantization: "linear"
fastspeech2.variance.energy.n_bins: 256
fastspeech2.variance.energy.feature_level: "phoneme_level"
fastspeech2.variance.energy.normalization: true
```

#### Audio/Mel 配置
```yaml
fastspeech2.audio.sampling_rate: 22050
fastspeech2.audio.max_wav_value: 32768.0
fastspeech2.audio.n_mel_channels: 80
fastspeech2.audio.mel_fmin: 0
fastspeech2.audio.mel_fmax: 8000
fastspeech2.audio.filter_length: 1024
fastspeech2.audio.hop_length: 256
fastspeech2.audio.win_length: 1024
```

#### PostNet 配置
```yaml
fastspeech2.postnet.n_convolutions: 5
fastspeech2.postnet.embedding_dim: 512
fastspeech2.postnet.kernel_size: 5
```

#### Speaker 配置 (多說話者模式)
```yaml
fastspeech2.multi_speaker: false  # or true
fastspeech2.n_speakers: 1         # 說話者數量
```

#### Tokenizer/Vocabulary 配置
```yaml
fastspeech2.vocab.size: 76        # len(symbols) + 1
fastspeech2.vocab.pad_token_id: 0
fastspeech2.vocab.language: "en"
```

#### Training Info (可選)
```yaml
fastspeech2.training.total_steps: 900000
fastspeech2.training.batch_size: 16
fastspeech2.training.optimizer: "Adam"
fastspeech2.training.learning_rate_schedule: "noam"
fastspeech2.training.warmup_steps: 4000
```

---

## FastSpeech2 GGUF Tensor 映射設計

### Tensor 命名規範

採用層級化命名，格式：`{component}.{layer_idx}.{sub_component}.{param_type}`

### 完整 Tensor 列表

#### 1. Encoder Tensors

```
# Embedding Layer
encoder.token_emb.weight              [vocab_size, hidden_size] = [76, 256]

# Positional Encoding (如果保存)
encoder.pos_enc.weight                [max_seq_len+1, hidden_size] = [1001, 256]

# Encoder Layers (4 layers)
encoder.layers.{N}.attn.q_proj.weight      [hidden_size, hidden_size] = [256, 256]
encoder.layers.{N}.attn.q_proj.bias        [hidden_size] = [256]
encoder.layers.{N}.attn.k_proj.weight      [256, 256]
encoder.layers.{N}.attn.k_proj.bias        [256]
encoder.layers.{N}.attn.v_proj.weight      [256, 256]
encoder.layers.{N}.attn.v_proj.bias        [256]
encoder.layers.{N}.attn.out_proj.weight    [256, 256]
encoder.layers.{N}.attn.out_proj.bias      [256]
encoder.layers.{N}.attn.layer_norm.weight  [256]
encoder.layers.{N}.attn.layer_norm.bias    [256]

encoder.layers.{N}.ffn.conv1.weight        [1024, 256, 9]
encoder.layers.{N}.ffn.conv1.bias          [1024]
encoder.layers.{N}.ffn.conv2.weight        [256, 1024, 1]
encoder.layers.{N}.ffn.conv2.bias          [256]
encoder.layers.{N}.ffn.layer_norm.weight   [256]
encoder.layers.{N}.ffn.layer_norm.bias     [256]
```
N ∈ {0, 1, 2, 3} (4 layers)

#### 2. Variance Adaptor Tensors

```
# Duration Predictor
variance.duration.conv.0.weight            [256, 256, 3]
variance.duration.conv.0.bias              [256]
variance.duration.norm.0.weight            [256]
variance.duration.norm.0.bias              [256]
variance.duration.conv.1.weight            [256, 256, 3]
variance.duration.conv.1.bias              [256]
variance.duration.norm.1.weight            [256]
variance.duration.norm.1.bias              [256]
variance.duration.linear.weight            [1, 256]
variance.duration.linear.bias              [1]

# Pitch Predictor (相同結構)
variance.pitch.conv.0.weight               [256, 256, 3]
variance.pitch.conv.0.bias                 [256]
variance.pitch.norm.0.weight               [256]
variance.pitch.norm.0.bias                 [256]
variance.pitch.conv.1.weight               [256, 256, 3]
variance.pitch.conv.1.bias                 [256]
variance.pitch.norm.1.weight               [256]
variance.pitch.norm.1.bias                 [256]
variance.pitch.linear.weight               [1, 256]
variance.pitch.linear.bias                 [1]

# Pitch Embedding
variance.pitch.bins                        [255]  # n_bins - 1
variance.pitch.emb.weight                  [256, 256]  # [n_bins, hidden_size]

# Energy Predictor (相同結構)
variance.energy.conv.0.weight              [256, 256, 3]
variance.energy.conv.0.bias                [256]
variance.energy.norm.0.weight              [256]
variance.energy.norm.0.bias                [256]
variance.energy.conv.1.weight              [256, 256, 3]
variance.energy.conv.1.bias                [256]
variance.energy.norm.1.weight              [256]
variance.energy.norm.1.bias                [256]
variance.energy.linear.weight              [1, 256]
variance.energy.linear.bias                [1]

# Energy Embedding
variance.energy.bins                       [255]
variance.energy.emb.weight                 [256, 256]
```

#### 3. Decoder Tensors

```
# Positional Encoding
decoder.pos_enc.weight                     [1001, 256]

# Decoder Layers (6 layers)
decoder.layers.{N}.attn.q_proj.weight      [256, 256]
decoder.layers.{N}.attn.q_proj.bias        [256]
decoder.layers.{N}.attn.k_proj.weight      [256, 256]
decoder.layers.{N}.attn.k_proj.bias        [256]
decoder.layers.{N}.attn.v_proj.weight      [256, 256]
decoder.layers.{N}.attn.v_proj.bias        [256]
decoder.layers.{N}.attn.out_proj.weight    [256, 256]
decoder.layers.{N}.attn.out_proj.bias      [256]
decoder.layers.{N}.attn.layer_norm.weight  [256]
decoder.layers.{N}.attn.layer_norm.bias    [256]

decoder.layers.{N}.ffn.conv1.weight        [1024, 256, 9]
decoder.layers.{N}.ffn.conv1.bias          [1024]
decoder.layers.{N}.ffn.conv2.weight        [256, 1024, 1]
decoder.layers.{N}.ffn.conv2.bias          [256]
decoder.layers.{N}.ffn.layer_norm.weight   [256]
decoder.layers.{N}.ffn.layer_norm.bias     [256]
```
N ∈ {0, 1, 2, 3, 4, 5} (6 layers)

#### 4. Mel Linear Layer

```
mel_linear.weight                          [80, 256]
mel_linear.bias                            [80]
```

#### 5. PostNet Tensors

```
postnet.convs.0.conv.weight                [512, 80, 5]
postnet.convs.0.conv.bias                  [512]
postnet.convs.0.bn.weight                  [512]
postnet.convs.0.bn.bias                    [512]
postnet.convs.0.bn.running_mean            [512]
postnet.convs.0.bn.running_var             [512]

postnet.convs.1.conv.weight                [512, 512, 5]
postnet.convs.1.conv.bias                  [512]
postnet.convs.1.bn.weight                  [512]
postnet.convs.1.bn.bias                    [512]
postnet.convs.1.bn.running_mean            [512]
postnet.convs.1.bn.running_var             [512]

postnet.convs.2.conv.weight                [512, 512, 5]
postnet.convs.2.conv.bias                  [512]
postnet.convs.2.bn.weight                  [512]
postnet.convs.2.bn.bias                    [512]
postnet.convs.2.bn.running_mean            [512]
postnet.convs.2.bn.running_var             [512]

postnet.convs.3.conv.weight                [512, 512, 5]
postnet.convs.3.conv.bias                  [512]
postnet.convs.3.bn.weight                  [512]
postnet.convs.3.bn.bias                    [512]
postnet.convs.3.bn.running_mean            [512]
postnet.convs.3.bn.running_var             [512]

postnet.convs.4.conv.weight                [80, 512, 5]
postnet.convs.4.conv.bias                  [80]
postnet.convs.4.bn.weight                  [80]
postnet.convs.4.bn.bias                    [80]
postnet.convs.4.bn.running_mean            [80]
postnet.convs.4.bn.running_var             [80]
```

#### 6. Speaker Embedding (多說話者模式)

```
speaker_emb.weight                         [n_speakers, 256]
```

---

## Tensor 數量統計

### 按組件統計：

1. **Encoder**:
   - Token Embedding: 1 tensor
   - Positional Encoding: 1 tensor
   - 4 Layers × 12 tensors/layer = 48 tensors
   - **Total: 50 tensors**

2. **Variance Adaptor**:
   - Duration Predictor: 9 tensors
   - Pitch Predictor: 9 tensors + 2 (bins + emb) = 11 tensors
   - Energy Predictor: 9 tensors + 2 (bins + emb) = 11 tensors
   - **Total: 31 tensors**

3. **Decoder**:
   - Positional Encoding: 1 tensor
   - 6 Layers × 12 tensors/layer = 72 tensors
   - **Total: 73 tensors**

4. **Mel Linear**: 2 tensors (weight + bias)

5. **PostNet**:
   - 5 Conv layers × 6 tensors/layer = 30 tensors
   - **Total: 30 tensors**

6. **Speaker Embedding** (可選): 1 tensor

**總計: 186 tensors** (單說話者模式) / **187 tensors** (多說話者模式)

---

## Tensor Info 結構

每個 tensor 在 Tensor Info 部分的條目包含：

```python
class TensorInfo:
    name: string           # Tensor 名稱
    n_dimensions: uint32   # 維度數量
    dimensions: uint64[]   # 各維度大小
    type: uint32          # 數據類型 (F32=0, F16=1, Q4_0=2, Q4_1=3, Q5_0=6, Q5_1=7, Q8_0=8, etc.)
    offset: uint64        # 在 Tensor Data 部分的偏移量
```

---

## 數據類型與量化策略

### GGUF 支持的數據類型

```python
GGML_TYPE_F32  = 0   # float32
GGML_TYPE_F16  = 1   # float16
GGML_TYPE_Q4_0 = 2   # 4-bit quantization
GGML_TYPE_Q4_1 = 3   # 4-bit quantization
GGML_TYPE_Q5_0 = 6   # 5-bit quantization
GGML_TYPE_Q5_1 = 7   # 5-bit quantization
GGML_TYPE_Q8_0 = 8   # 8-bit quantization
GGML_TYPE_Q8_1 = 9   # 8-bit quantization
# ... 更多量化類型
```

### FastSpeech2 量化建議

不同組件對精度的敏感度不同，建議分層量化：

| 組件 | 推薦類型 | 理由 |
|------|---------|------|
| **Encoder/Decoder Attention** | Q8_0 or F16 | 注意力機制對精度較敏感 |
| **FFN Weights** | Q4_0/Q4_1 | 前饋網路可容忍較低精度 |
| **Layer Norms** | F32 or F16 | 歸一化需要較高精度 |
| **Embeddings** | F16 or Q8_0 | 嵌入層保持較高精度 |
| **Variance Predictors** | Q8_0 or F16 | Pitch/Energy/Duration 預測需要精度 |
| **Pitch/Energy Bins** | F32 | 量化邊界必須高精度 |
| **PostNet** | Q4_0/Q8_0 | 後處理網路可量化 |
| **Mel Linear** | F16 or Q8_0 | 輸出層保持精度 |

### 混合精度配置示例

```yaml
# 完整精度模型 (F32)
general.file_type: 0

# 半精度模型 (F16)
general.file_type: 1

# 混合量化模型 (推薦)
general.file_type: 10  # Mixed quantization
```

---

## Tensor Data 存儲

### 對齊要求

- Tensor data 部分需要對齊到 `general.alignment` 字節邊界 (通常是 32 字節)
- 每個 tensor 的數據按照其在 Tensor Info 中聲明的順序存儲
- Tensor 之間可能需要填充 0x00 以滿足對齊要求

### 數據佈局

```
Offset 0: encoder.token_emb.weight (對齊到 32 字節)
Offset X: encoder.pos_enc.weight (對齊到 32 字節)
Offset Y: encoder.layers.0.attn.q_proj.weight (對齊到 32 字節)
...
```

---

## PyTorch to GGUF 權重映射

### PyTorch 模型結構 → GGUF Tensor 名稱映射

```python
PYTORCH_TO_GGUF_MAPPING = {
    # Encoder
    "encoder.src_word_emb.weight": "encoder.token_emb.weight",
    "encoder.position_enc": "encoder.pos_enc.weight",

    # Encoder Layers
    "encoder.layer_stack.{N}.slf_attn.w_qs.weight": "encoder.layers.{N}.attn.q_proj.weight",
    "encoder.layer_stack.{N}.slf_attn.w_qs.bias": "encoder.layers.{N}.attn.q_proj.bias",
    "encoder.layer_stack.{N}.slf_attn.w_ks.weight": "encoder.layers.{N}.attn.k_proj.weight",
    "encoder.layer_stack.{N}.slf_attn.w_ks.bias": "encoder.layers.{N}.attn.k_proj.bias",
    "encoder.layer_stack.{N}.slf_attn.w_vs.weight": "encoder.layers.{N}.attn.v_proj.weight",
    "encoder.layer_stack.{N}.slf_attn.w_vs.bias": "encoder.layers.{N}.attn.v_proj.bias",
    "encoder.layer_stack.{N}.slf_attn.fc.weight": "encoder.layers.{N}.attn.out_proj.weight",
    "encoder.layer_stack.{N}.slf_attn.fc.bias": "encoder.layers.{N}.attn.out_proj.bias",
    "encoder.layer_stack.{N}.slf_attn.layer_norm.weight": "encoder.layers.{N}.attn.layer_norm.weight",
    "encoder.layer_stack.{N}.slf_attn.layer_norm.bias": "encoder.layers.{N}.attn.layer_norm.bias",

    "encoder.layer_stack.{N}.pos_ffn.w_1.weight": "encoder.layers.{N}.ffn.conv1.weight",
    "encoder.layer_stack.{N}.pos_ffn.w_1.bias": "encoder.layers.{N}.ffn.conv1.bias",
    "encoder.layer_stack.{N}.pos_ffn.w_2.weight": "encoder.layers.{N}.ffn.conv2.weight",
    "encoder.layer_stack.{N}.pos_ffn.w_2.bias": "encoder.layers.{N}.ffn.conv2.bias",
    "encoder.layer_stack.{N}.pos_ffn.layer_norm.weight": "encoder.layers.{N}.ffn.layer_norm.weight",
    "encoder.layer_stack.{N}.pos_ffn.layer_norm.bias": "encoder.layers.{N}.ffn.layer_norm.bias",

    # Variance Adaptor - Duration
    "variance_adaptor.duration_predictor.conv_layer.conv1d_1.conv.weight": "variance.duration.conv.0.weight",
    "variance_adaptor.duration_predictor.conv_layer.conv1d_1.conv.bias": "variance.duration.conv.0.bias",
    "variance_adaptor.duration_predictor.conv_layer.layer_norm_1.weight": "variance.duration.norm.0.weight",
    "variance_adaptor.duration_predictor.conv_layer.layer_norm_1.bias": "variance.duration.norm.0.bias",
    "variance_adaptor.duration_predictor.conv_layer.conv1d_2.conv.weight": "variance.duration.conv.1.weight",
    "variance_adaptor.duration_predictor.conv_layer.conv1d_2.conv.bias": "variance.duration.conv.1.bias",
    "variance_adaptor.duration_predictor.conv_layer.layer_norm_2.weight": "variance.duration.norm.1.weight",
    "variance_adaptor.duration_predictor.conv_layer.layer_norm_2.bias": "variance.duration.norm.1.bias",
    "variance_adaptor.duration_predictor.linear_layer.weight": "variance.duration.linear.weight",
    "variance_adaptor.duration_predictor.linear_layer.bias": "variance.duration.linear.bias",

    # Variance Adaptor - Pitch
    "variance_adaptor.pitch_predictor.conv_layer.conv1d_1.conv.weight": "variance.pitch.conv.0.weight",
    "variance_adaptor.pitch_predictor.conv_layer.conv1d_1.conv.bias": "variance.pitch.conv.0.bias",
    "variance_adaptor.pitch_predictor.conv_layer.layer_norm_1.weight": "variance.pitch.norm.0.weight",
    "variance_adaptor.pitch_predictor.conv_layer.layer_norm_1.bias": "variance.pitch.norm.0.bias",
    "variance_adaptor.pitch_predictor.conv_layer.conv1d_2.conv.weight": "variance.pitch.conv.1.weight",
    "variance_adaptor.pitch_predictor.conv_layer.conv1d_2.conv.bias": "variance.pitch.conv.1.bias",
    "variance_adaptor.pitch_predictor.conv_layer.layer_norm_2.weight": "variance.pitch.norm.1.weight",
    "variance_adaptor.pitch_predictor.conv_layer.layer_norm_2.bias": "variance.pitch.norm.1.bias",
    "variance_adaptor.pitch_predictor.linear_layer.weight": "variance.pitch.linear.weight",
    "variance_adaptor.pitch_predictor.linear_layer.bias": "variance.pitch.linear.bias",
    "variance_adaptor.pitch_bins": "variance.pitch.bins",
    "variance_adaptor.pitch_embedding.weight": "variance.pitch.emb.weight",

    # Variance Adaptor - Energy (同 Pitch 結構)
    # ... (省略，結構相同)

    # Decoder
    "decoder.position_enc": "decoder.pos_enc.weight",
    # Decoder layers (同 Encoder 映射模式)

    # Mel Linear
    "mel_linear.weight": "mel_linear.weight",
    "mel_linear.bias": "mel_linear.bias",

    # PostNet
    "postnet.convolutions.{N}.0.conv.weight": "postnet.convs.{N}.conv.weight",
    "postnet.convolutions.{N}.0.conv.bias": "postnet.convs.{N}.conv.bias",
    "postnet.convolutions.{N}.1.weight": "postnet.convs.{N}.bn.weight",
    "postnet.convolutions.{N}.1.bias": "postnet.convs.{N}.bn.bias",
    "postnet.convolutions.{N}.1.running_mean": "postnet.convs.{N}.bn.running_mean",
    "postnet.convolutions.{N}.1.running_var": "postnet.convs.{N}.bn.running_var",

    # Speaker Embedding
    "speaker_emb.weight": "speaker_emb.weight",
}
```

---

## 轉換實現建議

### 1. 讀取 PyTorch 模型

```python
import torch

# Load checkpoint
checkpoint = torch.load("checkpoint.pth", map_location="cpu")
model_state = checkpoint["model"]  # or checkpoint depending on save format
```

### 2. 讀取配置文件

```python
import yaml

with open("config/LJSpeech/model.yaml") as f:
    model_config = yaml.safe_load(f)

with open("config/LJSpeech/preprocess.yaml") as f:
    preprocess_config = yaml.safe_load(f)
```

### 3. 構建 GGUF Metadata

```python
metadata = {
    "general.architecture": "fastspeech2",
    "general.name": "FastSpeech2-LJSpeech",
    "general.file_type": 1,  # F16
    "general.alignment": 32,

    # 從配置文件填充
    "fastspeech2.encoder.layers": model_config["transformer"]["encoder_layer"],
    "fastspeech2.encoder.hidden_size": model_config["transformer"]["encoder_hidden"],
    # ... 等等
}
```

### 4. 映射和轉換權重

```python
gguf_tensors = {}

for pytorch_name, tensor in model_state.items():
    # 映射名稱
    gguf_name = map_pytorch_to_gguf(pytorch_name)

    # 轉換數據類型 (如 F32 -> F16)
    if should_convert_to_f16(gguf_name):
        tensor = tensor.half()
    elif should_quantize(gguf_name):
        tensor = quantize_tensor(tensor, quant_type)

    gguf_tensors[gguf_name] = tensor.numpy()
```

### 5. 寫入 GGUF 文件

```python
import struct

with open("fastspeech2.gguf", "wb") as f:
    # Write header
    write_header(f, len(metadata), len(gguf_tensors))

    # Write metadata
    write_metadata(f, metadata)

    # Write tensor info
    tensor_offsets = write_tensor_info(f, gguf_tensors)

    # Align to alignment boundary
    align_to(f, 32)

    # Write tensor data
    write_tensor_data(f, gguf_tensors, tensor_offsets)
```

---

## 特殊考慮

### 1. Positional Encoding

PyTorch 模型中的 `position_enc` 是 `nn.Parameter` 且 `requires_grad=False`，在 GGUF 中可以：
- **選項 A**: 保存為常規 tensor
- **選項 B**: 在推理時動態計算（節省空間）

建議：保存為 tensor，簡化推理實現。

### 2. BatchNorm Running Stats

PostNet 的 BatchNorm 層包含 `running_mean` 和 `running_var`，這些在推理時需要，必須包含在 GGUF 中。

### 3. Pitch/Energy Bins

這些是用於量化的邊界值，必須保持 F32 精度，不應量化。

### 4. 多說話者支持

如果 `model_config["multi_speaker"] = True`，需要：
- 添加 `speaker_emb.weight` tensor
- 設置 `fastspeech2.multi_speaker = true`
- 設置 `fastspeech2.n_speakers` 為實際說話者數量

---

## 驗證和測試

轉換後應驗證：

1. **Tensor 數量**: 確認所有權重都已轉換
2. **Tensor 形狀**: 檢查維度是否正確
3. **數值精度**: 比較量化前後的輸出差異
4. **推理測試**: 使用相同輸入，比較 PyTorch 和 GGUF 模型的輸出

### 建議測試流程

```python
# 1. 加載原始 PyTorch 模型
pytorch_model = FastSpeech2(preprocess_config, model_config)
pytorch_model.load_state_dict(torch.load("checkpoint.pth"))
pytorch_model.eval()

# 2. 準備測試輸入
test_input = prepare_test_input()

# 3. PyTorch 推理
with torch.no_grad():
    pytorch_output = pytorch_model(**test_input)

# 4. GGUF 推理 (需要實現 GGUF 推理引擎)
gguf_output = gguf_inference("fastspeech2.gguf", test_input)

# 5. 比較輸出
mse = np.mean((pytorch_output - gguf_output) ** 2)
print(f"MSE: {mse}")
```

---

## 工具和庫

### 推薦使用的庫

1. **gguf-py**: 官方 Python GGUF 寫入庫
   ```bash
   pip install gguf
   ```

2. **numpy**: 數據處理
3. **PyTorch**: 加載原始模型
4. **PyYAML**: 讀取配置

### 示例代碼框架

```python
from gguf import GGUFWriter
import torch
import yaml
import numpy as np

def convert_fastspeech2_to_gguf(
    checkpoint_path: str,
    model_config_path: str,
    preprocess_config_path: str,
    output_path: str,
    file_type: str = "f16"
):
    # 1. Load configs
    model_config = yaml.safe_load(open(model_config_path))
    preprocess_config = yaml.safe_load(open(preprocess_config_path))

    # 2. Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"]

    # 3. Create GGUF writer
    writer = GGUFWriter(output_path, "fastspeech2")

    # 4. Add metadata
    add_metadata(writer, model_config, preprocess_config)

    # 5. Add tensors
    add_tensors(writer, state_dict, file_type)

    # 6. Write file
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()

    print(f"Conversion complete: {output_path}")

def add_metadata(writer, model_config, preprocess_config):
    writer.add_string("general.architecture", "fastspeech2")
    writer.add_uint32("fastspeech2.encoder.layers",
                      model_config["transformer"]["encoder_layer"])
    # ... add all metadata

def add_tensors(writer, state_dict, file_type):
    for pytorch_name, tensor in state_dict.items():
        gguf_name = map_name(pytorch_name)
        tensor_np = convert_tensor(tensor, file_type)
        writer.add_tensor(gguf_name, tensor_np)
```

---

## 總結

將 FastSpeech2 轉換為 GGUF 格式需要：

1. **187 個 tensors** (包含所有權重、偏置、BatchNorm 參數)
2. **~40 個 metadata entries** (架構配置、超參數、音頻參數)
3. **混合精度量化策略** (不同組件使用不同精度)
4. **完整的名稱映射** (PyTorch 到 GGUF)

主要優勢：
- ✅ 統一的模型格式
- ✅ 支持量化壓縮
- ✅ 高效的推理加載
- ✅ 跨平台兼容性
- ✅ 豐富的元數據支持

主要挑戰：
- ⚠️ 需要實現 GGUF 推理引擎（或使用 ggml）
- ⚠️ 量化可能影響音質，需要仔細調優
- ⚠️ TTS 模型與 LLM 推理模式不同，需要適配
