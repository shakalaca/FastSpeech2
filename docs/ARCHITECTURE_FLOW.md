# FastSpeech2 架構流程與資料結構詳解

本文檔詳細解說 FastSpeech2 的運作流程和資料在各階段的結構變化。

---

## 目錄

1. [整體架構概覽](#整體架構概覽)
2. [訓練流程](#訓練流程)
3. [推理流程](#推理流程)
4. [資料結構詳解](#資料結構詳解)
5. [張量形狀追蹤](#張量形狀追蹤)
6. [核心組件詳解](#核心組件詳解)

---

## 整體架構概覽

### 高層級流程圖

```
訓練階段：
文字 → 音素序列 → [FastSpeech2 模型] → Mel-Spectrogram → 損失計算
                                ↑
                    Ground Truth (Duration, Pitch, Energy, Mel)

推理階段：
文字 → 音素序列 → [FastSpeech2 模型] → Mel-Spectrogram → Vocoder → 音訊波形
```

### FastSpeech2 模型內部流程

```
音素序列 (Phoneme IDs)
    ↓
┌───────────────────────────────────────────────────────────┐
│ 1. Encoder                                                 │
│    - Token Embedding                                       │
│    - Positional Encoding                                   │
│    - 4 × FFT Blocks (Self-Attention + Feed-Forward)       │
│    - (可選) Speaker Embedding 加入                         │
├───────────────────────────────────────────────────────────┤
│ 2. Variance Adaptor                                        │
│    ┌─────────────────────────────────────────┐            │
│    │ Duration Predictor → 預測每個音素持續時間 │            │
│    └─────────────────────────────────────────┘            │
│    ┌─────────────────────────────────────────┐            │
│    │ Pitch Predictor → 預測音高 (F0)          │            │
│    │ Pitch Embedding → 將 pitch 嵌入特徵      │            │
│    └─────────────────────────────────────────┘            │
│    ┌─────────────────────────────────────────┐            │
│    │ Energy Predictor → 預測能量              │            │
│    │ Energy Embedding → 將 energy 嵌入特徵    │            │
│    └─────────────────────────────────────────┘            │
│    ┌─────────────────────────────────────────┐            │
│    │ Length Regulator → 根據 duration 擴展序列│            │
│    └─────────────────────────────────────────┘            │
├───────────────────────────────────────────────────────────┤
│ 3. Decoder                                                 │
│    - Positional Encoding                                   │
│    - 6 × FFT Blocks (Self-Attention + Feed-Forward)       │
├───────────────────────────────────────────────────────────┤
│ 4. Mel Linear                                              │
│    - 線性層將 decoder 輸出投影到 mel 維度                   │
├───────────────────────────────────────────────────────────┤
│ 5. PostNet                                                 │
│    - 5 × Conv1D + BatchNorm (Tacotron2 風格)              │
│    - Residual connection                                   │
└───────────────────────────────────────────────────────────┘
    ↓
Mel-Spectrogram (最終輸出)
```

---

## 訓練流程

### 完整訓練步驟 (train.py)

```python
# 步驟 1: 載入資料
dataset = Dataset("train.txt", preprocess_config, train_config)
# 輸出: 包含預處理好的音素、mel、pitch、energy、duration

# 步驟 2: Batch 處理
batch = {
    'ids': [...],           # 樣本 ID
    'raw_texts': [...],     # 原始文字
    'speakers': [B],        # 說話者 ID, shape: (B,)
    'texts': [B, T_text],   # 音素序列, shape: (B, max_text_len)
    'text_lens': [B],       # 每個序列的實際長度
    'max_text_len': int,    # Batch 中最長的文字序列長度
    'mels': [B, T_mel, 80], # Ground truth mel, shape: (B, max_mel_len, 80)
    'mel_lens': [B],        # 每個 mel 的實際長度
    'max_mel_len': int,     # Batch 中最長的 mel 長度
    'pitches': [B, T],      # Ground truth pitch
    'energies': [B, T],     # Ground truth energy
    'durations': [B, T_text] # Ground truth duration (每個音素對應多少幀)
}

# 步驟 3: 前向傳播
output = model(
    speakers=batch['speakers'],      # (B,)
    texts=batch['texts'],            # (B, T_text)
    src_lens=batch['text_lens'],     # (B,)
    max_src_len=batch['max_text_len'],
    mels=batch['mels'],              # (B, T_mel, 80) - 訓練時提供
    mel_lens=batch['mel_lens'],      # (B,)
    max_mel_len=batch['max_mel_len'],
    p_targets=batch['pitches'],      # (B, T_text) - 訓練時提供
    e_targets=batch['energies'],     # (B, T_text) - 訓練時提供
    d_targets=batch['durations'],    # (B, T_text) - 訓練時提供
)

# 步驟 4: 模型輸出
output = (
    mel_predictions,        # (B, T_mel, 80) - decoder 輸出
    postnet_mel_predictions,# (B, T_mel, 80) - postnet 輸出
    pitch_predictions,      # (B, T_text) - 預測的 pitch
    energy_predictions,     # (B, T_text) - 預測的 energy
    log_duration_predictions, # (B, T_text) - 預測的 log(duration+1)
    duration_rounded,       # (B, T_text) - 四捨五入的 duration
    src_masks,              # (B, T_text) - 文字的 mask
    mel_masks,              # (B, T_mel) - mel 的 mask
    src_lens,               # (B,) - 文字序列長度
    mel_lens,               # (B,) - mel 序列長度
)

# 步驟 5: 計算損失
losses = Loss(batch, output)
# 返回: (total_loss, mel_loss, postnet_mel_loss, pitch_loss, energy_loss, duration_loss)

# 步驟 6: 反向傳播
total_loss.backward()
optimizer.step()
```

### 損失函數詳解 (model/loss.py)

FastSpeech2 使用**多個損失函數的組合**：

```python
# 1. Mel-Spectrogram 損失 (L1 Loss / MAE)
mel_loss = L1(mel_predictions, mel_targets)
postnet_mel_loss = L1(postnet_mel_predictions, mel_targets)

# 2. Pitch 損失 (MSE)
pitch_loss = MSE(pitch_predictions, pitch_targets)

# 3. Energy 損失 (MSE)
energy_loss = MSE(energy_predictions, energy_targets)

# 4. Duration 損失 (MSE on log domain)
log_duration_loss = MSE(log(duration_predictions + 1), log(duration_targets + 1))

# 總損失 (所有項等權重相加)
total_loss = mel_loss + postnet_mel_loss + pitch_loss + energy_loss + duration_loss
```

**為何使用 log domain 的 duration loss？**
- Duration 值範圍很大（1 到數十幀）
- Log 空間讓小值和大值的誤差更平衡
- 例如：duration=1 vs 2 的差異 > duration=20 vs 21

---

## 推理流程

### 推理步驟 (synthesize.py)

```python
# 步驟 1: 文字預處理
text = "Hello, how are you?"

# 英文: 文字 → 音素
# 使用 G2P (Grapheme-to-Phoneme) 和 Lexicon
phonemes = preprocess_english(text, config)
# 輸出: "{HH}{AH0}{L}{OW1} {HH}{AW1} {AA1}{R} {Y}{UW1}"
# 轉換為 ID: [23, 15, 8, 32, 0, 23, 16, 0, 5, 21, 0, ...]

# 中文: 文字 → 拼音 → 音素
phonemes = preprocess_mandarin("你好", config)
# 輸出: "{n}{i3} {h}{ao3}" -> ID sequence

# 步驟 2: 準備輸入
batch = {
    'speakers': [speaker_id],       # (1,)
    'texts': [phoneme_ids],         # (1, T_text)
    'text_lens': [len(phoneme_ids)],# (1,)
    'max_text_len': len(phoneme_ids),
    # 推理時不提供以下項 (設為 None)
    'mels': None,
    'mel_lens': None,
    'max_mel_len': None,
    'p_targets': None,  # 模型會自己預測
    'e_targets': None,  # 模型會自己預測
    'd_targets': None,  # 模型會自己預測
}

# 步驟 3: 前向傳播 (推理模式)
with torch.no_grad():
    output = model(
        speakers=batch['speakers'],
        texts=batch['texts'],
        src_lens=batch['text_lens'],
        max_src_len=batch['max_text_len'],
        # 推理時可以控制 pitch/energy/duration
        p_control=1.0,   # >1: 提高音調, <1: 降低音調
        e_control=1.0,   # >1: 增加音量, <1: 降低音量
        d_control=1.0,   # >1: 放慢語速, <1: 加快語速
    )

# 步驟 4: 提取 mel-spectrogram
mel_output = output[1]  # postnet_mel_predictions
# Shape: (1, T_mel, 80)

# 步驟 5: Vocoder 生成音訊
audio = vocoder.infer(mel_output)
# 使用 HiFi-GAN 或 MelGAN 將 mel 轉換為波形
# 輸出: (1, T_audio) - T_audio = T_mel * hop_length
```

### Variance Predictor 在推理時的行為

```python
# 訓練時 (有 ground truth targets)
if p_targets is not None:
    # 使用真實值
    pitch_embedding = PitchEmbedding(quantize(p_targets))
else:
    # 推理時 (沒有 targets)
    pitch_prediction = PitchPredictor(encoder_output)
    # 可以應用控制
    pitch_prediction = pitch_prediction * p_control
    pitch_embedding = PitchEmbedding(quantize(pitch_prediction))

# Duration, Energy 同理
```

---

## 資料結構詳解

### 預處理階段的資料結構 (dataset.py)

#### 訓練資料格式 (train.txt / val.txt)

```
LJ001-0001|LJSpeech|{DH}{AH0} {T}{EH1}{K}{S}{T}...|The text content...
LJ001-0002|LJSpeech|{P}{R}{IH1}{N}{T}{IH0}{NG}...|Printing in the...
...
```

格式: `basename|speaker|phoneme_sequence|raw_text`

#### Dataset 返回的 Sample

```python
sample = {
    "id": "LJ001-0001",              # basename
    "speaker": 0,                     # speaker_id (int)
    "text": [23, 15, 8, 32, ...],    # phoneme IDs, shape: (T_text,)
    "raw_text": "The text content",   # 原始文字
    "mel": np.array(...),             # shape: (T_mel, 80)
    "pitch": np.array(...),           # shape: (T_text,) - phoneme level
    "energy": np.array(...),          # shape: (T_text,) - phoneme level
    "duration": np.array([3,2,4,...]),# shape: (T_text,) - 每個音素對應的幀數
}
```

**重要**: `sum(duration) = T_mel`（所有音素的持續時間總和 = mel 幀數）

#### Collate Function 輸出的 Batch

```python
batch = (
    ids,           # List[str], length B - ['LJ001-0001', 'LJ001-0002', ...]
    raw_texts,     # List[str], length B
    speakers,      # np.array, shape (B,)
    texts,         # np.array, shape (B, max_T_text) - padded
    text_lens,     # np.array, shape (B,)
    max_text_len,  # int
    mels,          # np.array, shape (B, max_T_mel, 80) - padded
    mel_lens,      # np.array, shape (B,)
    max_mel_len,   # int
    pitches,       # np.array, shape (B, max_T_text) - padded
    energies,      # np.array, shape (B, max_T_text) - padded
    durations,     # np.array, shape (B, max_T_text) - padded
)
```

---

## 張量形狀追蹤

### 完整的前向傳播張量形狀

假設配置：
- `B = 16` (batch size)
- `T_text = 50` (音素序列長度)
- `T_mel = 200` (mel 幀數，約 T_text × 平均 duration)
- `hidden_size = 256`
- `n_mel_channels = 80`

```python
# ===== 輸入 =====
speakers:     (B,) = (16,)
texts:        (B, T_text) = (16, 50)
src_lens:     (B,) = (16,)
p_targets:    (B, T_text) = (16, 50)  # phoneme-level
e_targets:    (B, T_text) = (16, 50)  # phoneme-level
d_targets:    (B, T_text) = (16, 50)
mels:         (B, T_mel, 80) = (16, 200, 80)

# ===== Encoder =====
# 1. Token Embedding
token_emb:    (16, 50, 256)

# 2. Positional Encoding
pos_enc:      (1, 1001, 256) -> broadcast to (16, 50, 256)
encoder_input: (16, 50, 256) = token_emb + pos_enc

# 3. (可選) Speaker Embedding
speaker_emb:  (1, 256) -> broadcast to (16, 50, 256)
encoder_input: (16, 50, 256) = encoder_input + speaker_emb

# 4. Encoder Layers (4 layers)
# 每一層:
#   Self-Attention: (16, 50, 256) -> (16, 50, 256)
#   Feed-Forward:   (16, 50, 256) -> (16, 50, 256)
encoder_output: (16, 50, 256)

# ===== Variance Adaptor =====
# 1. Duration Predictor
duration_pred: (16, 50) - 每個音素的預測持續幀數

# 2. Pitch Predictor & Embedding (phoneme-level)
pitch_pred:    (16, 50)
pitch_emb:     (16, 50, 256)
encoder_output: (16, 50, 256) + pitch_emb

# 3. Energy Predictor & Embedding (phoneme-level)
energy_pred:   (16, 50)
energy_emb:    (16, 50, 256)
encoder_output: (16, 50, 256) + energy_emb

# 4. Length Regulator (根據 duration 擴展)
# Input:  (16, 50, 256), duration: (16, 50)
# Output: (16, T_mel, 256) = (16, 200, 256)
# 原理: 每個音素重複 duration[i] 次
regulated_output: (16, 200, 256)

# ===== Decoder =====
# 1. Positional Encoding
decoder_input: (16, 200, 256) = regulated_output + pos_enc

# 2. Decoder Layers (6 layers)
decoder_output: (16, 200, 256)

# ===== Mel Linear =====
mel_output: (16, 200, 80) = Linear(decoder_output)

# ===== PostNet =====
# Input: (16, 200, 80)
# 5 × Conv1D (transpose to (16, 80, 200) for Conv1D)
postnet_output: (16, 200, 80)

# Final mel (residual connection)
final_mel: (16, 200, 80) = mel_output + postnet_output

# ===== 輸出 =====
return (
    mel_output,              # (16, 200, 80)
    final_mel,               # (16, 200, 80)
    pitch_pred,              # (16, 50)
    energy_pred,             # (16, 50)
    log_duration_pred,       # (16, 50)
    duration_rounded,        # (16, 50)
    src_masks,               # (16, 50)
    mel_masks,               # (16, 200)
    src_lens,                # (16,)
    mel_lens,                # (16,)
)
```

---

## 核心組件詳解

### 1. Encoder (transformer/Models.py)

**功能**: 將音素序列編碼為隱藏表示

```python
class Encoder:
    def __init__(self):
        self.src_word_emb = Embedding(vocab_size=76, hidden=256)
        self.position_enc = SinusoidEncoding(max_len=1001, hidden=256)
        self.layer_stack = [FFTBlock(...) for _ in range(4)]

    def forward(self, src_seq, mask):
        # src_seq: (B, T_text)

        # Embedding
        enc_output = self.src_word_emb(src_seq)  # (B, T_text, 256)
        enc_output += self.position_enc[:, :T_text, :]

        # Transformer layers
        for layer in self.layer_stack:
            enc_output = layer(enc_output, mask)  # (B, T_text, 256)

        return enc_output  # (B, T_text, 256)
```

**FFT Block 結構** (Feed-Forward Transformer):

```python
class FFTBlock:
    def forward(self, x, mask):
        # 1. Multi-Head Self-Attention
        attn_output = MultiHeadAttention(x, x, x, mask)
        x = LayerNorm(x + attn_output)  # Residual + Norm

        # 2. Position-wise Feed-Forward (Conv1D)
        ffn_output = Conv1D(x)
        x = LayerNorm(x + ffn_output)  # Residual + Norm

        return x
```

**Multi-Head Attention** (transformer/SubLayers.py):

```python
class MultiHeadAttention:
    def __init__(self, n_head=2, d_model=256):
        self.n_head = 2
        self.d_k = d_v = 256 // 2 = 128

        self.w_qs = Linear(256, 2 * 128)  # Query projection
        self.w_ks = Linear(256, 2 * 128)  # Key projection
        self.w_vs = Linear(256, 2 * 128)  # Value projection
        self.fc = Linear(2 * 128, 256)    # Output projection

    def forward(self, q, k, v, mask):
        # Input: (B, T, 256)

        # Project and split heads
        Q = self.w_qs(q).view(B, T, 2, 128)  # (B, T, n_head, d_k)
        K = self.w_ks(k).view(B, T, 2, 128)
        V = self.w_vs(v).view(B, T, 2, 128)

        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / sqrt(128)  # (B, n_head, T, T)
        attn = softmax(scores.masked_fill(mask, -inf))

        # Weighted sum
        output = attn @ V  # (B, n_head, T, d_v)

        # Concatenate heads and project
        output = output.view(B, T, 256)
        output = self.fc(output)  # (B, T, 256)

        return output
```

---

### 2. Variance Adaptor (model/modules.py)

**功能**: FastSpeech2 的核心創新，預測並加入韻律信息

#### Duration Predictor

```python
class VariancePredictor:
    def __init__(self):
        # 2 層 Conv1D + LayerNorm + Dropout
        self.conv_layer = Sequential(
            Conv1D(256, 256, kernel=3, padding=1),
            ReLU(),
            LayerNorm(256),
            Dropout(0.5),
            Conv1D(256, 256, kernel=3, padding=1),
            ReLU(),
            LayerNorm(256),
            Dropout(0.5),
        )
        self.linear = Linear(256, 1)  # 輸出預測值

    def forward(self, encoder_output, mask):
        # encoder_output: (B, T_text, 256)

        output = self.conv_layer(encoder_output)  # (B, T_text, 256)
        output = self.linear(output).squeeze(-1)  # (B, T_text)
        output = output.masked_fill(mask, 0.0)

        return output  # (B, T_text)
```

**Duration 使用方式**:
- **訓練時**: 使用 ground truth duration（從 MFA 對齊獲得）
- **推理時**: 使用預測的 duration
  ```python
  duration_rounded = round(exp(log_duration_pred) - 1) * d_control
  duration_rounded = clamp(duration_rounded, min=0)
  ```

#### Pitch & Energy Predictor

結構與 Duration Predictor 相同，但輸出會經過：

1. **Quantization** (量化到 bins):
   ```python
   # Pitch bins: 256 個等間距的區間 [pitch_min, pitch_max]
   pitch_bins = linspace(pitch_min, pitch_max, 255)  # 255 個邊界

   # Quantize continuous pitch to discrete bins
   pitch_indices = bucketize(pitch_values, pitch_bins)  # (B, T) -> indices [0-255]
   ```

2. **Embedding**:
   ```python
   pitch_embedding = PitchEmbedding(pitch_indices)  # (B, T, 256)
   encoder_output = encoder_output + pitch_embedding
   ```

**為何要量化？**
- 將連續值轉為離散表示
- 使用 Embedding 而非直接加上連續值，模型更容易學習
- 256 個 bins 足夠精細，不會損失太多信息

#### Length Regulator

**功能**: 將音素級別的表示擴展為幀級別

```python
class LengthRegulator:
    def forward(self, x, duration):
        # x: (B, T_text, 256)
        # duration: (B, T_text) - 例如 [3, 2, 4, 1, ...]

        output = []
        for batch_idx in range(B):
            expanded = []
            for phoneme_idx, dur in enumerate(duration[batch_idx]):
                # 將每個音素重複 dur 次
                phoneme_vec = x[batch_idx, phoneme_idx]  # (256,)
                expanded.append(phoneme_vec.repeat(dur, 1))  # (dur, 256)

            expanded = torch.cat(expanded, dim=0)  # (T_mel, 256)
            output.append(expanded)

        output = pad(output)  # (B, max_T_mel, 256)
        return output
```

**範例**:
```
音素序列: ["H", "E", "L", "L", "O"]  (T_text=5)
Duration:  [3,   2,   4,   1,   2]    (總和=12)

擴展後:   ["H","H","H", "E","E", "L","L","L","L", "L", "O","O"]  (T_mel=12)
```

---

### 3. Decoder (transformer/Models.py)

**功能**: 將幀級別的表示解碼為 mel-spectrogram

```python
class Decoder:
    def __init__(self):
        self.position_enc = SinusoidEncoding(max_len=1001, hidden=256)
        self.layer_stack = [FFTBlock(...) for _ in range(6)]

    def forward(self, enc_seq, mask):
        # enc_seq: (B, T_mel, 256) - 來自 Length Regulator

        # Add positional encoding
        dec_output = enc_seq + self.position_enc[:, :T_mel, :]

        # Transformer layers (6 layers)
        for layer in self.layer_stack:
            dec_output = layer(dec_output, mask)  # (B, T_mel, 256)

        return dec_output  # (B, T_mel, 256)
```

**與 Encoder 的區別**:
- Encoder: 4 layers
- Decoder: 6 layers（作者發現更深的 decoder 效果更好）
- 兩者都使用相同的 FFT Block 結構

---

### 4. PostNet (transformer/Layers.py)

**功能**: 進一步優化 mel-spectrogram 的細節

```python
class PostNet:
    def __init__(self):
        # 5 層 Conv1D + BatchNorm
        self.convolutions = [
            # Layer 0: 80 -> 512
            Conv1D(80, 512, kernel=5, padding=2) + BatchNorm(512),
            # Layers 1-3: 512 -> 512
            Conv1D(512, 512, kernel=5, padding=2) + BatchNorm(512),
            Conv1D(512, 512, kernel=5, padding=2) + BatchNorm(512),
            Conv1D(512, 512, kernel=5, padding=2) + BatchNorm(512),
            # Layer 4: 512 -> 80
            Conv1D(512, 80, kernel=5, padding=2) + BatchNorm(80),
        ]

    def forward(self, x):
        # x: (B, T_mel, 80)

        # Transpose for Conv1D: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)  # (B, 80, T_mel)

        # Apply convolutions
        for i, conv in enumerate(self.convolutions[:-1]):
            x = Dropout(Tanh(conv(x)), p=0.5)  # (B, 512, T_mel)

        x = Dropout(self.convolutions[-1](x), p=0.5)  # (B, 80, T_mel)

        # Transpose back
        x = x.transpose(1, 2)  # (B, T_mel, 80)

        return x
```

**使用方式**:
```python
mel_linear_output = MelLinear(decoder_output)  # (B, T_mel, 80)
postnet_output = PostNet(mel_linear_output)    # (B, T_mel, 80)
final_mel = mel_linear_output + postnet_output # Residual connection
```

---

## 重要資料結構總結

### 音素 (Phoneme) 表示

```python
# text/symbols.py 定義音素集合
symbols = [
    '_',  # padding
    'AA0', 'AA1', 'AA2',  # 英文音素
    'AE0', 'AE1', 'AE2',
    # ... 共 76 個符號
]

# 文字轉音素範例
"Hello" -> G2P -> ['HH', 'AH0', 'L', 'OW1']
-> text_to_sequence() -> [23, 15, 8, 32]  # 音素 ID
```

### Mel-Spectrogram 參數

```yaml
sampling_rate: 22050 Hz
n_mel_channels: 80      # Mel 頻帶數量
filter_length: 1024     # FFT 窗口大小
hop_length: 256         # 幀移
win_length: 1024        # 窗口長度
mel_fmin: 0 Hz
mel_fmax: 8000 Hz       # HiFi-GAN 推薦使用 8000 Hz
```

**時間關係**:
```
1 秒音訊 = 22050 個採樣點
1 幀 = hop_length = 256 個採樣點
1 秒音訊 = 22050 / 256 ≈ 86 幀
```

### Duration 的意義

```
Duration[i] = 音素 i 對應的 mel 幀數

範例:
文字: "cat"
音素: ['K', 'AE1', 'T']
Duration: [3, 5, 2]
含義:
  - 'K' 持續 3 幀
  - 'AE1' 持續 5 幀
  - 'T' 持續 2 幀
總共: 3 + 5 + 2 = 10 幀的 mel-spectrogram
```

---

## 訓練與推理的關鍵差異

| 項目 | 訓練 | 推理 |
|------|------|------|
| **Duration** | 使用 ground truth | 使用模型預測 |
| **Pitch** | 使用 ground truth | 使用模型預測（可調整 p_control） |
| **Energy** | 使用 ground truth | 使用模型預測（可調整 e_control） |
| **Length Regulator** | 使用 GT duration 擴展 | 使用預測 duration 擴展 |
| **Mel Target** | 用於計算 loss | 不需要 |
| **輸出** | Loss values | Mel-spectrogram → Vocoder → Audio |

**Teacher Forcing**:
- 訓練時使用真實的 duration/pitch/energy，模型學習預測它們
- 推理時完全依賴模型預測，無需任何 ground truth

---

## 完整範例追蹤

### 範例句子: "Hello"

```
1. 文字預處理
   Input:  "Hello"
   G2P:    ['HH', 'AH0', 'L', 'OW1']
   IDs:    [23, 15, 8, 32]
   Shape:  (4,)

2. Encoder
   Input:  (1, 4)  # batch_size=1, T_text=4
   Token Emb: (1, 4, 256)
   Pos Enc:   (1, 4, 256)
   4 × FFT:   (1, 4, 256)
   Output:    (1, 4, 256)

3. Variance Adaptor
   Duration Pred:  [3, 5, 2, 4]  # 預測每個音素持續幀數
   Pitch Pred:     [120, 115, 110, 105]  # Hz
   Energy Pred:    [0.8, 0.9, 0.7, 0.6]

   Pitch Embedding: (1, 4, 256)
   Energy Embedding: (1, 4, 256)

   Length Regulator:
     Input:  (1, 4, 256), duration=[3,5,2,4]
     Output: (1, 14, 256)  # 3+5+2+4=14 幀

4. Decoder
   Input:  (1, 14, 256)
   Pos Enc: (1, 14, 256)
   6 × FFT: (1, 14, 256)
   Output:  (1, 14, 256)

5. Mel Linear
   Input:  (1, 14, 256)
   Output: (1, 14, 80)  # Mel-spectrogram

6. PostNet
   Input:  (1, 14, 80)
   5 × Conv1D: (1, 14, 80)
   Output: (1, 14, 80)

7. Final Output
   Mel: (1, 14, 80)

8. Vocoder (HiFi-GAN)
   Input:  (1, 14, 80)
   Output: (1, 3584)  # 14 × 256 = 3584 個採樣點
   Time:   3584 / 22050 ≈ 0.16 秒
```

---

## 常見問題

### Q1: 為何需要 Duration Predictor？

**A**:
- TTS 需要知道每個音素應該持續多久
- 不同語境下同一音素持續時間不同（例如強調、語速）
- Duration 決定了語音的節奏和韻律

### Q2: Pitch 和 Energy 的區別？

**A**:
- **Pitch (音高)**: 基頻 F0，決定聲音的高低（Hz）
- **Energy (能量)**: 振幅大小，決定音量（dB 或歸一化值）
- 兩者共同決定語音的表現力

### Q3: 為何是 Phoneme-level 而非 Frame-level？

**A**:
作者實驗發現 phoneme-level 的 pitch/energy 預測韻律更自然：
- **Phoneme-level**: 每個音素一個值，更穩定
- **Frame-level**: 每幀一個值，可能過於細緻導致不穩定

配置在 `preprocess.yaml` 中：
```yaml
pitch:
  feature: "phoneme_level"  # 或 "frame_level"
```

### Q4: PostNet 的作用？

**A**:
- 來自 Tacotron2 的設計
- 通過 5 層卷積進一步優化 mel 細節
- Residual connection 確保不會破壞原始預測
- 訓練時同時監督 PostNet 前後的輸出

### Q5: 如何控制語音的韻律？

**A**:
推理時可調整控制參數：
```python
synthesize(
    p_control=1.2,  # 提高音調 20%
    e_control=0.8,  # 降低音量 20%
    d_control=1.5,  # 放慢語速 50%
)
```

---

## 總結

FastSpeech2 的核心設計：

1. **Non-autoregressive**: 並行生成所有幀，速度快
2. **Explicit Duration Modeling**: 明確預測每個音素的持續時間
3. **Variance Adaptor**: 預測並控制 pitch、energy、duration
4. **Teacher Forcing Training**: 訓練時使用 ground truth，推理時用預測
5. **Multi-task Learning**: 同時學習 mel、pitch、energy、duration

**優勢**:
- ✅ 生成速度快（並行）
- ✅ 韻律可控（調整 p/e/d_control）
- ✅ 訓練穩定（有明確的監督信號）
- ✅ 音質高（接近 ground truth）

**應用場景**:
- 文字轉語音 (TTS)
- 有聲書生成
- 語音助手
- 多語言語音合成
