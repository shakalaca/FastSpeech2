# FastSpeech2 C++ Implementation Roadmap

## Executive Summary

This document consolidates the final implementation plan for FastSpeech2 C++ inference engine, incorporating architectural decisions, file structure, testing methodology, and development roadmap.

**Design Philosophy**: llama2.c-inspired - minimal dependencies, straightforward code, clear structure

**Key Decisions**:
- ✅ 3-file core structure: main.cpp + fastspeech2.cpp + op.cpp
- ✅ Zero external dependencies (standard C++ library only)
- ✅ Phoneme IDs as input
- ✅ Include PostNet for high-quality mel-spectrograms
- ✅ Correctness prioritized over performance
- ✅ Bottom-up testing: operators → components → integration

---

## 1. File Structure

### 1.1 Final Architecture (3 Core Files)

```
fastspeech2_cpp/
├── main.cpp              (~150 lines)  - CLI interface & workflow
├── fastspeech2.h         (~100 lines)  - Model structures & API
├── fastspeech2.cpp       (~800 lines)  - Model implementation
├── op.h                  (~80 lines)   - Operator declarations
├── op.cpp                (~600 lines)  - Operator implementations
├── Makefile              (~50 lines)   - Build system
├── README.md             - Usage documentation
├── test/
│   ├── test_ops.cpp              - Operator unit tests
│   ├── test_components.cpp       - Component validation
│   ├── test_integration.cpp      - End-to-end tests
│   └── data/                     - Reference test data
└── tools/
    ├── convert_weights.py              - PyTorch → binary weights
    ├── text_to_phonemes.py             - Text → phoneme IDs
    ├── generate_op_test_data.py        - Operator test data
    ├── extract_component_outputs.py    - Component reference outputs
    └── compare_tensors.py              - Debugging comparison tool
```

### 1.2 Rationale: Why NOT Split encoder.cpp/decoder.cpp?

**Reasons to keep fastspeech2.cpp unified**:

1. **llama2.c Alignment**: Original llama2.c keeps the entire transformer in `run.c` (~700 lines). Splitting too much violates the "minimal & simple" principle.

2. **Manageable Code Size**:
   ```
   Encoder:         ~150 lines
   Variance Adaptor: ~200 lines
   Decoder:          ~150 lines
   PostNet:          ~100 lines
   Weight loading:   ~150 lines
   Utils:            ~50 lines
   ─────────────────────────
   Total:            ~800 lines ← Perfectly readable in one file
   ```

3. **Tight Coupling**: All components share:
   - Same `Config` struct
   - Same `RunState` buffers
   - Same `Weights` structure
   - Splitting creates artificial boundaries and more header dependencies

4. **Similar Structure**: Encoder and Decoder are both stacks of FFT blocks (Multi-Head Attention + Feed-Forward) - identical pattern, just different layer counts

5. **Clear Naming**: `fastspeech2.cpp` immediately tells developers "this is the model". Functions are prefixed clearly:
   ```cpp
   encoder_forward(...)
   variance_adaptor_forward(...)
   decoder_forward(...)
   postnet_forward(...)
   fastspeech2_forward(...)  // Orchestrates all above
   ```

6. **Testing Isolation**: Components can still be tested independently by exposing functions in header:
   ```cpp
   // In fastspeech2.h
   void encoder_forward(RunState* s, Config* p, Weights* w, int* x, int len);
   void decoder_forward(RunState* s, Config* p, Weights* w, int len);
   ```

**When would we split?**
- Only if fastspeech2.cpp exceeds **1500 lines**
- Or if adding significantly different model variants (e.g., FastSpeech2 vs Tacotron2)

**Conclusion**: Keep it simple - 3 files is optimal for clarity and maintainability.

---

## 2. Core Data Structures

### 2.1 Configuration (fastspeech2.h)

```cpp
typedef struct {
    // Model dimensions
    int dim;                    // 256 - hidden dimension
    int n_enc_layers;           // 4 - encoder FFT blocks
    int n_dec_layers;           // 6 - decoder FFT blocks
    int n_heads;                // 2 - attention heads
    int head_dim;               // 128 - dim per head
    int ffn_hidden;             // 1024 - FFN intermediate dim

    // Vocabulary & I/O
    int vocab_size;             // 76 - phoneme vocabulary
    int n_mels;                 // 80 - mel-spectrogram bins
    int max_seq_len;            // 1000 - max phoneme sequence

    // Variance predictor
    int var_pred_filter_size;   // 256
    int var_pred_kernel_size;   // 3
    float var_pred_dropout;     // 0.5
    int n_bins;                 // 256 - pitch/energy quantization bins

    // PostNet
    int postnet_embedding_dim;  // 512
    int postnet_kernel_size;    // 5
    int postnet_n_convolutions; // 5
} Config;
```

### 2.2 Weights (fastspeech2.h)

```cpp
typedef struct {
    // Encoder
    float* encoder_embedding;      // [vocab_size, dim]
    float* encoder_pe;             // [max_seq_len, dim] - positional encoding
    EncoderLayer* encoder_layers;  // Array of 4 layers

    // Variance Adaptor
    VariancePredictor duration_predictor;
    VariancePredictor pitch_predictor;
    VariancePredictor energy_predictor;
    float* pitch_embedding;        // [n_bins, dim]
    float* energy_embedding;       // [n_bins, dim]

    // Decoder
    float* decoder_pe;             // [max_seq_len, dim]
    DecoderLayer* decoder_layers;  // Array of 6 layers

    // Mel Linear
    float* mel_linear_weight;      // [dim, n_mels]
    float* mel_linear_bias;        // [n_mels]

    // PostNet
    PostNetLayer* postnet_layers;  // Array of 5 conv layers
} Weights;

typedef struct {
    // Multi-Head Attention
    float* attn_q_weight;          // [dim, dim]
    float* attn_q_bias;            // [dim]
    float* attn_k_weight;          // [dim, dim]
    float* attn_k_bias;            // [dim]
    float* attn_v_weight;          // [dim, dim]
    float* attn_v_bias;            // [dim]
    float* attn_out_weight;        // [dim, dim]
    float* attn_out_bias;          // [dim]
    float* attn_norm_gamma;        // [dim]
    float* attn_norm_beta;         // [dim]

    // Feed-Forward
    float* ffn_w1;                 // [dim, ffn_hidden]
    float* ffn_b1;                 // [ffn_hidden]
    float* ffn_w2;                 // [ffn_hidden, dim]
    float* ffn_b2;                 // [dim]
    float* ffn_norm_gamma;         // [dim]
    float* ffn_norm_beta;          // [dim]
} EncoderLayer;  // DecoderLayer has same structure
```

### 2.3 Runtime State (fastspeech2.h)

```cpp
typedef struct {
    // Encoder buffers
    float* encoder_emb;            // [max_seq_len, dim]
    float* encoder_out;            // [max_seq_len, dim]

    // Variance Adaptor buffers
    float* duration_pred;          // [max_seq_len]
    float* pitch_pred;             // [max_seq_len]
    float* energy_pred;            // [max_seq_len]
    int* durations;                // [max_seq_len] - rounded
    float* variance_out;           // [max_seq_len * max_duration, dim]
    int mel_len;                   // Actual mel frame count after length regulation

    // Decoder buffers
    float* decoder_out;            // [max_seq_len * max_duration, dim]

    // Output buffers
    float* mel_out;                // [max_seq_len * max_duration, n_mels]
    float* postnet_mel_out;        // [max_seq_len * max_duration, n_mels]

    // Temporary attention/FFN buffers (reused across layers)
    float* attn_q;                 // [max_seq_len, dim]
    float* attn_k;                 // [max_seq_len, dim]
    float* attn_v;                 // [max_seq_len, dim]
    float* attn_scores;            // [n_heads, max_seq_len, max_seq_len]
    float* attn_out;               // [max_seq_len, dim]
    float* ffn_hidden;             // [max_seq_len, ffn_hidden]
    float* ffn_out;                // [max_seq_len, dim]
} RunState;
```

---

## 3. Component Implementation Details

### 3.1 Encoder (in fastspeech2.cpp)

```cpp
void encoder_forward(RunState* s, Config* c, Weights* w, int* phoneme_ids, int len) {
    // 1. Embedding lookup
    for (int i = 0; i < len; i++) {
        int ph_id = phoneme_ids[i];
        memcpy(s->encoder_emb + i * c->dim,
               w->encoder_embedding + ph_id * c->dim,
               c->dim * sizeof(float));
    }

    // 2. Add positional encoding
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < c->dim; j++) {
            s->encoder_emb[i * c->dim + j] += w->encoder_pe[i * c->dim + j];
        }
    }

    // 3. Pass through encoder layers
    float* x = s->encoder_emb;
    for (int layer = 0; layer < c->n_enc_layers; layer++) {
        encoder_layer_forward(s, c, &w->encoder_layers[layer], x, len);
        x = s->attn_out;  // Output becomes input to next layer
    }

    // 4. Copy final output
    memcpy(s->encoder_out, x, len * c->dim * sizeof(float));
}

void encoder_layer_forward(RunState* s, Config* c, EncoderLayer* l, float* x, int len) {
    // Multi-Head Self-Attention
    multi_head_attention(s, c, l, x, x, x, len, len);  // Q=K=V=x (self-attention)

    // Residual + LayerNorm
    for (int i = 0; i < len * c->dim; i++) {
        s->attn_out[i] = s->attn_out[i] + x[i];
    }
    layer_norm(s->attn_out, l->attn_norm_gamma, l->attn_norm_beta, len, c->dim);

    // Position-wise Feed-Forward
    feed_forward(s, c, l, s->attn_out, len);

    // Residual + LayerNorm
    for (int i = 0; i < len * c->dim; i++) {
        s->ffn_out[i] = s->ffn_out[i] + s->attn_out[i];
    }
    layer_norm(s->ffn_out, l->ffn_norm_gamma, l->ffn_norm_beta, len, c->dim);

    memcpy(s->attn_out, s->ffn_out, len * c->dim * sizeof(float));
}
```

### 3.2 Variance Adaptor (in fastspeech2.cpp)

```cpp
void variance_adaptor_forward(RunState* s, Config* c, Weights* w, int src_len) {
    // 1. Predict duration, pitch, energy
    variance_predictor_forward(s, c, &w->duration_predictor, s->encoder_out, src_len, s->duration_pred);
    variance_predictor_forward(s, c, &w->pitch_predictor, s->encoder_out, src_len, s->pitch_pred);
    variance_predictor_forward(s, c, &w->energy_predictor, s->encoder_out, src_len, s->energy_pred);

    // 2. Quantize and embed pitch/energy
    float* pitch_emb = s->attn_q;  // Reuse buffer
    float* energy_emb = s->attn_k;
    for (int i = 0; i < src_len; i++) {
        int pitch_bin = quantize(s->pitch_pred[i], c->n_bins);
        int energy_bin = quantize(s->energy_pred[i], c->n_bins);

        memcpy(pitch_emb + i * c->dim,
               w->pitch_embedding + pitch_bin * c->dim,
               c->dim * sizeof(float));
        memcpy(energy_emb + i * c->dim,
               w->energy_embedding + energy_bin * c->dim,
               c->dim * sizeof(float));
    }

    // 3. Add embeddings to encoder output
    for (int i = 0; i < src_len * c->dim; i++) {
        s->encoder_out[i] += pitch_emb[i] + energy_emb[i];
    }

    // 4. Length Regulation - expand phoneme sequence to mel frames
    s->mel_len = 0;
    for (int i = 0; i < src_len; i++) {
        s->durations[i] = (int)(s->duration_pred[i] + 0.5);  // Round
        if (s->durations[i] < 1) s->durations[i] = 1;

        // Repeat phoneme embedding duration[i] times
        for (int d = 0; d < s->durations[i]; d++) {
            memcpy(s->variance_out + s->mel_len * c->dim,
                   s->encoder_out + i * c->dim,
                   c->dim * sizeof(float));
            s->mel_len++;
        }
    }
}
```

### 3.3 Decoder (in fastspeech2.cpp)

```cpp
void decoder_forward(RunState* s, Config* c, Weights* w, int mel_len) {
    // 1. Add positional encoding
    float* x = s->variance_out;
    for (int i = 0; i < mel_len; i++) {
        for (int j = 0; j < c->dim; j++) {
            x[i * c->dim + j] += w->decoder_pe[i * c->dim + j];
        }
    }

    // 2. Pass through decoder layers (similar to encoder)
    for (int layer = 0; layer < c->n_dec_layers; layer++) {
        decoder_layer_forward(s, c, &w->decoder_layers[layer], x, mel_len);
        x = s->attn_out;
    }

    // 3. Copy final output
    memcpy(s->decoder_out, x, mel_len * c->dim * sizeof(float));
}
```

### 3.4 PostNet (in fastspeech2.cpp)

```cpp
void postnet_forward(RunState* s, Config* c, Weights* w, int mel_len) {
    // 1. Project to PostNet dimension
    // mel_out: [mel_len, 80] -> [mel_len, 512]
    float* x = s->attn_q;  // Reuse buffer
    matmul(x, s->mel_out, w->postnet_layers[0].conv_weight,
           mel_len, c->n_mels, c->postnet_embedding_dim);

    // 2. Pass through 5 Conv1D + BatchNorm layers
    for (int layer = 0; layer < c->postnet_n_convolutions; layer++) {
        conv1d_forward(s, c, &w->postnet_layers[layer], x, mel_len);
        batch_norm(s->ffn_out, w->postnet_layers[layer].bn_gamma,
                   w->postnet_layers[layer].bn_beta, mel_len, c->postnet_embedding_dim);

        if (layer < c->postnet_n_convolutions - 1) {
            tanh_activation(s->ffn_out, mel_len * c->postnet_embedding_dim);
        }
        x = s->ffn_out;
    }

    // 3. Project back to mel dimension and add residual
    matmul(s->postnet_mel_out, x, w->postnet_layers[4].conv_weight,
           mel_len, c->postnet_embedding_dim, c->n_mels);
    for (int i = 0; i < mel_len * c->n_mels; i++) {
        s->postnet_mel_out[i] += s->mel_out[i];
    }
}
```

---

## 4. Operator Implementations (op.cpp)

All operators in `op.cpp` (~600 lines):

```cpp
// Basic linear algebra
void matmul(float* C, float* A, float* B, int M, int K, int N);
void vec_add(float* out, float* a, float* b, int n);

// Normalizations
void layer_norm(float* out, float* x, float* gamma, float* beta, int seq_len, int dim);
void batch_norm(float* out, float* x, float* gamma, float* beta, int frames, int channels);

// Activations
void gelu(float* x, int n);
void tanh_activation(float* x, int n);
void softmax(float* x, int n_heads, int seq_len, int kv_len);

// Attention
void multi_head_attention(RunState* s, Config* c, EncoderLayer* l,
                          float* q, float* k, float* v, int q_len, int kv_len);

// Convolutions
void conv1d(float* out, float* in, float* weight, float* bias,
            int batch, int in_ch, int out_ch, int len, int kernel);

// Positional encoding
void sinusoidal_position_encoding(float* pe, int max_len, int dim);

// Utilities
int quantize(float value, int n_bins);
void dump_tensor(const char* name, float* tensor, int rows, int cols);
```

---

## 5. Testing Strategy

See **TESTING_PLAN.md** for comprehensive details. Summary:

### 5.1 Three-Level Testing

1. **Operator Unit Tests** (`test/test_ops.cpp`)
   - Test each operator in isolation
   - Compare against NumPy/PyTorch reference
   - Threshold: MSE < 1e-5

2. **Component Validation** (`test/test_components.cpp`)
   - Extract intermediate outputs from PyTorch with hooks
   - Compare C++ component outputs layer-by-layer
   - Threshold: MSE < 1e-4

3. **Integration Tests** (`test/test_integration.cpp`)
   - Full phoneme → mel inference
   - Multiple test cases (short, medium, long sentences)
   - Threshold: MSE < 1e-3

### 5.2 Development Workflow

```bash
# Generate reference data
make test-data

# Test operators
make test-ops

# Test components
make test-components

# Test end-to-end
make test-integration

# Debug
make debug
./fastspeech2_debug input.txt
python tools/compare_tensors.py
python tools/visualize_mel.py
```

---

## 6. Implementation Roadmap

### Phase 1: Foundation (Days 1-2)

**Goal**: Set up project structure and build system

- [x] Create directory structure
- [ ] Write Makefile with targets: `build`, `test-data`, `test-ops`, `test-components`, `test-integration`, `debug`
- [ ] Define all data structures in `fastspeech2.h` and `op.h`
- [ ] Write weight conversion tool `tools/convert_weights.py`
- [ ] Write phoneme preprocessing tool `tools/text_to_phonemes.py`

**Deliverable**: Compilable skeleton with headers

### Phase 2: Operators (Days 3-5)

**Goal**: Implement and validate all operators

- [ ] Implement basic ops: `matmul`, `vec_add`, `layer_norm`
- [ ] Implement activations: `gelu`, `tanh`, `softmax`
- [ ] Implement `multi_head_attention`
- [ ] Implement `conv1d`, `batch_norm`
- [ ] Implement `sinusoidal_position_encoding`
- [ ] Generate operator test data with `tools/generate_op_test_data.py`
- [ ] Write `test/test_ops.cpp`
- [ ] **Validate**: All operator tests pass (MSE < 1e-5)

**Deliverable**: Fully tested `op.cpp` (~600 lines)

### Phase 3: Model Components (Days 6-10)

**Goal**: Implement Encoder, Variance Adaptor, Decoder, PostNet

**Day 6-7: Encoder**
- [ ] Implement `encoder_forward()`
- [ ] Implement `encoder_layer_forward()`
- [ ] Extract PyTorch encoder output with `tools/extract_component_outputs.py`
- [ ] Write `test_encoder()` in `test/test_components.cpp`
- [ ] **Validate**: Encoder MSE < 1e-4

**Day 8: Variance Adaptor**
- [ ] Implement `variance_predictor_forward()`
- [ ] Implement `variance_adaptor_forward()` with length regulation
- [ ] Extract PyTorch variance adaptor outputs
- [ ] Write `test_variance_adaptor()`
- [ ] **Validate**: Variance adaptor MSE < 1e-4

**Day 9: Decoder**
- [ ] Implement `decoder_forward()`
- [ ] Implement `decoder_layer_forward()`
- [ ] Extract PyTorch decoder output
- [ ] Write `test_decoder()`
- [ ] **Validate**: Decoder MSE < 1e-4

**Day 10: PostNet**
- [ ] Implement `postnet_forward()`
- [ ] Extract PyTorch PostNet output
- [ ] Write `test_postnet()`
- [ ] **Validate**: PostNet MSE < 1e-4

**Deliverable**: Fully tested `fastspeech2.cpp` (~800 lines)

### Phase 4: Integration (Days 11-12)

**Goal**: End-to-end inference pipeline

- [ ] Implement `fastspeech2_forward()` orchestration
- [ ] Implement weight loading `load_weights()`
- [ ] Implement `main.cpp` CLI interface
- [ ] Generate integration test data with `tools/generate_integration_test_data.py`
- [ ] Write `test/test_integration.cpp`
- [ ] **Validate**: All integration tests pass (MSE < 1e-3)

**Deliverable**: Working inference binary

### Phase 5: Refinement (Days 13-14)

**Goal**: Debugging, optimization, documentation

- [ ] Fix any precision issues (target MSE < 1e-4)
- [ ] Visual mel-spectrogram validation
- [ ] Memory leak check (valgrind)
- [ ] Write comprehensive README.md
- [ ] Add usage examples
- [ ] Performance benchmarking

**Deliverable**: Production-ready C++ inference engine

---

## 7. Acceptance Criteria

### 7.1 Correctness
- ✅ All operator tests pass: MSE < 1e-5
- ✅ All component tests pass: MSE < 1e-4
- ✅ All integration tests pass: MSE < 1e-3
- ✅ Visual mel comparison shows no artifacts

### 7.2 Code Quality
- ✅ Zero external dependencies
- ✅ No memory leaks (valgrind clean)
- ✅ Clear code structure (3 files: main, model, ops)
- ✅ Comprehensive comments

### 7.3 Usability
- ✅ Simple CLI: `./fastspeech2 checkpoint.bin phonemes.txt output.mel`
- ✅ Clear error messages
- ✅ README with examples

### 7.4 Performance
- ⏱️ Inference time < 100ms for 10-phoneme input (CPU)
- 💾 Memory usage < 500MB

---

## 8. Next Steps

### Immediate Actions:

1. **Review & Approve**: Confirm file structure and roadmap
2. **Start Phase 1**: Set up project skeleton
3. **Generate Test Data**: Run `tools/generate_op_test_data.py`
4. **Begin Operator Implementation**: Start with `matmul`, `layer_norm`, `gelu`

### Questions for Clarification:

1. ✅ File structure confirmed (3 files)
2. ✅ Testing approach confirmed (bottom-up)
3. Do you want to proceed with implementation immediately?
4. Should we start with operator implementation or full skeleton first?

---

## 9. Summary

This roadmap provides:

- ✅ **Clear file structure**: 3 core files, minimal dependencies
- ✅ **Comprehensive testing**: Operator → Component → Integration
- ✅ **Phased implementation**: 14-day timeline with clear milestones
- ✅ **Quality gates**: Numerical thresholds at every level
- ✅ **llama2.c philosophy**: Simple, direct, zero dependencies

**Estimated Timeline**: 10-14 days to production-ready C++ inference engine

**Key Philosophy**: Correctness first, clarity second, performance third.

Ready to begin implementation! 🚀
