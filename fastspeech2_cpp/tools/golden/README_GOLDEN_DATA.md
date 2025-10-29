# Golden Data Generation Strategy

This document describes the strategy for generating test data (golden data) to validate the C++ implementation against PyTorch reference outputs.

## Overview

Golden data is organized into three levels, matching the bottom-up testing strategy:

1. **Operator-Level**: Test individual operators (matmul, layer_norm, attention, etc.)
2. **Component-Level**: Test model components (Encoder, Decoder, Variance Adaptor, PostNet)
3. **Integration-Level**: Test end-to-end inference (phonemes → mel-spectrogram)

---

## 1. Operator-Level Test Data

**Purpose**: Validate basic operators in `op.cpp` against PyTorch implementations.

**Script**: `tools/golden/generate_op_test_data.py`

**Requirements**: PyTorch, NumPy (no trained model needed)

**Generated Data**:

| Operator | Input Shape | Output Shape | Test Case |
|----------|-------------|--------------|-----------|
| matmul | (64, 128), (128, 256) | (64, 256) | Matrix multiplication |
| layer_norm | (1, 100, 256) | (1, 100, 256) | Layer normalization with γ, β |
| softmax | (1, 2, 10, 10) | (1, 2, 10, 10) | Attention scores |
| gelu | (1000,) | (1000,) | Activation across range [-5, 5] |
| tanh | (1000,) | (1000,) | Activation across range [-5, 5] |
| conv1d | (1, 256, 100) | (1, 512, 100) | 1D convolution with kernel=3 |
| batch_norm | (1, 512, 100) | (1, 512, 100) | Batch normalization |
| multi_head_attention | Q,K,V: (1, 10, 256) | (1, 10, 256) | 2-head attention |
| positional_encoding | - | (1000, 256) | Sinusoidal encoding |

**Usage**:
```bash
cd fastspeech2_cpp
python3 tools/golden/generate_op_test_data.py --output_dir test/data
```

**Output**: `test/data/` directory with `.bin` and `_meta.txt` files for each operator.

**Validation Threshold**: MSE < 1e-5

---

## 2. Component-Level Test Data

**Purpose**: Validate model components against PyTorch intermediate outputs.

**Script**: `tools/extract_component_outputs.py`

**Requirements**: PyTorch, trained FastSpeech2 checkpoint

**Status**: ⚠️ **Requires trained model** (not yet available)

**Data to Extract**:

| Component | Input | Output | Extraction Method |
|-----------|-------|--------|-------------------|
| Encoder | phoneme_ids: (1, N) | encoder_out: (1, N, 256) | Hook after encoder.forward() |
| Variance Adaptor | encoder_out: (1, N, 256) | variance_out: (1, M, 256) | Hook after variance_adaptor.forward() |
| - Duration Predictor | encoder_out: (1, N, 256) | duration_pred: (1, N) | Intermediate hook |
| - Pitch Predictor | encoder_out: (1, N, 256) | pitch_pred: (1, N) | Intermediate hook |
| - Energy Predictor | encoder_out: (1, N, 256) | energy_pred: (1, N) | Intermediate hook |
| Decoder | variance_out: (1, M, 256) | decoder_out: (1, M, 256) | Hook after decoder.forward() |
| PostNet | mel: (1, M, 80) | postnet_mel: (1, M, 80) | Hook after postnet.forward() |

**Test Cases**:
- Short sentence: "Hello" (4-6 phonemes)
- Medium sentence: "The quick brown fox" (15-20 phonemes)
- Long sentence: Full sentence (50+ phonemes)

**Usage** (when checkpoint available):
```bash
python3 tools/extract_component_outputs.py \
    --checkpoint output/ckpt/LJSpeech/900000.pth.tar \
    --text "Hello world" \
    --output_dir test/data/components
```

**Validation Threshold**: MSE < 1e-4

---

## 3. Integration-Level Test Data

**Purpose**: Validate end-to-end inference from phoneme IDs to mel-spectrogram.

**Script**: `tools/generate_integration_test_data.py`

**Requirements**: PyTorch, trained FastSpeech2 checkpoint

**Status**: ⚠️ **Requires trained model** (not yet available)

**Test Cases**:

| Test Name | Text | Expected Phonemes | Expected Mel Frames |
|-----------|------|-------------------|---------------------|
| test_hello | "Hello" | ~4-6 | ~15-20 |
| test_world | "World" | ~5 | ~18-25 |
| test_fox | "The quick brown fox" | ~15 | ~60-80 |
| test_sentence | "Printing, in the only sense..." | ~50 | ~200-250 |

**Output**:
- Phoneme IDs: `.bin` and `.txt` formats
- Mel-spectrogram: `.bin` format with shape metadata
- Duration predictions: `.bin` format
- Pitch/Energy predictions: `.bin` format

**Usage** (when checkpoint available):
```bash
python3 tools/generate_integration_test_data.py \
    --checkpoint output/ckpt/LJSpeech/900000.pth.tar \
    --output_dir test/data/integration
```

**Validation Threshold**: MSE < 1e-3

---

## Current Status

### ✅ Completed

- [x] **Operator-level data generation script** (`generate_op_test_data.py`)
- [x] **Text-to-phonemes preprocessing tool** (`text_to_phonemes.py`)
- [x] **Golden data strategy documented**

### ⏳ In Progress

- [ ] **Generate operator-level test data** (waiting for dependencies)
- [ ] **Verify operator test data completeness**

### 🔒 Blocked (Need Trained Model)

- [ ] Component-level data extraction script
- [ ] Integration-level data generation script
- [ ] Generate component-level test data
- [ ] Generate integration-level test data

---

## Dependency Installation

To generate test data, install required dependencies:

```bash
# Minimal (for operator-level data)
pip3 install numpy torch

# Full (for all test data generation)
pip3 install -r requirements.txt
```

---

## Directory Structure

After generating all test data:

```
fastspeech2_cpp/test/data/
├── README.md                      # This file
│
├── operators/                     # Operator-level test data
│   ├── matmul_A.bin
│   ├── matmul_B.bin
│   ├── matmul_C.bin
│   ├── matmul_meta.txt
│   ├── layer_norm_input.bin
│   ├── layer_norm_gamma.bin
│   ├── layer_norm_beta.bin
│   ├── layer_norm_output.bin
│   ├── ... (all operator test files)
│
├── components/                    # Component-level test data
│   ├── hello_encoder_out.bin
│   ├── hello_variance_out.bin
│   ├── hello_decoder_out.bin
│   ├── hello_postnet_out.bin
│   ├── ... (test cases for each component)
│
└── integration/                   # Integration-level test data
    ├── test_hello_phonemes.bin
    ├── test_hello_mel.bin
    ├── test_hello_meta.json
    ├── ... (end-to-end test cases)
```

---

## Validation Workflow

### Phase 2: Operator Implementation

1. Generate operator test data: `make test-data-ops`
2. Implement operator in `op.cpp`
3. Write unit test in `test/test_ops.cpp`
4. Run test: `make test-ops`
5. Verify MSE < 1e-5

### Phase 3: Component Implementation

1. **[Blocked]** Generate component test data (needs trained model)
2. Implement component in `fastspeech2.cpp`
3. Write component test in `test/test_components.cpp`
4. Run test: `make test-components`
5. Verify MSE < 1e-4

### Phase 4: Integration

1. **[Blocked]** Generate integration test data (needs trained model)
2. Implement full pipeline in `main.cpp`
3. Write integration test in `test/test_integration.cpp`
4. Run test: `make test-integration`
5. Verify MSE < 1e-3

---

## Notes

1. **Operator-level data is model-agnostic**: Can be generated without a trained FastSpeech2 model.

2. **Component and integration data require trained model**: These tests extract intermediate outputs from an actual trained model.

3. **Binary format**: All test data is saved as float32 binary files for easy C++ loading.

4. **Metadata files**: Each test includes a `_meta.txt` file describing shapes and parameters.

5. **Reproducibility**: All random seeds are fixed (`torch.manual_seed(42)`) for reproducible test data.

---

## Troubleshooting

### Issue: "No module named 'numpy'"

**Solution**: Install dependencies:
```bash
pip3 install numpy torch
```

### Issue: "Checkpoint not found"

**Solution**: Component and integration tests require a trained model. Either:
- Train a model first, or
- Use operator-level tests only for now

### Issue: "MSE too large in tests"

**Solution**:
- Check operator implementation carefully
- Verify weight loading is correct
- Use `dump_tensor()` to debug intermediate values
- Compare tensor values element-by-element

---

## Summary

This three-tier testing strategy ensures:
- ✅ **Bottom-up validation**: Start with operators, build up to full model
- ✅ **Numerical correctness**: All outputs compared against PyTorch
- ✅ **Incremental development**: Test at each level before proceeding
- ✅ **Clear acceptance criteria**: MSE thresholds defined for each level

**Current milestone**: Generate operator-level test data and begin Phase 2 (operator implementation).
