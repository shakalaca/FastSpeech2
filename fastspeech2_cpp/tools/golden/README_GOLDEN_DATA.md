# Operator-Level Fixture Generator

The C++ runtime is now on par with the PyTorch implementation, and higher-level
parity checks happen via the scripts in `tools/parity/`. The remaining value of
this directory is the operator-level fixture generator, which predates weight
conversion and still helps when unit-testing low-level kernels in isolation.

If you need quick synthetic inputs/outputs for functions in `op.cpp`, use the
script below; otherwise prefer the parity tooling for real-model validation.

## Operator Fixtures

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

## Current Status

### ✅ Completed

- [x] **Operator-level data generation script** (`generate_op_test_data.py`)
- [x] **Text-to-phonemes preprocessing tool** (`text_to_phonemes.py`)
- [x] **Golden data strategy documented**

### ⏳ In Progress

- [ ] **Generate operator-level test data** (waiting for dependencies)
- [ ] **Verify operator test data completeness**

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

## Notes

1. **Operator-level data is model-agnostic**: Can be generated without a trained FastSpeech2 model.

2. Prefer the parity scripts (`tools/parity/compare_runtime.py`, `tools/parity/compare_dumps.py`) for validating the real model; they operate directly on converted weights and checkpoints.

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

**Solution**: Higher-level fixtures are no longer maintained here. Use the parity scripts with a converted checkpoint instead.

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
