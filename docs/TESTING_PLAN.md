# FastSpeech2 C++ Implementation Testing Plan

## Overview

This document outlines a comprehensive testing strategy for validating the C++ implementation against the PyTorch reference model. The testing approach follows a **bottom-up methodology**: validate operators first, then components, then end-to-end inference.

## Testing Philosophy

1. **Layer-by-layer validation**: Compare intermediate outputs at every stage
2. **Numerical precision**: MSE < 1e-4 between PyTorch and C++ outputs
3. **Automated regression**: All tests should be reproducible and automated
4. **Reference-driven**: Generate ground truth from PyTorch model

---

## 1. Operator-Level Unit Tests

### 1.1 Test Suite: `test/test_ops.cpp`

Test each operator in `op.cpp` individually against NumPy/PyTorch reference outputs.

#### Test Cases:

| Operator | Test Input | Reference | Validation |
|----------|------------|-----------|------------|
| `matmul` | Random (64, 128) × (128, 256) | `np.matmul` | MSE < 1e-5 |
| `layer_norm` | Random (1, 100, 256) with γ, β | `F.layer_norm` | MSE < 1e-5 |
| `softmax` | Random (1, 2, 10, 10) | `F.softmax` | MSE < 1e-6 |
| `gelu` | Linspace(-5, 5, 1000) | `F.gelu` | MSE < 1e-5 |
| `conv1d` | Random (1, 256, 100), kernel (512, 256, 3) | `F.conv1d` | MSE < 1e-5 |
| `rmsnorm` | Random (1, 100, 256) | Custom reference | MSE < 1e-5 |
| `multi_head_attention` | Q,K,V (1, 10, 256) | `MultiHeadAttention` | MSE < 1e-4 |
| `positional_encoding` | Seq len 100, dim 256 | Sinusoidal formula | MSE < 1e-6 |

#### Implementation:

```cpp
// test/test_ops.cpp
#include "../fastspeech2.h"
#include "../op.h"
#include <fstream>
#include <cmath>

float mse(float* a, float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum / n;
}

void test_matmul() {
    // Load test data from reference
    float* A = load_tensor("test/data/matmul_input_A.bin", 64 * 128);
    float* B = load_tensor("test/data/matmul_input_B.bin", 128 * 256);
    float* expected = load_tensor("test/data/matmul_output.bin", 64 * 256);

    float* C = new float[64 * 256];
    matmul(C, A, B, 64, 128, 256);

    float error = mse(C, expected, 64 * 256);
    printf("test_matmul: MSE = %.8f %s\n", error, error < 1e-5 ? "PASS" : "FAIL");

    delete[] A; delete[] B; delete[] C; delete[] expected;
}

void test_layer_norm() {
    float* x = load_tensor("test/data/layernorm_input.bin", 100 * 256);
    float* gamma = load_tensor("test/data/layernorm_gamma.bin", 256);
    float* beta = load_tensor("test/data/layernorm_beta.bin", 256);
    float* expected = load_tensor("test/data/layernorm_output.bin", 100 * 256);

    float* output = new float[100 * 256];
    layer_norm(output, x, gamma, beta, 100, 256);

    float error = mse(output, expected, 100 * 256);
    printf("test_layer_norm: MSE = %.8f %s\n", error, error < 1e-5 ? "PASS" : "FAIL");

    delete[] x; delete[] gamma; delete[] beta; delete[] output; delete[] expected;
}

// ... similar for all operators

int main() {
    printf("Running operator unit tests...\n\n");
    test_matmul();
    test_layer_norm();
    test_softmax();
    test_gelu();
    test_conv1d();
    test_rmsnorm();
    test_multi_head_attention();
    test_positional_encoding();
    printf("\nAll operator tests completed.\n");
    return 0;
}
```

#### Reference Data Generation:

```python
# tools/generate_op_test_data.py
import torch
import torch.nn.functional as F
import numpy as np

def save_tensor(tensor, path):
    tensor.detach().cpu().numpy().astype(np.float32).tofile(path)

# Matmul test
A = torch.randn(64, 128)
B = torch.randn(128, 256)
C = torch.matmul(A, B)
save_tensor(A, 'test/data/matmul_input_A.bin')
save_tensor(B, 'test/data/matmul_input_B.bin')
save_tensor(C, 'test/data/matmul_output.bin')

# LayerNorm test
x = torch.randn(1, 100, 256)
ln = torch.nn.LayerNorm(256)
output = ln(x)
save_tensor(x, 'test/data/layernorm_input.bin')
save_tensor(ln.weight, 'test/data/layernorm_gamma.bin')
save_tensor(ln.bias, 'test/data/layernorm_beta.bin')
save_tensor(output, 'test/data/layernorm_output.bin')

# GELU test
x = torch.linspace(-5, 5, 1000)
output = F.gelu(x)
save_tensor(x, 'test/data/gelu_input.bin')
save_tensor(output, 'test/data/gelu_output.bin')

# Conv1D test
x = torch.randn(1, 256, 100)  # (batch, channels, length)
conv = torch.nn.Conv1d(256, 512, kernel_size=3, padding=1)
output = conv(x)
save_tensor(x, 'test/data/conv1d_input.bin')
save_tensor(conv.weight, 'test/data/conv1d_weight.bin')
save_tensor(conv.bias, 'test/data/conv1d_bias.bin')
save_tensor(output, 'test/data/conv1d_output.bin')

# Multi-head attention test
Q = torch.randn(1, 10, 256)
K = torch.randn(1, 10, 256)
V = torch.randn(1, 10, 256)
attn = torch.nn.MultiheadAttention(embed_dim=256, num_heads=2, batch_first=True)
output, _ = attn(Q, K, V)
save_tensor(Q, 'test/data/mha_query.bin')
save_tensor(K, 'test/data/mha_key.bin')
save_tensor(V, 'test/data/mha_value.bin')
save_tensor(attn.in_proj_weight, 'test/data/mha_in_proj_weight.bin')
save_tensor(attn.in_proj_bias, 'test/data/mha_in_proj_bias.bin')
save_tensor(attn.out_proj.weight, 'test/data/mha_out_proj_weight.bin')
save_tensor(attn.out_proj.bias, 'test/data/mha_out_proj_bias.bin')
save_tensor(output, 'test/data/mha_output.bin')

print("✓ Generated all operator test data")
```

---

## 2. Component-Level Validation

### 2.1 Test Suite: `test/test_components.cpp`

Test each model component (Encoder, Variance Adaptor, Decoder, PostNet) against PyTorch outputs.

#### Test Strategy:

1. Load actual model weights from trained checkpoint
2. Use real phoneme input from validation set
3. Extract intermediate outputs from PyTorch at component boundaries
4. Compare C++ outputs against PyTorch outputs

#### Component Test Cases:

| Component | Input | PyTorch Hook | Expected Output Shape | Threshold |
|-----------|-------|--------------|----------------------|-----------|
| Encoder | phoneme_ids (1, 10) | `after encoder.forward()` | (1, 10, 256) | MSE < 1e-4 |
| Variance Adaptor | encoder_out (1, 10, 256) | `after variance_adaptor.forward()` | (1, 45, 256) + predictions | MSE < 1e-4 |
| Decoder | variance_out (1, 45, 256) | `after decoder.forward()` | (1, 45, 256) | MSE < 1e-4 |
| PostNet | decoder_out (1, 45, 80) | `after postnet.forward()` | (1, 45, 80) | MSE < 1e-4 |

#### Implementation:

```cpp
// test/test_components.cpp
#include "../fastspeech2.h"

void test_encoder() {
    Config config = load_config("output/ckpt/LJSpeech/config.bin");
    Weights weights = load_weights("output/ckpt/LJSpeech/weights.bin", &config);
    RunState state = create_run_state(&config);

    // Load test input
    int phoneme_ids[] = {23, 15, 8, 32, 45, 12, 0, 0, 0, 0};
    int n_phonemes = 6;

    // Load expected output from PyTorch
    float* expected = load_tensor("test/data/encoder_output.bin", 10 * 256);

    // Run encoder
    encoder_forward(&state, &config, &weights, phoneme_ids, n_phonemes);

    // Compare
    float error = mse(state.encoder_out, expected, n_phonemes * 256);
    printf("test_encoder: MSE = %.8f %s\n", error, error < 1e-4 ? "PASS" : "FAIL");

    free_run_state(&state);
    free_weights(&weights);
    delete[] expected;
}

void test_variance_adaptor() {
    // Similar structure
    // Load encoder output as input
    // Run variance_adaptor_forward()
    // Compare outputs (expanded sequence + pitch/energy/duration predictions)
}

void test_decoder() {
    // Load variance adaptor output as input
    // Run decoder_forward()
    // Compare decoder output
}

void test_postnet() {
    // Load decoder mel output as input
    // Run postnet_forward()
    // Compare refined mel output
}

int main() {
    printf("Running component tests...\n\n");
    test_encoder();
    test_variance_adaptor();
    test_decoder();
    test_postnet();
    printf("\nAll component tests completed.\n");
    return 0;
}
```

#### Reference Data Extraction:

```python
# tools/extract_component_outputs.py
import torch
import yaml
from model import FastSpeech2
from text import text_to_sequence
import numpy as np

# Load model
device = torch.device('cpu')
checkpoint = torch.load('output/ckpt/LJSpeech/900000.pth.tar', map_location=device)
model_config = yaml.load(open('config/LJSpeech/model.yaml'), Loader=yaml.FullLoader)
model = FastSpeech2(preprocess_config, model_config).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

# Test input
text = "Hello world"
phoneme_ids = torch.LongTensor([23, 15, 8, 32, 45, 12]).unsqueeze(0)  # (1, 6)
src_len = torch.LongTensor([6])

# Hook to capture intermediate outputs
outputs = {}

def hook(name):
    def fn(module, input, output):
        if isinstance(output, tuple):
            outputs[name] = output[0].detach().cpu().numpy()
        else:
            outputs[name] = output.detach().cpu().numpy()
    return fn

# Register hooks
model.encoder.register_forward_hook(hook('encoder'))
model.variance_adaptor.register_forward_hook(hook('variance_adaptor'))
model.decoder.register_forward_hook(hook('decoder'))
model.postnet.register_forward_hook(hook('postnet'))

# Forward pass
with torch.no_grad():
    output = model(
        speakers=torch.LongTensor([0]),
        texts=phoneme_ids,
        src_lens=src_len,
        max_src_len=6
    )

# Save intermediate outputs
outputs['encoder'].astype(np.float32).tofile('test/data/encoder_output.bin')
outputs['variance_adaptor'].astype(np.float32).tofile('test/data/variance_output.bin')
outputs['decoder'].astype(np.float32).tofile('test/data/decoder_output.bin')
outputs['postnet'].astype(np.float32).tofile('test/data/postnet_output.bin')

# Save final mel
output[0].cpu().numpy().astype(np.float32).tofile('test/data/final_mel.bin')

print("✓ Extracted all component outputs")
```

---

## 3. End-to-End Integration Tests

### 3.1 Test Suite: `test/test_integration.cpp`

Test the complete inference pipeline from phoneme IDs to mel-spectrogram.

#### Test Cases:

| Test Case | Input | Expected Output | Validation |
|-----------|-------|-----------------|------------|
| Short utterance | "Hello" (4 phonemes) | Mel (1, ~15, 80) | MSE < 1e-3 |
| Medium utterance | "The quick brown fox" (15 phonemes) | Mel (1, ~60, 80) | MSE < 1e-3 |
| Long utterance | Full sentence (50 phonemes) | Mel (1, ~200, 80) | MSE < 1e-3 |
| Edge case: Single phoneme | "A" (1 phoneme) | Mel (1, ~3, 80) | MSE < 1e-3 |
| Edge case: Max length | 100 phonemes | Mel (1, ~400, 80) | MSE < 1e-3 |

#### Implementation:

```cpp
// test/test_integration.cpp
void test_full_inference(const char* test_name, int* phoneme_ids, int n_phonemes,
                         const char* expected_mel_path) {
    Config config = load_config("output/ckpt/LJSpeech/config.bin");
    Weights weights = load_weights("output/ckpt/LJSpeech/weights.bin", &config);
    RunState state = create_run_state(&config);

    // Run full inference
    fastspeech2_forward(&state, &config, &weights, phoneme_ids, n_phonemes);

    // Load expected mel from PyTorch
    int mel_frames = state.mel_len;
    float* expected_mel = load_tensor(expected_mel_path, mel_frames * 80);

    // Compare
    float error = mse(state.mel_out, expected_mel, mel_frames * 80);
    printf("%s: mel_frames=%d, MSE=%.6f %s\n",
           test_name, mel_frames, error, error < 1e-3 ? "PASS" : "FAIL");

    free_run_state(&state);
    free_weights(&weights);
    delete[] expected_mel;
}

int main() {
    printf("Running integration tests...\n\n");

    int test1[] = {23, 15, 8, 32};
    test_full_inference("short_hello", test1, 4, "test/data/hello_mel.bin");

    int test2[] = {/* The quick brown fox phonemes */};
    test_full_inference("medium_sentence", test2, 15, "test/data/fox_mel.bin");

    int test3[] = {/* 50 phonemes */};
    test_full_inference("long_sentence", test3, 50, "test/data/long_mel.bin");

    printf("\nAll integration tests completed.\n");
    return 0;
}
```

#### Reference Generation:

```python
# tools/generate_integration_test_data.py
import torch
import yaml
from model import FastSpeech2

model = load_trained_model()
model.eval()

test_cases = [
    ("hello", [23, 15, 8, 32]),
    ("fox", [/* 15 phoneme IDs */]),
    ("long", [/* 50 phoneme IDs */]),
]

for name, phoneme_ids in test_cases:
    phonemes = torch.LongTensor(phoneme_ids).unsqueeze(0)
    src_len = torch.LongTensor([len(phoneme_ids)])

    with torch.no_grad():
        output = model(
            speakers=torch.LongTensor([0]),
            texts=phonemes,
            src_lens=src_len,
            max_src_len=len(phoneme_ids)
        )

    mel = output[0]  # mel_output
    postnet_mel = output[1]  # postnet_output

    # Save both for comparison
    mel.cpu().numpy().astype(np.float32).tofile(f'test/data/{name}_mel.bin')
    postnet_mel.cpu().numpy().astype(np.float32).tofile(f'test/data/{name}_postnet_mel.bin')

    # Save metadata
    with open(f'test/data/{name}_meta.txt', 'w') as f:
        f.write(f"n_phonemes: {len(phoneme_ids)}\n")
        f.write(f"mel_frames: {mel.shape[1]}\n")
        f.write(f"mel_shape: {mel.shape}\n")

print("✓ Generated integration test data")
```

---

## 4. Debugging Tools

### 4.1 Tensor Dumping Utility

```cpp
// In op.cpp or fastspeech2.cpp
void dump_tensor(const char* name, float* tensor, int rows, int cols) {
    char filename[256];
    sprintf(filename, "debug/%s.bin", name);
    FILE* f = fopen(filename, "wb");
    fwrite(tensor, sizeof(float), rows * cols, f);
    fclose(f);
    printf("Dumped tensor: %s [%d, %d]\n", name, rows, cols);
}

// Usage during debugging:
// encoder_forward(...) {
//     ... compute encoder_out ...
//     dump_tensor("encoder_out_cpp", state->encoder_out, n_phonemes, 256);
// }
```

### 4.2 Comparison Script

```python
# tools/compare_tensors.py
import numpy as np
import sys

def compare(cpp_path, pytorch_path, name):
    cpp = np.fromfile(cpp_path, dtype=np.float32)
    pytorch = np.fromfile(pytorch_path, dtype=np.float32)

    if cpp.shape != pytorch.shape:
        print(f"❌ {name}: Shape mismatch! C++={cpp.shape}, PyTorch={pytorch.shape}")
        return

    mse = np.mean((cpp - pytorch) ** 2)
    max_diff = np.max(np.abs(cpp - pytorch))

    print(f"{name}:")
    print(f"  MSE: {mse:.8f}")
    print(f"  Max diff: {max_diff:.8f}")
    print(f"  Status: {'✓ PASS' if mse < 1e-4 else '❌ FAIL'}")

if __name__ == "__main__":
    compare('debug/encoder_out_cpp.bin', 'test/data/encoder_output.bin', 'Encoder')
    compare('debug/variance_out_cpp.bin', 'test/data/variance_output.bin', 'Variance')
    compare('debug/decoder_out_cpp.bin', 'test/data/decoder_output.bin', 'Decoder')
    compare('debug/mel_out_cpp.bin', 'test/data/final_mel.bin', 'Final Mel')
```

### 4.3 Visual Mel Comparison

```python
# tools/visualize_mel.py
import numpy as np
import matplotlib.pyplot as plt

cpp_mel = np.fromfile('debug/mel_out_cpp.bin', dtype=np.float32).reshape(-1, 80).T
pytorch_mel = np.fromfile('test/data/final_mel.bin', dtype=np.float32).reshape(-1, 80).T

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].imshow(pytorch_mel, aspect='auto', origin='lower', interpolation='none')
axes[0].set_title('PyTorch Mel-Spectrogram')
axes[0].set_ylabel('Mel Bins')

axes[1].imshow(cpp_mel, aspect='auto', origin='lower', interpolation='none')
axes[1].set_title('C++ Mel-Spectrogram')
axes[1].set_ylabel('Mel Bins')

diff = np.abs(pytorch_mel - cpp_mel)
axes[2].imshow(diff, aspect='auto', origin='lower', interpolation='none', cmap='hot')
axes[2].set_title('Absolute Difference')
axes[2].set_ylabel('Mel Bins')
axes[2].set_xlabel('Frames')

plt.tight_layout()
plt.savefig('debug/mel_comparison.png', dpi=150)
print("✓ Saved mel comparison to debug/mel_comparison.png")
```

---

## 5. Continuous Testing Workflow

### 5.1 Makefile Targets

```makefile
# In Makefile

# Generate all test data from PyTorch
test-data:
	python tools/generate_op_test_data.py
	python tools/extract_component_outputs.py
	python tools/generate_integration_test_data.py

# Run unit tests
test-ops: test/test_ops.cpp op.cpp
	$(CC) -O3 test/test_ops.cpp op.cpp -o test_ops -lm
	./test_ops

# Run component tests
test-components: test/test_components.cpp fastspeech2.cpp op.cpp
	$(CC) -O3 test/test_components.cpp fastspeech2.cpp op.cpp -o test_components -lm
	./test_components

# Run integration tests
test-integration: test/test_integration.cpp fastspeech2.cpp op.cpp
	$(CC) -O3 test/test_integration.cpp fastspeech2.cpp op.cpp -o test_integration -lm
	./test_integration

# Run all tests
test: test-ops test-components test-integration
	@echo "All tests completed!"

# Debug mode - dump intermediate tensors
debug: main.cpp fastspeech2.cpp op.cpp
	$(CC) -g -DDEBUG -O0 main.cpp fastspeech2.cpp op.cpp -o fastspeech2_debug -lm
```

### 5.2 Development Workflow

```bash
# Step 1: Generate test data from PyTorch model
make test-data

# Step 2: Implement operators and test
make test-ops

# Step 3: Fix any failing operators, re-run
make test-ops

# Step 4: Implement components and test
make test-components

# Step 5: Test full integration
make test-integration

# Step 6: Debug if needed
make debug
./fastspeech2_debug input.txt
python tools/compare_tensors.py
python tools/visualize_mel.py
```

---

## 6. Acceptance Criteria

### 6.1 Operator Tests
- ✅ All 8 operator tests pass with MSE < 1e-5
- ✅ No memory leaks (valgrind clean)

### 6.2 Component Tests
- ✅ Encoder output MSE < 1e-4 vs PyTorch
- ✅ Variance Adaptor output MSE < 1e-4 vs PyTorch
- ✅ Decoder output MSE < 1e-4 vs PyTorch
- ✅ PostNet output MSE < 1e-4 vs PyTorch

### 6.3 Integration Tests
- ✅ All 5 end-to-end tests pass with MSE < 1e-3
- ✅ Visual mel-spectrogram comparison shows no obvious artifacts
- ✅ Output audio (after vocoder) is intelligible

### 6.4 Performance Benchmarks
- ⏱️ Inference time < 100ms for 10-phoneme input on CPU
- 💾 Memory usage < 500MB

---

## 7. Test Directory Structure

```
fastspeech2_cpp/
├── test/
│   ├── test_ops.cpp              # Operator unit tests
│   ├── test_components.cpp       # Component validation tests
│   ├── test_integration.cpp      # End-to-end tests
│   └── data/                     # Reference data from PyTorch
│       ├── matmul_input_A.bin
│       ├── matmul_input_B.bin
│       ├── matmul_output.bin
│       ├── layernorm_input.bin
│       ├── encoder_output.bin
│       ├── variance_output.bin
│       ├── decoder_output.bin
│       ├── postnet_output.bin
│       ├── hello_mel.bin
│       └── ... (all test data)
├── tools/
│   ├── generate_op_test_data.py        # Generate operator test data
│   ├── extract_component_outputs.py    # Extract PyTorch intermediate outputs
│   ├── generate_integration_test_data.py # Generate full inference test data
│   ├── compare_tensors.py              # Compare C++ vs PyTorch tensors
│   └── visualize_mel.py                # Visual mel comparison
└── debug/                          # Runtime debug dumps
    ├── encoder_out_cpp.bin
    ├── mel_out_cpp.bin
    └── mel_comparison.png
```

---

## 8. Timeline Estimate

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **Phase 1: Setup** | 1 day | Test directory, Makefile targets, reference data generation |
| **Phase 2: Operator Tests** | 2-3 days | All operator tests passing |
| **Phase 3: Component Tests** | 3-4 days | All component tests passing |
| **Phase 4: Integration Tests** | 2-3 days | End-to-end inference validation |
| **Phase 5: Debugging & Refinement** | 2-3 days | Fix precision issues, optimize |
| **Total** | **10-14 days** | Fully validated C++ implementation |

---

## Summary

This testing plan provides:
1. ✅ **Bottom-up validation**: Operators → Components → Integration
2. ✅ **Automated regression testing**: All tests reproducible
3. ✅ **Clear acceptance criteria**: Numerical thresholds defined
4. ✅ **Debugging tools**: Tensor dumping, comparison scripts, visualization
5. ✅ **Reference-driven approach**: PyTorch is ground truth
6. ✅ **Comprehensive coverage**: Unit, component, and integration tests

Following this plan will ensure the C++ implementation is **numerically correct** and **production-ready**.
