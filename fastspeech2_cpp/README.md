# FastSpeech2 C++ Inference Engine

A lightweight, zero-dependency C++ implementation of FastSpeech2 for text-to-speech inference.

## Features

- ✅ **Zero external dependencies** - Only standard C++11 library
- ✅ **Complete implementation** - Encoder, Variance Adaptor, Decoder, PostNet
- ✅ **Efficient inference** - Direct matrix operations, no overhead
- ✅ **Simple CLI** - Easy-to-use command-line interface
- ✅ **Portable** - Compiles on Linux, macOS, Windows

## Architecture

```
Phoneme IDs → Encoder (4 layers) → Variance Adaptor → Length Regulation
    → Decoder (6 layers) → Mel Linear → PostNet (5 layers) → Mel-Spectrogram
```

**Model Parameters**: ~26M parameters
- **Encoder**: 4 FFT blocks (Multi-Head Attention + Feed-Forward)
- **Variance Adaptor**: Duration/Pitch/Energy predictors + Length Regulator
- **Decoder**: 6 FFT blocks
- **PostNet**: 5 Conv1D layers with Batch Normalization

## Quick Start

### 1. Build

```bash
make build
```

### 2. Download Pretrained Model

Download the LJSpeech model (900000 steps) from:
https://drive.google.com/drive/folders/1DOhZGlTLMbbAAFZmZGDdc77kz1PloS7F

Place `900000.pth.tar` in `../output/ckpt/LJSpeech/`

Or use the helper script:
```bash
./tools/download_model.sh
```

### 3. Convert Weights

Convert PyTorch weights to binary format:

```bash
python3 tools/convert_weights.py \
    --checkpoint ../output/ckpt/LJSpeech/900000.pth.tar \
    --preprocess_config ../config/LJSpeech/preprocess.yaml \
    --model_config ../config/LJSpeech/model.yaml \
    --output_dir weights/
```

This will create:
```
weights/
├── config.bin
├── encoder/
│   ├── embedding.bin
│   └── layer_*/
├── variance_adaptor/
│   ├── duration_predictor/
│   ├── pitch_predictor/
│   ├── energy_predictor/
│   ├── pitch_embedding.bin
│   └── energy_embedding.bin
├── decoder/
│   └── layer_*/
├── mel_linear_weight.bin
├── mel_linear_bias.bin
└── postnet/
    └── layer_*/
```

### 4. Run Inference

```bash
# Using phoneme IDs directly
./fastspeech2 \
    --config weights/config.bin \
    --weights weights/ \
    --phonemes "23,15,8,32,45,12" \
    --output output.mel

# Using phoneme ID file
./fastspeech2 \
    --config weights/config.bin \
    --weights weights/ \
    --input phonemes.txt \
    --output output.mel
```

### 5. Convert Text to Phonemes

```bash
# Convert text to phoneme IDs
python3 tools/text_to_phonemes.py \
    --text "Hello world" \
    --output phonemes.txt

# The output will be saved as both .txt and .bin formats
```

## Project Structure

```
fastspeech2_cpp/
├── main.cpp              # CLI interface (210 lines)
├── fastspeech2.h         # Data structures (200 lines)
├── fastspeech2.cpp       # Model implementation (750 lines)
├── op.h                  # Operator declarations (80 lines)
├── op.cpp                # Operator implementations (370 lines)
├── Makefile              # Build system
├── tools/
│   ├── convert_weights.py          # PyTorch → Binary conversion
│   ├── text_to_phonemes.py         # Text → Phoneme IDs
│   ├── generate_op_test_data.py    # Test data generation
│   ├── download_model.sh           # Model download helper
│   └── README_GOLDEN_DATA.md       # Testing strategy
└── test/
    ├── test_ops.cpp                # Operator unit tests (TODO)
    ├── test_components.cpp         # Component tests (TODO)
    └── test_integration.cpp        # End-to-end tests (TODO)
```

**Total Lines of Code**: ~1,610 lines

## Usage

### Command Line Options

```bash
./fastspeech2 [OPTIONS]

Options:
  --config <path>       Path to config.bin file (required)
  --weights <dir>       Path to weights directory (required)
  --input <path>        Path to input phoneme IDs file (.txt or .bin)
  --phonemes <ids>      Phoneme IDs as comma-separated integers
  --output <path>       Output mel-spectrogram file (.bin)
  --help                Show help message

Example:
  ./fastspeech2 --config weights/config.bin --weights weights/ \
                --phonemes "23,15,8,32,45,12" --output output.mel
```

### Output Format

The output mel-spectrogram is saved in binary format:
- First 8 bytes: Shape (2 integers - [frames, n_mels])
- Remaining bytes: Mel data (float32, row-major order)

## Implementation Details

### Operators (op.cpp)

- **Linear algebra**: `matmul`, `vec_add`, `vec_scale`
- **Normalizations**: `layer_norm`, `batch_norm`
- **Activations**: `gelu`, `relu`, `tanh`, `softmax`
- **Attention**: `multi_head_attention` (scaled dot-product, 2 heads)
- **Feed-forward**: `feed_forward` (2-layer with GELU)
- **Convolution**: `conv1d` (with padding support)
- **Positional encoding**: `sinusoidal_position_encoding`

### Model Components (fastspeech2.cpp)

1. **Encoder**: 4 FFT blocks with self-attention
2. **Variance Adaptor**:
   - Duration Predictor: Predicts phoneme durations
   - Pitch Predictor: Predicts F0 values
   - Energy Predictor: Predicts energy values
   - Length Regulator: Expands phoneme-level to frame-level
3. **Decoder**: 6 FFT blocks with self-attention
4. **PostNet**: 5 Conv1D layers for mel refinement

### Design Philosophy

- **llama2.c-inspired**: Minimal, direct, readable code
- **Zero dependencies**: Only standard C++11 library
- **Correctness first**: Prioritize numerical accuracy over performance
- **Simple structure**: 3 core files (main, model, ops)

## Testing (TODO)

### Operator Tests

Generate test data:
```bash
python3 tools/generate_op_test_data.py --output_dir test/data
```

Run operator tests:
```bash
make test-ops
```

### Component Tests

Extract PyTorch intermediate outputs:
```bash
python3 tools/extract_component_outputs.py \
    --checkpoint ../output/ckpt/LJSpeech/900000.pth.tar \
    --text "Hello world" \
    --output_dir test/data/components
```

Run component tests:
```bash
make test-components
```

### Integration Tests

Run end-to-end tests:
```bash
make test-integration
```

## Performance

**Expected performance** (on modern CPU):
- Inference time: < 100ms for 10-phoneme input
- Memory usage: < 500MB
- Real-time factor: ~10x faster than real-time

**Note**: Current implementation prioritizes correctness. Performance optimizations will be added in future releases.

## Troubleshooting

### Issue: "Could not open config.bin"

**Solution**: Convert weights first using `convert_weights.py`

### Issue: "Could not open encoder/embedding.bin"

**Solution**: Ensure weights directory structure matches the expected format. Re-run `convert_weights.py`.

### Issue: "Warning: encoder/embedding.bin has X elements but expected Y"

**Solution**: Check that the PyTorch model matches the expected architecture (vocab_size=76, dim=256, etc.)

### Issue: Compilation errors

**Solution**: Ensure you have g++ with C++11 support:
```bash
g++ --version  # Should be >= 4.8
```

## Current Status

- ✅ Phase 1: Infrastructure complete
- ✅ Phase 2: Core implementation complete
- ⏳ Phase 3: Testing (operator tests pending)
- ⏳ Phase 4: Component validation (needs trained model)
- ⏳ Phase 5: Optimization & documentation

## Limitations

1. **Weight loading**: Requires converted binary weights (not GGUF format yet)
2. **Testing**: Operator/component tests not yet implemented
3. **Performance**: No SIMD/GPU optimization yet
4. **Vocoder**: Mel-to-waveform conversion not included (use HiFi-GAN separately)

## Future Work

- [ ] Operator unit tests with PyTorch validation
- [ ] Component-level tests
- [ ] GGUF format support
- [ ] SIMD optimization (AVX2, AVX-512)
- [ ] Multi-threading support
- [ ] Quantization (INT8/INT4)
- [ ] Integrated vocoder (HiFi-GAN/MelGAN)

## References

- **Paper**: [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)
- **PyTorch Implementation**: [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)
- **Inspiration**: [karpathy/llama2.c](https://github.com/karpathy/llama2.c)

## License

MIT License (same as original PyTorch implementation)

## Contributing

Contributions are welcome! Please see the [testing strategy](tools/README_GOLDEN_DATA.md) for guidelines.

---

**Status**: Core implementation complete (Phase 2). Ready for testing and validation.
