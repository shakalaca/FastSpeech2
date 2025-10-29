# FastSpeech2 C++ Inference Engine

A lightweight, dependency-free C++11 runtime for FastSpeech2 text-to-speech inference.  
It mirrors the original PyTorch implementation (ming024/FastSpeech2) and ships with helper
tools to convert checkpoints and prepare inputs.

## Directory Overview
- `fastspeech2.cpp`, `op.cpp`, `fastspeech2.h`, `op.h` – core model implementation and math ops
- `main.cpp` – CLI entry point
- `Makefile` – build, debug, and test targets
- `weights/` – binary weights produced by the conversion scripts
- `tools/` – Python utilities (checkpoint conversion, phoneme conversion, sanity checks)
- `requirements.txt` – Python dependencies needed by the tools

## Quick Start
1. **Prepare the Python environment**
   ```bash
   cd fastspeech2_cpp
   python3 -m venv .venv
   source .venv/bin/activate         # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt   # Installs PyYAML, torch, gdown, g2p-en, etc.
   ```

2. **Download the pretrained checkpoint**
   ```bash
   cd tools
   ./download_model.sh               # Uses gdown to fetch LJSpeech 900000.pth.tar
   cd ..
   ```
   The checkpoint ends up in `../output/ckpt/LJSpeech/900000.pth.tar`.

3. **Convert PyTorch weights to the C++ binary format**
   ```bash
   python3 tools/convert_weights.py \
     --checkpoint ../output/ckpt/LJSpeech/900000.pth.tar \
     --preprocess_config ../config/LJSpeech/preprocess.yaml \
     --model_config ../config/LJSpeech/model.yaml \
     --output_dir weights
   ```
   The converter reads the stats from the preprocessing directory, embeds the model
   configuration, and produces the layout expected by the runtime (`weights/encoder/...`,
   `weights/decoder/...`, `weights/config.bin`, etc.).

4. **(Optional) Validate the converted weights**
   ```bash
   python3 tools/verify_weights.py \
     --reference weights \
     --candidate /path/to/fresh_conversion \
     --tolerance 1e-5
   ```
   Omit `--candidate` to perform a quick NaN/Inf scan on a single directory.

5. **(Optional) Compare C++ runtime with PyTorch output**
   ```bash
   python3 tools/compare_runtime.py \
     --checkpoint ../output/ckpt/LJSpeech/900000.pth.tar \
     --preprocess_config ../config/LJSpeech/preprocess.yaml \
     --model_config ../config/LJSpeech/model.yaml \
     --weights_dir weights \
     --runtime ./fastspeech2 \
     --phonemes "23,15,8,32,45,12"
   ```
   The script runs both implementations and reports the maximum absolute/relative
   difference between their mel-spectrograms.

6. **Build the C++ binary**
   ```bash
   make build        # Produces ./fastspeech2
   ```

7. **Run inference**
   - With phoneme IDs in a file (comma or whitespace separated):
     ```bash
     ./fastspeech2 \
       --config weights/config.bin \
       --weights weights \
       --input test_input.txt \
       --output output_mel.bin
     ```
   - Passing phoneme IDs directly:
     ```bash
     ./fastspeech2 \
       --config weights/config.bin \
       --weights weights \
       --phonemes "23, 15, 8, 32, 45, 12" \
       --output output_mel.bin
     ```
   The binary prepends the mel shape `[frames, 80]` to the output file for easy inspection.

   Need phonemes from raw text? Use:
   ```bash
   python3 tools/text_to_phonemes.py --text "Hello world" --output phonemes.txt
   ```
   then feed `phonemes.txt` into the CLI.

## Command-line Reference
```
./fastspeech2 [OPTIONS]
  --config <path>    Required. Path to weights/config.bin.
  --weights <dir>    Required. Directory containing converted layer weights.
  --input <path>     Optional. Text file (comma/space separated) or binary int32 phoneme IDs.
  --phonemes <ids>   Optional. Inline comma-separated phoneme IDs.
  --output <path>    Optional. Destination for the mel spectrogram (default: output_mel.bin).
  --help             Print usage information.
```
Exactly one of `--input` or `--phonemes` must be supplied.

## Python Tools
- `convert_weights.py` – Torch-free checkpoint converter. Requires PyYAML, torch, and access
  to the original preprocessing stats. Outputs the full binary weight tree.
- `download_model.sh` – Convenience wrapper around `gdown` for the LJSpeech checkpoint.
- `text_to_phonemes.py` – Uses `g2p_en` to map text to phoneme IDs aligned with the model config.
- `verify_weights.py` – Compares two weight directories (or sanity-checks one) for mismatches.
- `compare_runtime.py` – Runs the PyTorch checkpoint and C++ runtime on the same phoneme IDs and reports numerical differences.
- `generate_op_test_data.py`, `extract_component_outputs.py`, `generate_integration_test_data.py` –
  Helper scripts for creating golden data used by the C++ test harness.
- `test_inference.py`, `test_checkpoint_load.py` – Lightweight smoke tests against PyTorch.

> **Tip:** Tools live outside the runtime and expect the Python environment from
> `requirements.txt`. Missing modules (e.g., `ModuleNotFoundError: No module named 'yaml'`)
> mean the requirements are not installed.

## Validation & Testing
- `make test-data` – Regenerates reference tensors from the PyTorch pipeline.
- `make test-ops` – Builds and runs operator unit tests (requires generated data).
- `make test-components` / `make test-integration` – Component and E2E comparisons.
- `make debug` – Builds a symbol-enabled binary for stepping through layers.

The current repository focuses on inference correctness. Operator/component test data
must be generated from a matching checkpoint before the test binaries can succeed.

## Runtime Verification
- `python3 tools/compare_runtime.py` – Compare mel-spectrograms produced by the PyTorch
  checkpoint and the C++ binary for a shared phoneme sequence. The script reports maximum
  absolute/relative differences and exits non-zero if tolerances are exceeded.

## Troubleshooting
- **Missing Python packages** – Activate your virtual environment and reinstall via
  `pip install -r requirements.txt`. `PyYAML` and `g2p-en` are mandatory for the tools.
- **Different checkpoint layout** – Rerun `convert_weights.py`; it embeds the conv kernel
  sizes that the C++ runtime now consumes automatically.
- **Unexpected durations or mel lengths** – Inspect the phoneme IDs passed to the CLI;
  the loader accepts both comma and whitespace separated values after the latest update.
- **Binary fails to open weights** – Confirm `weights/config.bin` and the directory tree
  exist. Conversion must be run before building inference.

## Implementation Notes
- Encoder and decoder stacks each use the FFT blocks from FastSpeech2 (multi-head attention,
  residual connections, and position-wise feed-forward layers).
- Feed-forward modules are implemented as Conv1D layers (kernel sizes taken from the
  converted weights) followed by GELU, matching the PyTorch architecture.
- The variance adaptor implements duration, pitch, and energy predictors with shared
  Conv1D + LayerNorm blocks, followed by a length regulator to produce mel frames.
- PostNet adds five convolutional blocks with batch-norm refinement before writing
  the mel spectrogram.

## License
MIT License — consistent with ming024/FastSpeech2.
