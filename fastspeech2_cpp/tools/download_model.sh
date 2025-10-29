#!/usr/bin/env bash
# Download pretrained FastSpeech2 model from Google Drive
# Model: LJSpeech single-speaker English TTS (900000 steps)

set -e

echo "=========================================="
echo "FastSpeech2 Pretrained Model Downloader"
echo "=========================================="
echo ""

# Configuration
FILE_ID="1r3fYhnblBJ8hDKDSUDtidJ-BN-xAM9pe"
FILE_NAME="900000.pth.tar"
CHECKPOINT_DIR="../output/ckpt/LJSpeech"
TARGET_FILE="$CHECKPOINT_DIR/$FILE_NAME"

# Create directory
mkdir -p "$CHECKPOINT_DIR"

echo "📥 Downloading LJSpeech model (900000 steps)..."
echo "   File ID: $FILE_ID"
echo "   Target: $TARGET_FILE"
echo ""

# Check if file already exists
if [ -f "$TARGET_FILE" ]; then
    FILE_SIZE=$(du -h "$TARGET_FILE" | cut -f1)
    echo "✓ File already exists: $TARGET_FILE"
    echo "  File size: $FILE_SIZE"
    echo ""
    read -p "Do you want to re-download? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download."
        echo ""
        echo "Next step: Convert weights to binary format"
        echo "  python3 tools/convert_weights.py \\"
        echo "    --checkpoint $TARGET_FILE \\"
        echo "    --preprocess_config ../config/LJSpeech/preprocess.yaml \\"
        echo "    --model_config ../config/LJSpeech/model.yaml \\"
        echo "    --output_dir weights/"
        exit 0
    fi
    echo "Re-downloading..."
    rm -f "$TARGET_FILE"
fi

# Try using gdown
if command -v gdown &> /dev/null; then
    echo "✓ Found gdown, attempting download..."
    echo ""

    # Download using gdown
    if gdown "https://drive.google.com/uc?id=$FILE_ID" -O "$TARGET_FILE"; then
        echo ""
        echo "✓ Download successful!"
        FILE_SIZE=$(du -h "$TARGET_FILE" | cut -f1)
        echo "  File: $TARGET_FILE"
        echo "  Size: $FILE_SIZE"
    else
        echo ""
        echo "❌ Download failed with gdown."
        echo ""
        echo "Please try one of these alternatives:"
        echo ""
        echo "1. Download with wget:"
        echo "   wget --no-check-certificate \"https://drive.google.com/uc?export=download&id=$FILE_ID\" -O \"$TARGET_FILE\""
        echo ""
        echo "2. Download with curl:"
        echo "   curl -L \"https://drive.google.com/uc?export=download&id=$FILE_ID\" -o \"$TARGET_FILE\""
        echo ""
        echo "3. Download manually:"
        echo "   https://drive.google.com/file/d/$FILE_ID/view"
        echo "   Save as: $TARGET_FILE"
        exit 1
    fi
else
    echo "❌ gdown not found."
    echo ""
    echo "Please install gdown or download manually:"
    echo ""
    echo "Option 1: Install gdown (recommended)"
    echo "  pip3 install gdown"
    echo "  Then re-run this script."
    echo ""
    echo "Option 2: Download with wget"
    echo "  wget --no-check-certificate \"https://drive.google.com/uc?export=download&id=$FILE_ID\" -O \"$TARGET_FILE\""
    echo ""
    echo "Option 3: Download with curl"
    echo "  curl -L \"https://drive.google.com/uc?export=download&id=$FILE_ID\" -o \"$TARGET_FILE\""
    echo ""
    echo "Option 4: Download manually"
    echo "  1. Go to: https://drive.google.com/file/d/$FILE_ID/view"
    echo "  2. Click 'Download'"
    echo "  3. Save as: $TARGET_FILE"
    exit 1
fi

# Verify download
echo ""
echo "=========================================="
echo "Verification"
echo "=========================================="

if [ -f "$TARGET_FILE" ]; then
    FILE_SIZE=$(du -h "$TARGET_FILE" | cut -f1)
    echo "✓ Checkpoint file verified"
    echo "  Path: $TARGET_FILE"
    echo "  Size: $FILE_SIZE"
    echo ""

    # Check file size (should be around 100-200MB)
    FILE_SIZE_BYTES=$(stat -f%z "$TARGET_FILE" 2>/dev/null || stat -c%s "$TARGET_FILE" 2>/dev/null)
    if [ "$FILE_SIZE_BYTES" -lt 10000000 ]; then
        echo "⚠️  Warning: File size is suspiciously small (< 10MB)"
        echo "   This might be an HTML error page from Google Drive."
        echo "   Please check the file or try manual download."
        echo ""
    fi

    echo "=========================================="
    echo "Next Steps"
    echo "=========================================="
    echo ""
    echo "1. Convert weights to binary format:"
    echo "   cd fastspeech2_cpp/"
    echo "   python3 tools/convert_weights.py \\"
    echo "     --checkpoint $TARGET_FILE \\"
    echo "     --preprocess_config ../config/LJSpeech/preprocess.yaml \\"
    echo "     --model_config ../config/LJSpeech/model.yaml \\"
    echo "     --output_dir weights/"
    echo ""
    echo "2. Build the C++ inference engine:"
    echo "   make build"
    echo ""
    echo "3. Run inference:"
    echo "   ./fastspeech2 \\"
    echo "     --config weights/config.bin \\"
    echo "     --weights weights/ \\"
    echo "     --phonemes \"23,15,8,32,45,12\" \\"
    echo "     --output output.mel"
    echo ""
else
    echo "❌ Checkpoint file not found"
    echo "   Expected: $TARGET_FILE"
    echo ""
    echo "Please download manually:"
    echo "  https://drive.google.com/file/d/$FILE_ID/view"
    echo ""
fi

echo "=========================================="
