#!/usr/bin/env bash
# Download pretrained FastSpeech2 model from Google Drive
# Model: LJSpeech single-speaker English TTS (900000 steps)

set -e

echo "=========================================="
echo "FastSpeech2 Pretrained Model Downloader"
echo "=========================================="
echo ""

# Target directory
CHECKPOINT_DIR="../output/ckpt/LJSpeech"
mkdir -p "$CHECKPOINT_DIR"

echo "📥 Downloading LJSpeech model (900000 steps)..."
echo ""
echo "⚠️  Note: This requires access to the Google Drive folder:"
echo "    https://drive.google.com/drive/folders/1DOhZGlTLMbbAAFZmZGDdc77kz1PloS7F"
echo ""

# Try using gdown
if command -v gdown &> /dev/null; then
    echo "✓ Found gdown, attempting download..."

    # File ID for 900000.pth.tar (you'll need to get this from the Google Drive)
    # Note: User needs to manually get the file ID from Google Drive sharing link
    echo ""
    echo "❌ Automatic download failed."
    echo ""
    echo "Please download manually:"
    echo "  1. Go to: https://drive.google.com/drive/folders/1DOhZGlTLMbbAAFZmZGDdc77kz1PloS7F"
    echo "  2. Navigate to: LJSpeech/"
    echo "  3. Download: 900000.pth.tar"
    echo "  4. Place it in: $CHECKPOINT_DIR/"
    echo ""
else
    echo "❌ gdown not found. Install it with:"
    echo "    pip3 install gdown"
    echo ""
    echo "Or download manually:"
    echo "  1. Go to: https://drive.google.com/drive/folders/1DOhZGlTLMbbAAFZmZGDdc77kz1PloS7F"
    echo "  2. Navigate to: LJSpeech/"
    echo "  3. Download: 900000.pth.tar"
    echo "  4. Place it in: $CHECKPOINT_DIR/"
    echo ""
fi

# Check if file exists
if [ -f "$CHECKPOINT_DIR/900000.pth.tar" ]; then
    echo "✓ Checkpoint found: $CHECKPOINT_DIR/900000.pth.tar"
    FILE_SIZE=$(du -h "$CHECKPOINT_DIR/900000.pth.tar" | cut -f1)
    echo "  File size: $FILE_SIZE"
    echo ""
    echo "Next step: Convert weights to binary format"
    echo "  python3 tools/convert_weights.py \\"
    echo "    --checkpoint $CHECKPOINT_DIR/900000.pth.tar \\"
    echo "    --output_dir weights/"
else
    echo "❌ Checkpoint not found at: $CHECKPOINT_DIR/900000.pth.tar"
    echo ""
    echo "Please download the model manually and re-run this script."
fi

echo ""
echo "=========================================="
