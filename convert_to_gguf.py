"""
FastSpeech2 to GGUF Conversion Script

This script converts a trained FastSpeech2 PyTorch model to GGUF format
for efficient inference and quantization support.

Usage:
    python convert_to_gguf.py \\
        --checkpoint output/ckpt/LJSpeech/900000.pth.tar \\
        --model_config config/LJSpeech/model.yaml \\
        --preprocess_config config/LJSpeech/preprocess.yaml \\
        --output fastspeech2_ljspeech.gguf \\
        --file_type f16
"""

import argparse
import struct
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import json

try:
    from gguf import GGUFWriter, GGUFValueType
    GGUF_AVAILABLE = True
except ImportError:
    print("Warning: gguf library not available. Install with: pip install gguf")
    GGUF_AVAILABLE = False


# GGUF data type mapping
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8


class FastSpeech2ToGGUFConverter:
    """Convert FastSpeech2 PyTorch model to GGUF format"""

    def __init__(
        self,
        checkpoint_path: str,
        model_config_path: str,
        preprocess_config_path: str,
        output_path: str,
        file_type: str = "f16"
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.model_config_path = Path(model_config_path)
        self.preprocess_config_path = Path(preprocess_config_path)
        self.output_path = Path(output_path)
        self.file_type = file_type.lower()

        # Load configs
        self.model_config = self._load_yaml(model_config_path)
        self.preprocess_config = self._load_yaml(preprocess_config_path)

        # Load checkpoint
        self.checkpoint = self._load_checkpoint(checkpoint_path)

        # Initialize mapping
        self._init_name_mapping()

    @staticmethod
    def _load_yaml(path: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def _load_checkpoint(path: str) -> Dict[str, torch.Tensor]:
        """Load PyTorch checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            return checkpoint['model']
        return checkpoint

    def _init_name_mapping(self):
        """Initialize PyTorch to GGUF name mapping"""
        self.name_map = {}

        # Encoder
        self.name_map["encoder.src_word_emb.weight"] = "encoder.token_emb.weight"
        self.name_map["encoder.position_enc"] = "encoder.pos_enc.weight"

        # Encoder layers
        for i in range(self.model_config["transformer"]["encoder_layer"]):
            self._add_layer_mapping("encoder", i)

        # Variance Adaptor
        self._add_variance_mapping()

        # Decoder
        self.name_map["decoder.position_enc"] = "decoder.pos_enc.weight"
        for i in range(self.model_config["transformer"]["decoder_layer"]):
            self._add_layer_mapping("decoder", i)

        # Mel Linear
        self.name_map["mel_linear.weight"] = "mel_linear.weight"
        self.name_map["mel_linear.bias"] = "mel_linear.bias"

        # PostNet
        self._add_postnet_mapping()

        # Speaker Embedding (if multi-speaker)
        if self.model_config.get("multi_speaker", False):
            self.name_map["speaker_emb.weight"] = "speaker_emb.weight"

    def _add_layer_mapping(self, prefix: str, layer_idx: int):
        """Add transformer layer name mappings"""
        pt_prefix = f"{prefix}.layer_stack.{layer_idx}"
        gguf_prefix = f"{prefix}.layers.{layer_idx}"

        # Attention
        mappings = {
            f"{pt_prefix}.slf_attn.w_qs.weight": f"{gguf_prefix}.attn.q_proj.weight",
            f"{pt_prefix}.slf_attn.w_qs.bias": f"{gguf_prefix}.attn.q_proj.bias",
            f"{pt_prefix}.slf_attn.w_ks.weight": f"{gguf_prefix}.attn.k_proj.weight",
            f"{pt_prefix}.slf_attn.w_ks.bias": f"{gguf_prefix}.attn.k_proj.bias",
            f"{pt_prefix}.slf_attn.w_vs.weight": f"{gguf_prefix}.attn.v_proj.weight",
            f"{pt_prefix}.slf_attn.w_vs.bias": f"{gguf_prefix}.attn.v_proj.bias",
            f"{pt_prefix}.slf_attn.fc.weight": f"{gguf_prefix}.attn.out_proj.weight",
            f"{pt_prefix}.slf_attn.fc.bias": f"{gguf_prefix}.attn.out_proj.bias",
            f"{pt_prefix}.slf_attn.layer_norm.weight": f"{gguf_prefix}.attn.layer_norm.weight",
            f"{pt_prefix}.slf_attn.layer_norm.bias": f"{gguf_prefix}.attn.layer_norm.bias",
        }

        # FFN
        mappings.update({
            f"{pt_prefix}.pos_ffn.w_1.weight": f"{gguf_prefix}.ffn.conv1.weight",
            f"{pt_prefix}.pos_ffn.w_1.bias": f"{gguf_prefix}.ffn.conv1.bias",
            f"{pt_prefix}.pos_ffn.w_2.weight": f"{gguf_prefix}.ffn.conv2.weight",
            f"{pt_prefix}.pos_ffn.w_2.bias": f"{gguf_prefix}.ffn.conv2.bias",
            f"{pt_prefix}.pos_ffn.layer_norm.weight": f"{gguf_prefix}.ffn.layer_norm.weight",
            f"{pt_prefix}.pos_ffn.layer_norm.bias": f"{gguf_prefix}.ffn.layer_norm.bias",
        })

        self.name_map.update(mappings)

    def _add_variance_mapping(self):
        """Add variance adaptor name mappings"""
        for predictor_type in ["duration", "pitch", "energy"]:
            pt_prefix = f"variance_adaptor.{predictor_type}_predictor"
            gguf_prefix = f"variance.{predictor_type}"

            # Conv layers
            self.name_map.update({
                f"{pt_prefix}.conv_layer.conv1d_1.conv.weight": f"{gguf_prefix}.conv.0.weight",
                f"{pt_prefix}.conv_layer.conv1d_1.conv.bias": f"{gguf_prefix}.conv.0.bias",
                f"{pt_prefix}.conv_layer.layer_norm_1.weight": f"{gguf_prefix}.norm.0.weight",
                f"{pt_prefix}.conv_layer.layer_norm_1.bias": f"{gguf_prefix}.norm.0.bias",
                f"{pt_prefix}.conv_layer.conv1d_2.conv.weight": f"{gguf_prefix}.conv.1.weight",
                f"{pt_prefix}.conv_layer.conv1d_2.conv.bias": f"{gguf_prefix}.conv.1.bias",
                f"{pt_prefix}.conv_layer.layer_norm_2.weight": f"{gguf_prefix}.norm.1.weight",
                f"{pt_prefix}.conv_layer.layer_norm_2.bias": f"{gguf_prefix}.norm.1.bias",
                f"{pt_prefix}.linear_layer.weight": f"{gguf_prefix}.linear.weight",
                f"{pt_prefix}.linear_layer.bias": f"{gguf_prefix}.linear.bias",
            })

        # Pitch/Energy embeddings and bins
        self.name_map.update({
            "variance_adaptor.pitch_bins": "variance.pitch.bins",
            "variance_adaptor.pitch_embedding.weight": "variance.pitch.emb.weight",
            "variance_adaptor.energy_bins": "variance.energy.bins",
            "variance_adaptor.energy_embedding.weight": "variance.energy.emb.weight",
        })

    def _add_postnet_mapping(self):
        """Add PostNet name mappings"""
        for i in range(5):  # 5 convolution layers
            self.name_map.update({
                f"postnet.convolutions.{i}.0.conv.weight": f"postnet.convs.{i}.conv.weight",
                f"postnet.convolutions.{i}.0.conv.bias": f"postnet.convs.{i}.conv.bias",
                f"postnet.convolutions.{i}.1.weight": f"postnet.convs.{i}.bn.weight",
                f"postnet.convolutions.{i}.1.bias": f"postnet.convs.{i}.bn.bias",
                f"postnet.convolutions.{i}.1.running_mean": f"postnet.convs.{i}.bn.running_mean",
                f"postnet.convolutions.{i}.1.running_var": f"postnet.convs.{i}.bn.running_var",
            })

    def _should_use_f32(self, name: str) -> bool:
        """Determine if tensor should remain in F32"""
        # Keep bins in high precision
        if "bins" in name:
            return True
        # Keep layer norms in high precision (optional)
        if "layer_norm" in name or ".bn." in name:
            return False  # Can use F16
        return False

    def _should_quantize(self, name: str) -> bool:
        """Determine if tensor should be quantized"""
        # For now, only support F32/F16
        # Quantization can be added later
        return False

    def _convert_tensor(self, tensor: torch.Tensor, name: str) -> np.ndarray:
        """Convert tensor to appropriate data type"""
        # Detach and move to CPU
        tensor = tensor.detach().cpu()

        # Determine target dtype
        if self.file_type == "f32" or self._should_use_f32(name):
            return tensor.float().numpy()
        elif self.file_type == "f16":
            return tensor.half().numpy()
        else:
            # For quantization, would implement here
            raise NotImplementedError(f"Quantization type {self.file_type} not yet implemented")

    def add_metadata(self, writer):
        """Add all metadata to GGUF file"""
        # General metadata
        writer.add_string("general.architecture", "fastspeech2")
        writer.add_string("general.name", f"FastSpeech2-{self.preprocess_config['dataset']}")
        writer.add_string("general.description", "FastSpeech2 Text-to-Speech Model")
        writer.add_string("general.license", "MIT")
        writer.add_string("general.author", "ming024")

        # File type
        file_type_map = {"f32": 0, "f16": 1}
        writer.add_uint32("general.file_type", file_type_map.get(self.file_type, 1))
        writer.add_uint32("general.alignment", 32)

        # Encoder config
        enc_config = self.model_config["transformer"]
        writer.add_uint32("fastspeech2.encoder.layers", enc_config["encoder_layer"])
        writer.add_uint32("fastspeech2.encoder.hidden_size", enc_config["encoder_hidden"])
        writer.add_uint32("fastspeech2.encoder.attention.heads", enc_config["encoder_head"])
        writer.add_uint32("fastspeech2.encoder.attention.head_dim",
                         enc_config["encoder_hidden"] // enc_config["encoder_head"])
        writer.add_uint32("fastspeech2.encoder.ffn.hidden_size", enc_config["conv_filter_size"])
        writer.add_array("fastspeech2.encoder.ffn.kernel_size", enc_config["conv_kernel_size"])
        writer.add_float32("fastspeech2.encoder.dropout", enc_config["encoder_dropout"])
        writer.add_uint32("fastspeech2.encoder.max_seq_len", self.model_config["max_seq_len"])

        # Decoder config
        writer.add_uint32("fastspeech2.decoder.layers", enc_config["decoder_layer"])
        writer.add_uint32("fastspeech2.decoder.hidden_size", enc_config["decoder_hidden"])
        writer.add_uint32("fastspeech2.decoder.attention.heads", enc_config["decoder_head"])
        writer.add_uint32("fastspeech2.decoder.attention.head_dim",
                         enc_config["decoder_hidden"] // enc_config["decoder_head"])
        writer.add_uint32("fastspeech2.decoder.ffn.hidden_size", enc_config["conv_filter_size"])
        writer.add_float32("fastspeech2.decoder.dropout", enc_config["decoder_dropout"])

        # Variance predictor config
        var_config = self.model_config["variance_predictor"]
        var_emb_config = self.model_config["variance_embedding"]

        for predictor_type in ["duration", "pitch", "energy"]:
            prefix = f"fastspeech2.variance.{predictor_type}"
            writer.add_uint32(f"{prefix}.filter_size", var_config["filter_size"])
            writer.add_uint32(f"{prefix}.kernel_size", var_config["kernel_size"])
            writer.add_float32(f"{prefix}.dropout", var_config["dropout"])

        # Pitch/Energy specific
        writer.add_string("fastspeech2.variance.pitch.quantization",
                         var_emb_config["pitch_quantization"])
        writer.add_uint32("fastspeech2.variance.pitch.n_bins", var_emb_config["n_bins"])
        writer.add_string("fastspeech2.variance.pitch.feature_level",
                         self.preprocess_config["preprocessing"]["pitch"]["feature"])
        writer.add_bool("fastspeech2.variance.pitch.normalization",
                       self.preprocess_config["preprocessing"]["pitch"]["normalization"])

        writer.add_string("fastspeech2.variance.energy.quantization",
                         var_emb_config["energy_quantization"])
        writer.add_uint32("fastspeech2.variance.energy.n_bins", var_emb_config["n_bins"])
        writer.add_string("fastspeech2.variance.energy.feature_level",
                         self.preprocess_config["preprocessing"]["energy"]["feature"])
        writer.add_bool("fastspeech2.variance.energy.normalization",
                       self.preprocess_config["preprocessing"]["energy"]["normalization"])

        # Audio config
        audio_config = self.preprocess_config["preprocessing"]
        writer.add_uint32("fastspeech2.audio.sampling_rate", audio_config["audio"]["sampling_rate"])
        writer.add_float32("fastspeech2.audio.max_wav_value", audio_config["audio"]["max_wav_value"])
        writer.add_uint32("fastspeech2.audio.n_mel_channels", audio_config["mel"]["n_mel_channels"])
        writer.add_uint32("fastspeech2.audio.mel_fmin", audio_config["mel"]["mel_fmin"])
        mel_fmax = audio_config["mel"]["mel_fmax"]
        if mel_fmax is not None:
            writer.add_uint32("fastspeech2.audio.mel_fmax", mel_fmax)
        writer.add_uint32("fastspeech2.audio.filter_length", audio_config["stft"]["filter_length"])
        writer.add_uint32("fastspeech2.audio.hop_length", audio_config["stft"]["hop_length"])
        writer.add_uint32("fastspeech2.audio.win_length", audio_config["stft"]["win_length"])

        # PostNet config
        writer.add_uint32("fastspeech2.postnet.n_convolutions", 5)
        writer.add_uint32("fastspeech2.postnet.embedding_dim", 512)
        writer.add_uint32("fastspeech2.postnet.kernel_size", 5)

        # Multi-speaker
        multi_speaker = self.model_config.get("multi_speaker", False)
        writer.add_bool("fastspeech2.multi_speaker", multi_speaker)
        if multi_speaker:
            # Try to determine number of speakers from checkpoint
            if "speaker_emb.weight" in self.checkpoint:
                n_speakers = self.checkpoint["speaker_emb.weight"].shape[0]
                writer.add_uint32("fastspeech2.n_speakers", n_speakers)

        # Vocabulary
        # Note: This requires importing text.symbols
        try:
            from text.symbols import symbols
            writer.add_uint32("fastspeech2.vocab.size", len(symbols) + 1)
            writer.add_uint32("fastspeech2.vocab.pad_token_id", 0)
            writer.add_string("fastspeech2.vocab.language",
                            self.preprocess_config["preprocessing"]["text"]["language"])
        except ImportError:
            print("Warning: Could not import text.symbols for vocabulary size")

    def add_tensors(self, writer):
        """Add all model tensors to GGUF file"""
        converted_count = 0
        skipped_tensors = []

        for pt_name, tensor in self.checkpoint.items():
            # Map name
            if pt_name in self.name_map:
                gguf_name = self.name_map[pt_name]
            else:
                # Try to handle unmapped names
                print(f"Warning: No mapping for {pt_name}, skipping")
                skipped_tensors.append(pt_name)
                continue

            # Convert tensor
            try:
                tensor_np = self._convert_tensor(tensor, gguf_name)
                writer.add_tensor(gguf_name, tensor_np)
                converted_count += 1
                print(f"Converted: {pt_name} -> {gguf_name} | Shape: {tensor_np.shape}")
            except Exception as e:
                print(f"Error converting {pt_name}: {e}")
                skipped_tensors.append(pt_name)

        print(f"\nConversion summary:")
        print(f"  Converted: {converted_count} tensors")
        print(f"  Skipped: {len(skipped_tensors)} tensors")
        if skipped_tensors:
            print(f"  Skipped tensor names: {skipped_tensors}")

    def convert(self):
        """Main conversion function"""
        if not GGUF_AVAILABLE:
            raise RuntimeError("gguf library not available. Install with: pip install gguf")

        print(f"Converting FastSpeech2 to GGUF format...")
        print(f"  Checkpoint: {self.checkpoint_path}")
        print(f"  Output: {self.output_path}")
        print(f"  File type: {self.file_type}")

        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create GGUF writer
        writer = GGUFWriter(str(self.output_path), "fastspeech2")

        # Add metadata
        print("\nAdding metadata...")
        self.add_metadata(writer)

        # Add tensors
        print("\nAdding tensors...")
        self.add_tensors(writer)

        # Write to file
        print("\nWriting GGUF file...")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        print(f"\n✓ Conversion complete: {self.output_path}")
        print(f"  File size: {self.output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert FastSpeech2 PyTorch model to GGUF format"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PyTorch checkpoint (.pth.tar)"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to model config YAML"
    )
    parser.add_argument(
        "--preprocess_config",
        type=str,
        required=True,
        help="Path to preprocess config YAML"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output GGUF file path"
    )
    parser.add_argument(
        "--file_type",
        type=str,
        default="f16",
        choices=["f32", "f16"],
        help="Output data type (default: f16)"
    )

    args = parser.parse_args()

    # Create converter and run
    converter = FastSpeech2ToGGUFConverter(
        checkpoint_path=args.checkpoint,
        model_config_path=args.model_config,
        preprocess_config_path=args.preprocess_config,
        output_path=args.output,
        file_type=args.file_type
    )

    converter.convert()


if __name__ == "__main__":
    main()
