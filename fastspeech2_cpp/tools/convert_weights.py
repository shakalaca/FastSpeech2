#!/usr/bin/env python3
"""
Convert PyTorch FastSpeech2 model weights to binary format for C++ inference.

Usage:
    python tools/convert_weights.py --checkpoint <path_to_checkpoint.pth.tar> \\
                                     --preprocess_config <path_to_preprocess.yaml> \\
                                     --model_config <path_to_model.yaml> \\
                                     --output_dir <output_directory>
"""

import argparse
import os
import json
import yaml
import torch
import numpy as np
from pathlib import Path


def save_tensor(tensor, filepath):
    """Save a PyTorch tensor as binary float32 file."""
    if tensor is None:
        return
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    arr.tofile(filepath)
    print(f"  Saved {filepath.name}: {arr.shape}")


def save_config(preprocess_config, model_config, stats, output_dir):
    """Save model configuration as binary file."""
    config_path = os.path.join(output_dir, "config.bin")

    # Extract configuration values
    transformer = model_config["transformer"]
    variance_predictor = model_config["variance_predictor"]
    variance_embedding = model_config["variance_embedding"]

    # Prepare config struct (matches C++ Config structure)
    config = {
        "dim": transformer["encoder_hidden"],
        "n_enc_layers": transformer["encoder_layer"],
        "n_dec_layers": transformer["decoder_layer"],
        "n_heads": transformer["encoder_head"],
        "head_dim": transformer["encoder_hidden"] // transformer["encoder_head"],
        "ffn_hidden": transformer["conv_filter_size"],

        "vocab_size": 300,  # Will be updated if we can determine actual size
        "n_mels": preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        "max_seq_len": model_config["max_seq_len"],

        "var_pred_filter_size": variance_predictor["filter_size"],
        "var_pred_kernel_size": variance_predictor["kernel_size"],
        "var_pred_dropout": variance_predictor["dropout"],
        "n_bins": variance_embedding["n_bins"],

        "postnet_embedding_dim": model_config["postnet"]["postnet_embedding_dim"],
        "postnet_kernel_size": model_config["postnet"]["postnet_kernel_size"],
        "postnet_n_convolutions": model_config["postnet"]["postnet_n_convolutions"],

        "pitch_min": float(stats["pitch"][0]),
        "pitch_max": float(stats["pitch"][1]),
        "energy_min": float(stats["energy"][0]),
        "energy_max": float(stats["energy"][1]),
    }

    # Save as binary file (int and float values)
    with open(config_path, "wb") as f:
        # Write integers (11 ints)
        for key in ["dim", "n_enc_layers", "n_dec_layers", "n_heads", "head_dim",
                    "ffn_hidden", "vocab_size", "n_mels", "max_seq_len",
                    "var_pred_filter_size", "var_pred_kernel_size"]:
            f.write(np.int32(config[key]).tobytes())

        # Write floats (5 floats: var_pred_dropout + 4 stats)
        for key in ["var_pred_dropout", "pitch_min", "pitch_max",
                    "energy_min", "energy_max"]:
            f.write(np.float32(config[key]).tobytes())

        # Write remaining ints (2 ints)
        for key in ["n_bins", "postnet_embedding_dim", "postnet_kernel_size",
                    "postnet_n_convolutions"]:
            f.write(np.int32(config[key]).tobytes())

    print(f"✓ Saved config to {config_path}")
    print(f"  Config: dim={config['dim']}, enc_layers={config['n_enc_layers']}, "
          f"dec_layers={config['n_dec_layers']}, n_mels={config['n_mels']}")

    return config


def convert_variance_predictor(state_dict, prefix, output_dir, name):
    """Convert a variance predictor's weights."""
    vp_dir = output_dir / name
    vp_dir.mkdir(exist_ok=True)

    print(f"\n  Converting {name}...")

    # Conv layers
    save_tensor(state_dict[f"{prefix}.conv_layer.0.conv.weight"],
                vp_dir / "conv1_weight.bin")
    save_tensor(state_dict[f"{prefix}.conv_layer.0.conv.bias"],
                vp_dir / "conv1_bias.bin")
    save_tensor(state_dict[f"{prefix}.conv_layer.1.conv.weight"],
                vp_dir / "conv2_weight.bin")
    save_tensor(state_dict[f"{prefix}.conv_layer.1.conv.bias"],
                vp_dir / "conv2_bias.bin")

    # Layer norms
    save_tensor(state_dict[f"{prefix}.conv_layer.0.layer_norm.weight"],
                vp_dir / "ln1_gamma.bin")
    save_tensor(state_dict[f"{prefix}.conv_layer.0.layer_norm.bias"],
                vp_dir / "ln1_beta.bin")
    save_tensor(state_dict[f"{prefix}.conv_layer.1.layer_norm.weight"],
                vp_dir / "ln2_gamma.bin")
    save_tensor(state_dict[f"{prefix}.conv_layer.1.layer_norm.bias"],
                vp_dir / "ln2_beta.bin")

    # Linear projection
    save_tensor(state_dict[f"{prefix}.linear_layer.weight"],
                vp_dir / "linear_weight.bin")
    save_tensor(state_dict[f"{prefix}.linear_layer.bias"],
                vp_dir / "linear_bias.bin")


def convert_fft_layer(state_dict, prefix, output_dir, layer_idx):
    """Convert an encoder/decoder FFT layer."""
    layer_dir = output_dir / f"layer_{layer_idx}"
    layer_dir.mkdir(exist_ok=True)

    # Multi-Head Attention
    # Note: PyTorch MultiheadAttention uses in_proj_weight (combined Q, K, V)
    # We need to split it
    in_proj_weight = state_dict[f"{prefix}.{layer_idx}.slf_attn.w_qs.weight"]
    dim = in_proj_weight.shape[0]

    save_tensor(state_dict[f"{prefix}.{layer_idx}.slf_attn.w_qs.weight"],
                layer_dir / "attn_q_weight.bin")
    save_tensor(state_dict[f"{prefix}.{layer_idx}.slf_attn.w_qs.bias"],
                layer_dir / "attn_q_bias.bin")
    save_tensor(state_dict[f"{prefix}.{layer_idx}.slf_attn.w_ks.weight"],
                layer_dir / "attn_k_weight.bin")
    save_tensor(state_dict[f"{prefix}.{layer_idx}.slf_attn.w_ks.bias"],
                layer_dir / "attn_k_bias.bin")
    save_tensor(state_dict[f"{prefix}.{layer_idx}.slf_attn.w_vs.weight"],
                layer_dir / "attn_v_weight.bin")
    save_tensor(state_dict[f"{prefix}.{layer_idx}.slf_attn.w_vs.bias"],
                layer_dir / "attn_v_bias.bin")
    save_tensor(state_dict[f"{prefix}.{layer_idx}.slf_attn.fc.weight"],
                layer_dir / "attn_out_weight.bin")
    save_tensor(state_dict[f"{prefix}.{layer_idx}.slf_attn.fc.bias"],
                layer_dir / "attn_out_bias.bin")
    save_tensor(state_dict[f"{prefix}.{layer_idx}.slf_attn.layer_norm.weight"],
                layer_dir / "attn_norm_gamma.bin")
    save_tensor(state_dict[f"{prefix}.{layer_idx}.slf_attn.layer_norm.bias"],
                layer_dir / "attn_norm_beta.bin")

    # Feed-Forward Network
    save_tensor(state_dict[f"{prefix}.{layer_idx}.pos_ffn.w_1.weight"],
                layer_dir / "ffn_w1.bin")
    save_tensor(state_dict[f"{prefix}.{layer_idx}.pos_ffn.w_1.bias"],
                layer_dir / "ffn_b1.bin")
    save_tensor(state_dict[f"{prefix}.{layer_idx}.pos_ffn.w_2.weight"],
                layer_dir / "ffn_w2.bin")
    save_tensor(state_dict[f"{prefix}.{layer_idx}.pos_ffn.w_2.bias"],
                layer_dir / "ffn_b2.bin")
    save_tensor(state_dict[f"{prefix}.{layer_idx}.pos_ffn.layer_norm.weight"],
                layer_dir / "ffn_norm_gamma.bin")
    save_tensor(state_dict[f"{prefix}.{layer_idx}.pos_ffn.layer_norm.bias"],
                layer_dir / "ffn_norm_beta.bin")


def convert_postnet_layer(state_dict, prefix, output_dir, layer_idx):
    """Convert a PostNet layer."""
    layer_dir = output_dir / f"layer_{layer_idx}"
    layer_dir.mkdir(exist_ok=True)

    save_tensor(state_dict[f"{prefix}.{layer_idx}.0.weight"],
                layer_dir / "conv_weight.bin")
    save_tensor(state_dict[f"{prefix}.{layer_idx}.0.bias"],
                layer_dir / "conv_bias.bin")

    # Batch normalization
    save_tensor(state_dict[f"{prefix}.{layer_idx}.1.weight"],
                layer_dir / "bn_gamma.bin")
    save_tensor(state_dict[f"{prefix}.{layer_idx}.1.bias"],
                layer_dir / "bn_beta.bin")
    save_tensor(state_dict[f"{prefix}.{layer_idx}.1.running_mean"],
                layer_dir / "bn_mean.bin")
    save_tensor(state_dict[f"{prefix}.{layer_idx}.1.running_var"],
                layer_dir / "bn_var.bin")


def convert_model(checkpoint_path, preprocess_config, model_config, output_dir):
    """Convert complete FastSpeech2 model."""
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"]

    print(f"Model keys: {len(state_dict.keys())}")

    # Load stats
    stats_path = os.path.join(
        preprocess_config["path"]["preprocessed_path"], "stats.json"
    )
    with open(stats_path) as f:
        stats = json.load(f)

    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    encoder_dir = output_path / "encoder"
    encoder_dir.mkdir(exist_ok=True)

    variance_dir = output_path / "variance_adaptor"
    variance_dir.mkdir(exist_ok=True)

    decoder_dir = output_path / "decoder"
    decoder_dir.mkdir(exist_ok=True)

    postnet_dir = output_path / "postnet"
    postnet_dir.mkdir(exist_ok=True)

    # Save configuration
    config = save_config(preprocess_config, model_config, stats, output_dir)

    print("\n=== Converting Encoder ===")
    # Encoder embedding
    save_tensor(state_dict["encoder.src_word_emb.weight"],
                encoder_dir / "embedding.bin")

    # Encoder positional encoding (generated, not learned)
    print("  Note: Positional encoding will be generated in C++")

    # Encoder layers
    n_enc_layers = config["n_enc_layers"]
    for i in range(n_enc_layers):
        print(f"  Converting encoder layer {i}...")
        convert_fft_layer(state_dict, "encoder.layer_stack", encoder_dir, i)

    print("\n=== Converting Variance Adaptor ===")
    convert_variance_predictor(state_dict, "variance_adaptor.duration_predictor",
                               variance_dir, "duration_predictor")
    convert_variance_predictor(state_dict, "variance_adaptor.pitch_predictor",
                               variance_dir, "pitch_predictor")
    convert_variance_predictor(state_dict, "variance_adaptor.energy_predictor",
                               variance_dir, "energy_predictor")

    # Pitch and energy embeddings
    save_tensor(state_dict["variance_adaptor.pitch_embedding.weight"],
                variance_dir / "pitch_embedding.bin")
    save_tensor(state_dict["variance_adaptor.energy_embedding.weight"],
                variance_dir / "energy_embedding.bin")

    print("\n=== Converting Decoder ===")
    # Decoder layers
    n_dec_layers = config["n_dec_layers"]
    for i in range(n_dec_layers):
        print(f"  Converting decoder layer {i}...")
        convert_fft_layer(state_dict, "decoder.layer_stack", decoder_dir, i)

    # Mel linear projection
    save_tensor(state_dict["mel_linear.weight"],
                output_path / "mel_linear_weight.bin")
    save_tensor(state_dict["mel_linear.bias"],
                output_path / "mel_linear_bias.bin")

    print("\n=== Converting PostNet ===")
    n_postnet_layers = config["postnet_n_convolutions"]
    for i in range(n_postnet_layers):
        print(f"  Converting PostNet layer {i}...")
        convert_postnet_layer(state_dict, "postnet.conv_list", postnet_dir, i)

    print(f"\n✓ Conversion complete! Weights saved to {output_dir}")
    print(f"\nDirectory structure:")
    print(f"  {output_dir}/")
    print(f"    ├── config.bin")
    print(f"    ├── encoder/")
    print(f"    ├── variance_adaptor/")
    print(f"    ├── decoder/")
    print(f"    └── postnet/")


def main():
    parser = argparse.ArgumentParser(
        description="Convert FastSpeech2 PyTorch weights to C++ binary format"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/ckpt/LJSpeech/900000.pth.tar",
        help="Path to PyTorch checkpoint file",
    )
    parser.add_argument(
        "--preprocess_config",
        type=str,
        default="config/LJSpeech/preprocess.yaml",
        help="Path to preprocess config YAML",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="config/LJSpeech/model.yaml",
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="fastspeech2_cpp/weights",
        help="Output directory for converted weights",
    )

    args = parser.parse_args()

    # Load configs
    with open(args.preprocess_config) as f:
        preprocess_config = yaml.load(f, Loader=yaml.FullLoader)

    with open(args.model_config) as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)

    # Convert model
    convert_model(args.checkpoint, preprocess_config, model_config, args.output_dir)


if __name__ == "__main__":
    main()
