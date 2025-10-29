#!/usr/bin/env python3
"""
Generate operator-level test data for C++ implementation validation.

This script generates reference outputs for all basic operators using PyTorch,
which will be used to validate the C++ implementations.

Usage:
    python tools/generate_op_test_data.py --output_dir test/data
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path


def save_tensor(tensor, filepath, verbose=True):
    """Save a PyTorch tensor as binary float32 file."""
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    arr.tofile(filepath)
    if verbose:
        print(f"  ✓ {filepath.name}: shape {arr.shape}, range [{arr.min():.4f}, {arr.max():.4f}]")
    return arr


def generate_matmul_data(output_dir):
    """Generate matmul test data: C = A @ B"""
    print("\n1. Generating matmul test data...")

    # Test case 1: Small matrix multiplication (64 x 128) @ (128 x 256)
    torch.manual_seed(42)
    A = torch.randn(64, 128)
    B = torch.randn(128, 256)
    C = torch.matmul(A, B)

    save_tensor(A, output_dir / "matmul_A.bin")
    save_tensor(B, output_dir / "matmul_B.bin")
    save_tensor(C, output_dir / "matmul_C.bin")

    # Save metadata
    with open(output_dir / "matmul_meta.txt", 'w') as f:
        f.write(f"M={A.shape[0]}, K={A.shape[1]}, N={B.shape[1]}\n")
        f.write(f"A: {A.shape}\n")
        f.write(f"B: {B.shape}\n")
        f.write(f"C: {C.shape}\n")


def generate_layer_norm_data(output_dir):
    """Generate layer normalization test data"""
    print("\n2. Generating layer_norm test data...")

    torch.manual_seed(42)
    # Input: (batch=1, seq_len=100, dim=256)
    x = torch.randn(1, 100, 256)

    # Create LayerNorm
    ln = torch.nn.LayerNorm(256)
    torch.manual_seed(123)  # Set seed for initialization
    torch.nn.init.normal_(ln.weight, mean=1.0, std=0.02)
    torch.nn.init.zeros_(ln.bias)

    output = ln(x)

    save_tensor(x, output_dir / "layer_norm_input.bin")
    save_tensor(ln.weight, output_dir / "layer_norm_gamma.bin")
    save_tensor(ln.bias, output_dir / "layer_norm_beta.bin")
    save_tensor(output, output_dir / "layer_norm_output.bin")

    # Save metadata
    with open(output_dir / "layer_norm_meta.txt", 'w') as f:
        f.write(f"input: {x.shape}\n")
        f.write(f"normalized_shape: {ln.normalized_shape}\n")


def generate_softmax_data(output_dir):
    """Generate softmax test data"""
    print("\n3. Generating softmax test data...")

    torch.manual_seed(42)
    # Attention scores: (batch=1, n_heads=2, seq_len=10, seq_len=10)
    x = torch.randn(1, 2, 10, 10)
    output = F.softmax(x, dim=-1)

    save_tensor(x, output_dir / "softmax_input.bin")
    save_tensor(output, output_dir / "softmax_output.bin")

    # Save metadata
    with open(output_dir / "softmax_meta.txt", 'w') as f:
        f.write(f"input: {x.shape}\n")
        f.write(f"dim: -1 (last dimension)\n")


def generate_gelu_data(output_dir):
    """Generate GELU activation test data"""
    print("\n4. Generating GELU test data...")

    torch.manual_seed(42)
    # Test across range including negative and positive values
    x = torch.linspace(-5, 5, 1000)
    output = F.gelu(x)

    save_tensor(x, output_dir / "gelu_input.bin")
    save_tensor(output, output_dir / "gelu_output.bin")

    # Save metadata
    with open(output_dir / "gelu_meta.txt", 'w') as f:
        f.write(f"input: {x.shape}\n")
        f.write(f"range: [{x.min():.2f}, {x.max():.2f}]\n")


def generate_tanh_data(output_dir):
    """Generate tanh activation test data"""
    print("\n5. Generating tanh test data...")

    torch.manual_seed(42)
    x = torch.linspace(-5, 5, 1000)
    output = torch.tanh(x)

    save_tensor(x, output_dir / "tanh_input.bin")
    save_tensor(output, output_dir / "tanh_output.bin")


def generate_conv1d_data(output_dir):
    """Generate Conv1D test data"""
    print("\n6. Generating Conv1D test data...")

    torch.manual_seed(42)
    # Input: (batch=1, in_channels=256, length=100)
    x = torch.randn(1, 256, 100)

    # Conv1D: in_channels=256, out_channels=512, kernel_size=3, padding=1
    conv = torch.nn.Conv1d(256, 512, kernel_size=3, padding=1)
    torch.manual_seed(123)
    torch.nn.init.xavier_uniform_(conv.weight)
    torch.nn.init.zeros_(conv.bias)

    output = conv(x)

    save_tensor(x, output_dir / "conv1d_input.bin")
    save_tensor(conv.weight, output_dir / "conv1d_weight.bin")
    save_tensor(conv.bias, output_dir / "conv1d_bias.bin")
    save_tensor(output, output_dir / "conv1d_output.bin")

    # Save metadata
    with open(output_dir / "conv1d_meta.txt", 'w') as f:
        f.write(f"input: {x.shape}\n")
        f.write(f"weight: {conv.weight.shape}\n")
        f.write(f"kernel_size: {conv.kernel_size}\n")
        f.write(f"padding: {conv.padding}\n")
        f.write(f"output: {output.shape}\n")


def generate_batch_norm_data(output_dir):
    """Generate Batch Normalization test data"""
    print("\n7. Generating Batch Normalization test data...")

    torch.manual_seed(42)
    # Input: (batch=1, channels=512, length=100)
    x = torch.randn(1, 512, 100)

    # BatchNorm1d
    bn = torch.nn.BatchNorm1d(512)
    torch.manual_seed(123)
    torch.nn.init.normal_(bn.weight, mean=1.0, std=0.02)
    torch.nn.init.zeros_(bn.bias)

    # Set running statistics
    bn.running_mean = torch.randn(512)
    bn.running_var = torch.abs(torch.randn(512)) + 0.1

    bn.eval()  # Use eval mode to use running stats
    output = bn(x)

    save_tensor(x, output_dir / "batch_norm_input.bin")
    save_tensor(bn.weight, output_dir / "batch_norm_gamma.bin")
    save_tensor(bn.bias, output_dir / "batch_norm_beta.bin")
    save_tensor(bn.running_mean, output_dir / "batch_norm_mean.bin")
    save_tensor(bn.running_var, output_dir / "batch_norm_var.bin")
    save_tensor(output, output_dir / "batch_norm_output.bin")


def generate_multi_head_attention_data(output_dir):
    """Generate Multi-Head Attention test data"""
    print("\n8. Generating Multi-Head Attention test data...")

    torch.manual_seed(42)
    embed_dim = 256
    num_heads = 2

    # Input: (batch=1, seq_len=10, embed_dim=256)
    Q = torch.randn(1, 10, embed_dim)
    K = torch.randn(1, 10, embed_dim)
    V = torch.randn(1, 10, embed_dim)

    # Create MultiheadAttention
    attn = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    torch.manual_seed(123)

    # Initialize weights
    torch.nn.init.xavier_uniform_(attn.in_proj_weight)
    torch.nn.init.zeros_(attn.in_proj_bias)
    torch.nn.init.xavier_uniform_(attn.out_proj.weight)
    torch.nn.init.zeros_(attn.out_proj.bias)

    attn.eval()
    with torch.no_grad():
        output, attn_weights = attn(Q, K, V)

    save_tensor(Q, output_dir / "mha_query.bin")
    save_tensor(K, output_dir / "mha_key.bin")
    save_tensor(V, output_dir / "mha_value.bin")
    save_tensor(attn.in_proj_weight, output_dir / "mha_in_proj_weight.bin")
    save_tensor(attn.in_proj_bias, output_dir / "mha_in_proj_bias.bin")
    save_tensor(attn.out_proj.weight, output_dir / "mha_out_proj_weight.bin")
    save_tensor(attn.out_proj.bias, output_dir / "mha_out_proj_bias.bin")
    save_tensor(output, output_dir / "mha_output.bin")
    save_tensor(attn_weights, output_dir / "mha_attn_weights.bin")

    # Save metadata
    with open(output_dir / "mha_meta.txt", 'w') as f:
        f.write(f"embed_dim: {embed_dim}\n")
        f.write(f"num_heads: {num_heads}\n")
        f.write(f"Q: {Q.shape}\n")
        f.write(f"K: {K.shape}\n")
        f.write(f"V: {V.shape}\n")
        f.write(f"output: {output.shape}\n")


def generate_positional_encoding_data(output_dir):
    """Generate sinusoidal positional encoding reference"""
    print("\n9. Generating Positional Encoding test data...")

    max_len = 1000
    dim = 256

    # Create sinusoidal positional encoding
    pe = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    save_tensor(pe, output_dir / "positional_encoding.bin")

    # Save metadata
    with open(output_dir / "positional_encoding_meta.txt", 'w') as f:
        f.write(f"max_len: {max_len}\n")
        f.write(f"dim: {dim}\n")
        f.write(f"shape: {pe.shape}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate operator-level test data for C++ validation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test/data",
        help="Output directory for test data"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating operator test data to: {output_dir}")
    print("=" * 70)

    # Generate all test data
    generate_matmul_data(output_dir)
    generate_layer_norm_data(output_dir)
    generate_softmax_data(output_dir)
    generate_gelu_data(output_dir)
    generate_tanh_data(output_dir)
    generate_conv1d_data(output_dir)
    generate_batch_norm_data(output_dir)
    generate_multi_head_attention_data(output_dir)
    generate_positional_encoding_data(output_dir)

    print("\n" + "=" * 70)
    print(f"✓ All operator test data generated successfully!")
    print(f"  Output directory: {output_dir}")
    print(f"  Total files: {len(list(output_dir.glob('*')))}")

    # Print summary
    print("\n📊 Test Data Summary:")
    print("  1. matmul          - Matrix multiplication (64x128) @ (128x256)")
    print("  2. layer_norm      - Layer normalization (1, 100, 256)")
    print("  3. softmax         - Softmax activation (1, 2, 10, 10)")
    print("  4. gelu            - GELU activation (1000 points)")
    print("  5. tanh            - Tanh activation (1000 points)")
    print("  6. conv1d          - 1D convolution (256 → 512 channels)")
    print("  7. batch_norm      - Batch normalization (512 channels)")
    print("  8. multi_head_attn - Multi-head attention (2 heads, dim=256)")
    print("  9. pos_encoding    - Sinusoidal positional encoding (1000, 256)")

    print("\n💡 Next steps:")
    print("  1. Implement operators in op.cpp")
    print("  2. Write test_ops.cpp to validate against this data")
    print("  3. Target: MSE < 1e-5 for all operators")


if __name__ == "__main__":
    main()
