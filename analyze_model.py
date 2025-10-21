"""
FastSpeech2 Model Analysis Tool

This script analyzes a FastSpeech2 checkpoint and provides detailed statistics
about model architecture, tensor shapes, and parameter counts.

Usage:
    python analyze_model.py --checkpoint output/ckpt/LJSpeech/900000.pth.tar
"""

import argparse
import torch
from pathlib import Path
from collections import defaultdict
import yaml


def load_checkpoint(path: str):
    """Load PyTorch checkpoint"""
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def analyze_state_dict(state_dict):
    """Analyze model state dict"""
    stats = {
        'total_tensors': 0,
        'total_params': 0,
        'components': defaultdict(lambda: {'tensors': 0, 'params': 0, 'details': []}),
        'tensor_list': []
    }

    for name, tensor in state_dict.items():
        # Count tensors
        stats['total_tensors'] += 1
        params = tensor.numel()
        stats['total_params'] += params

        # Determine component
        if name.startswith('encoder.'):
            component = 'Encoder'
        elif name.startswith('variance_adaptor.'):
            if 'duration' in name:
                component = 'Variance/Duration'
            elif 'pitch' in name:
                component = 'Variance/Pitch'
            elif 'energy' in name:
                component = 'Variance/Energy'
            else:
                component = 'Variance/Other'
        elif name.startswith('decoder.'):
            component = 'Decoder'
        elif name.startswith('mel_linear'):
            component = 'MelLinear'
        elif name.startswith('postnet'):
            component = 'PostNet'
        elif name.startswith('speaker_emb'):
            component = 'SpeakerEmb'
        else:
            component = 'Other'

        # Update component stats
        stats['components'][component]['tensors'] += 1
        stats['components'][component]['params'] += params
        stats['components'][component]['details'].append({
            'name': name,
            'shape': list(tensor.shape),
            'params': params,
            'dtype': str(tensor.dtype)
        })

        # Add to tensor list
        stats['tensor_list'].append({
            'name': name,
            'component': component,
            'shape': list(tensor.shape),
            'params': params,
            'dtype': str(tensor.dtype)
        })

    return stats


def print_summary(stats):
    """Print analysis summary"""
    print("\n" + "="*80)
    print("FastSpeech2 Model Analysis Summary")
    print("="*80)

    # Overall stats
    print(f"\n📊 Overall Statistics:")
    print(f"  Total Tensors: {stats['total_tensors']}")
    print(f"  Total Parameters: {stats['total_params']:,}")
    print(f"  Total Size (F32): {stats['total_params'] * 4 / 1024 / 1024:.2f} MB")
    print(f"  Total Size (F16): {stats['total_params'] * 2 / 1024 / 1024:.2f} MB")

    # Component breakdown
    print(f"\n🔧 Component Breakdown:")
    print(f"  {'Component':<20} {'Tensors':<10} {'Parameters':<15} {'Size (F32)':<12} {'Size (F16)':<12}")
    print(f"  {'-'*20} {'-'*10} {'-'*15} {'-'*12} {'-'*12}")

    # Sort by parameter count
    sorted_components = sorted(
        stats['components'].items(),
        key=lambda x: x[1]['params'],
        reverse=True
    )

    for component, info in sorted_components:
        size_f32 = info['params'] * 4 / 1024 / 1024
        size_f16 = info['params'] * 2 / 1024 / 1024
        pct = info['params'] / stats['total_params'] * 100
        print(f"  {component:<20} {info['tensors']:<10} "
              f"{info['params']:<15,} {size_f32:<12.2f} {size_f16:<12.2f} ({pct:.1f}%)")


def print_component_details(stats, component_name):
    """Print detailed tensor information for a component"""
    if component_name not in stats['components']:
        print(f"\n⚠️  Component '{component_name}' not found")
        return

    component = stats['components'][component_name]
    print(f"\n📋 Detailed Tensor List for {component_name}:")
    print(f"  {'Tensor Name':<60} {'Shape':<25} {'Parameters':<12}")
    print(f"  {'-'*60} {'-'*25} {'-'*12}")

    for detail in component['details']:
        shape_str = str(detail['shape'])
        print(f"  {detail['name']:<60} {shape_str:<25} {detail['params']:<12,}")


def print_layer_analysis(stats):
    """Analyze and print transformer layer patterns"""
    print(f"\n🔍 Transformer Layer Analysis:")

    # Analyze encoder layers
    encoder_layers = defaultdict(list)
    decoder_layers = defaultdict(list)

    for tensor in stats['tensor_list']:
        name = tensor['name']
        if 'encoder.layer_stack.' in name:
            # Extract layer number
            parts = name.split('.')
            layer_idx = parts[2] if len(parts) > 2 and parts[2].isdigit() else None
            if layer_idx:
                encoder_layers[int(layer_idx)].append(tensor)
        elif 'decoder.layer_stack.' in name:
            parts = name.split('.')
            layer_idx = parts[2] if len(parts) > 2 and parts[2].isdigit() else None
            if layer_idx:
                decoder_layers[int(layer_idx)].append(tensor)

    # Encoder layer stats
    if encoder_layers:
        print(f"\n  Encoder Layers: {len(encoder_layers)} layers")
        for layer_idx in sorted(encoder_layers.keys()):
            layer_tensors = encoder_layers[layer_idx]
            layer_params = sum(t['params'] for t in layer_tensors)
            print(f"    Layer {layer_idx}: {len(layer_tensors)} tensors, "
                  f"{layer_params:,} parameters ({layer_params * 4 / 1024:.2f} KB)")

    # Decoder layer stats
    if decoder_layers:
        print(f"\n  Decoder Layers: {len(decoder_layers)} layers")
        for layer_idx in sorted(decoder_layers.keys()):
            layer_tensors = decoder_layers[layer_idx]
            layer_params = sum(t['params'] for t in layer_tensors)
            print(f"    Layer {layer_idx}: {len(layer_tensors)} tensors, "
                  f"{layer_params:,} parameters ({layer_params * 4 / 1024:.2f} KB)")


def print_weight_shape_analysis(stats):
    """Analyze weight matrix shapes"""
    print(f"\n📐 Weight Shape Analysis:")

    # Find largest tensors
    sorted_tensors = sorted(stats['tensor_list'], key=lambda x: x['params'], reverse=True)

    print(f"\n  Top 10 Largest Tensors:")
    print(f"  {'Rank':<6} {'Tensor Name':<60} {'Shape':<25} {'Parameters':<12} {'Size (MB)':<10}")
    print(f"  {'-'*6} {'-'*60} {'-'*25} {'-'*12} {'-'*10}")

    for i, tensor in enumerate(sorted_tensors[:10], 1):
        size_mb = tensor['params'] * 4 / 1024 / 1024
        shape_str = str(tensor['shape'])
        print(f"  {i:<6} {tensor['name']:<60} {shape_str:<25} "
              f"{tensor['params']:<12,} {size_mb:<10.2f}")

    # Analyze by dimension
    print(f"\n  Tensor Dimension Distribution:")
    dim_count = defaultdict(int)
    for tensor in stats['tensor_list']:
        ndim = len(tensor['shape'])
        dim_count[ndim] += 1

    for ndim in sorted(dim_count.keys()):
        print(f"    {ndim}D tensors: {dim_count[ndim]}")


def export_to_json(stats, output_path):
    """Export analysis to JSON"""
    import json

    # Prepare data for JSON serialization
    json_data = {
        'summary': {
            'total_tensors': stats['total_tensors'],
            'total_parameters': stats['total_params'],
            'size_f32_mb': stats['total_params'] * 4 / 1024 / 1024,
            'size_f16_mb': stats['total_params'] * 2 / 1024 / 1024,
        },
        'components': {
            name: {
                'tensors': info['tensors'],
                'parameters': info['params'],
                'size_f32_mb': info['params'] * 4 / 1024 / 1024,
                'size_f16_mb': info['params'] * 2 / 1024 / 1024,
            }
            for name, info in stats['components'].items()
        },
        'tensor_list': stats['tensor_list']
    }

    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"\n✓ Exported analysis to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze FastSpeech2 model checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PyTorch checkpoint"
    )
    parser.add_argument(
        "--component",
        type=str,
        help="Show detailed information for specific component"
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export analysis to JSON file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output including layer analysis"
    )

    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = load_checkpoint(args.checkpoint)

    # Analyze
    stats = analyze_state_dict(state_dict)

    # Print summary
    print_summary(stats)

    # Print layer analysis if verbose
    if args.verbose:
        print_layer_analysis(stats)
        print_weight_shape_analysis(stats)

    # Print component details if requested
    if args.component:
        print_component_details(stats, args.component)

    # Export if requested
    if args.export:
        export_to_json(stats, args.export)

    # Print available components
    print(f"\n💡 Available components for --component flag:")
    for component in sorted(stats['components'].keys()):
        print(f"  - {component}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
