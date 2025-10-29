#!/usr/bin/env python3
"""
Convert text to phoneme IDs for FastSpeech2 C++ inference.

Usage:
    python tools/text_to_phonemes.py --text "Hello world" --output phonemes.txt
    python tools/text_to_phonemes.py --input input.txt --output phonemes.txt
"""

import argparse
import sys
import os

# Add parent directory to path to import text module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from text import text_to_sequence
from text.symbols import symbols


def text_to_phoneme_ids(text, cleaner_names=["english_cleaners"]):
    """
    Convert text to phoneme ID sequence.

    Args:
        text: Input text string
        cleaner_names: List of text cleaner functions to apply

    Returns:
        List of phoneme IDs (integers)
    """
    sequence = text_to_sequence(text, cleaner_names)
    return sequence


def save_phoneme_ids(phoneme_ids, output_path):
    """
    Save phoneme IDs to file in two formats:
    1. Human-readable text format (.txt): space-separated integers
    2. Binary format (.bin): int32 array
    """
    # Save as text file
    txt_path = output_path if output_path.endswith('.txt') else output_path + '.txt'
    with open(txt_path, 'w') as f:
        f.write(' '.join(map(str, phoneme_ids)) + '\n')
    print(f"✓ Saved text format to {txt_path}")

    # Save as binary file for C++
    bin_path = output_path.replace('.txt', '.bin') if output_path.endswith('.txt') else output_path + '.bin'
    import numpy as np
    np.array(phoneme_ids, dtype=np.int32).tofile(bin_path)
    print(f"✓ Saved binary format to {bin_path}")

    return phoneme_ids


def print_phoneme_mapping(phoneme_ids):
    """Print phoneme IDs with their corresponding symbols."""
    from text.symbols import symbols

    print("\nPhoneme mapping:")
    print(f"  Total phonemes: {len(phoneme_ids)}")
    print(f"  IDs: {phoneme_ids}")
    print(f"  Symbols: ", end="")
    for pid in phoneme_ids:
        if pid < len(symbols):
            print(f"{symbols[pid]}", end=" ")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Convert text to phoneme IDs for FastSpeech2 C++ inference"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text",
        type=str,
        help="Input text string"
    )
    input_group.add_argument(
        "--input",
        type=str,
        help="Input text file (one sentence per line)"
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: print to stdout)"
    )

    # Cleaner options
    parser.add_argument(
        "--cleaners",
        type=str,
        nargs='+',
        default=["english_cleaners"],
        help="Text cleaner functions to apply (default: english_cleaners)"
    )

    # Verbose option
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print phoneme mapping details"
    )

    args = parser.parse_args()

    # Process input
    if args.text:
        texts = [args.text]
    else:
        with open(args.input, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(texts)} text(s)...")

    # Convert each text
    all_results = []
    for i, text in enumerate(texts):
        print(f"\n[{i+1}/{len(texts)}] Text: \"{text}\"")

        phoneme_ids = text_to_phoneme_ids(text, args.cleaners)

        if args.verbose:
            print_phoneme_mapping(phoneme_ids)
        else:
            print(f"  Phoneme IDs ({len(phoneme_ids)}): {phoneme_ids}")

        all_results.append({
            'text': text,
            'phoneme_ids': phoneme_ids
        })

    # Save output
    if args.output:
        if len(texts) == 1:
            # Single text: save as simple format
            save_phoneme_ids(all_results[0]['phoneme_ids'], args.output)
        else:
            # Multiple texts: save with metadata
            import json
            output_dir = os.path.dirname(args.output) or '.'
            base_name = os.path.splitext(os.path.basename(args.output))[0]

            # Save metadata
            meta_path = os.path.join(output_dir, f"{base_name}_meta.json")
            with open(meta_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n✓ Saved metadata to {meta_path}")

            # Save each phoneme sequence
            for i, result in enumerate(all_results):
                seq_path = os.path.join(output_dir, f"{base_name}_{i}")
                save_phoneme_ids(result['phoneme_ids'], seq_path)

    # Print vocabulary info
    print(f"\n📊 Vocabulary info:")
    print(f"  Total symbols: {len(symbols)}")
    print(f"  Symbols: {symbols[:20]}... (showing first 20)")

    return all_results


if __name__ == "__main__":
    main()
