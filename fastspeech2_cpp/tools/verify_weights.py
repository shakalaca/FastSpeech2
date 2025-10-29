#!/usr/bin/env python3
"""Utility helpers to validate FastSpeech2 binary weights.

Two primary modes are supported:

* Single directory sanity check (scan for NaN/Inf, report ranges).
  Example:
      python3 tools/verify_weights.py --reference weights

* One-to-one comparison between a freshly converted directory and a
  known-good reference directory.
  Example:
      python3 tools/verify_weights.py \
          --reference weights \
          --candidate /tmp/new_weights \
          --tolerance 1e-5
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


INT_KEYS_1 = [
    "dim",
    "n_enc_layers",
    "n_dec_layers",
    "n_heads",
    "head_dim",
    "ffn_hidden",
    "vocab_size",
    "n_mels",
    "max_seq_len",
    "var_pred_filter_size",
    "var_pred_kernel_size",
]
FLOAT_KEYS = ["var_pred_dropout", "pitch_min", "pitch_max", "energy_min", "energy_max"]
INT_KEYS_2 = ["n_bins", "postnet_embedding_dim", "postnet_kernel_size", "postnet_n_convolutions"]


@dataclass
class WeightStats:
    min: float
    max: float
    mean: float
    std: float
    count: int


def load_config(path: Path) -> Dict[str, float | int]:
    """Load config.bin using the same layout as the C++ runtime."""
    with path.open("rb") as f:
        ints1 = struct.unpack("<11i", f.read(11 * 4))
        floats = struct.unpack("<5f", f.read(5 * 4))
        ints2 = struct.unpack("<4i", f.read(4 * 4))

    config: Dict[str, float | int] = {}
    config.update(dict(zip(INT_KEYS_1, ints1)))
    config.update(dict(zip(FLOAT_KEYS, floats)))
    config.update(dict(zip(INT_KEYS_2, ints2)))
    return config


def load_tensor(path: Path) -> np.ndarray:
    """Load a binary tensor as float32 array."""
    return np.fromfile(path, dtype=np.float32)


def tensor_stats(tensor: np.ndarray) -> WeightStats:
    return WeightStats(
        min=float(tensor.min()),
        max=float(tensor.max()),
        mean=float(tensor.mean()),
        std=float(tensor.std()),
        count=int(tensor.size),
    )


def scan_for_issues(directory: Path) -> Tuple[Dict[str, WeightStats], Dict[str, str]]:
    """Walk through *.bin files, computing stats and locating problems."""
    stats: Dict[str, WeightStats] = {}
    errors: Dict[str, str] = {}

    for path in sorted(directory.rglob("*.bin")):
        if path.name == "config.bin":
            continue
        rel = str(path.relative_to(directory))
        try:
            values = load_tensor(path)
        except FileNotFoundError:
            errors[rel] = "File not found"
            continue
        except OSError as exc:
            errors[rel] = f"I/O error: {exc}"
            continue

        if values.size == 0:
            errors[rel] = "Empty tensor"
            continue

        if np.isnan(values).any():
            errors[rel] = "Contains NaN"
            continue

        if np.isinf(values).any():
            errors[rel] = "Contains Inf"
            continue

        stats[rel] = tensor_stats(values)

    return stats, errors


def compare_configs(ref_path: Path, cand_path: Path) -> Tuple[bool, Dict[str, Tuple[float, float]]]:
    ref = load_config(ref_path)
    cand = load_config(cand_path)

    mismatches: Dict[str, Tuple[float, float]] = {}
    for key in sorted(ref.keys()):
        if key not in cand or ref[key] != cand[key]:
            mismatches[key] = (ref[key], cand.get(key))

    return not mismatches, mismatches


def compare_tensors(
    ref_dir: Path,
    cand_dir: Path,
    tolerance: float,
    rtol: float,
) -> Tuple[Dict[str, WeightStats], Dict[str, str], Dict[str, Dict[str, float]]]:
    """Compare every tensor shared between two directories."""
    ref_files = {
        path.relative_to(ref_dir) for path in ref_dir.rglob("*.bin") if path.name != "config.bin"
    }
    cand_files = {
        path.relative_to(cand_dir) for path in cand_dir.rglob("*.bin") if path.name != "config.bin"
    }

    missing = {str(p): "Missing in candidate" for p in sorted(ref_files - cand_files)}
    extra = {str(p): "Missing in reference" for p in sorted(cand_files - ref_files)}

    stats: Dict[str, WeightStats] = {}
    errors: Dict[str, str] = {}
    deltas: Dict[str, Dict[str, float]] = {}

    for rel_path in sorted(ref_files & cand_files):
        ref_tensor = load_tensor(ref_dir / rel_path)
        cand_tensor = load_tensor(cand_dir / rel_path)

        rel_str = str(rel_path)

        if ref_tensor.shape != cand_tensor.shape:
            errors[rel_str] = f"Shape mismatch: ref {ref_tensor.shape} vs cand {cand_tensor.shape}"
            continue

        diff = cand_tensor.astype(np.float64) - ref_tensor.astype(np.float64)
        abs_max = float(np.max(np.abs(diff)))
        rms = float(np.sqrt(np.mean(diff ** 2)))
        denom = np.maximum(np.abs(ref_tensor), 1e-8)
        rel_max = float(np.max(np.abs(diff) / denom))

        deltas[rel_str] = {"abs_max": abs_max, "rel_max": rel_max, "rms": rms}
        stats[rel_str] = tensor_stats(ref_tensor)

        if abs_max > tolerance and rel_max > rtol:
            errors[rel_str] = (
                f"Difference (abs_max={abs_max:.2e}, rel_max={rel_max:.2e}) exceeds "
                f"tolerance (tol={tolerance:.2e}, rtol={rtol:.2e})"
            )

    errors.update(missing)
    errors.update(extra)
    return stats, errors, deltas


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def single_directory_report(directory: Path) -> bool:
    print_header("Weights Sanity Check")
    print(f"Directory: {directory.resolve()}")

    stats, errors = scan_for_issues(directory)
    if errors:
        print("\nFound issues:")
        for rel, message in errors.items():
            print(f"  ✗ {rel}: {message}")
    else:
        print("\n✓ No NaN/Inf detected and all tensors are non-empty.")

    print("\nSample statistics:")
    for rel in sorted(stats.keys())[:5]:
        s = stats[rel]
        print(
            f"  {rel}: count={s.count}, range=[{s.min:.6f}, {s.max:.6f}], "
            f"mean={s.mean:.6f}, std={s.std:.6f}"
        )

    print("\nSummary:")
    print(f"  Total tensors scanned: {len(stats)}")
    print(f"  Issues detected: {len(errors)}")
    print("\n" + "=" * 70 + "\n")
    return not errors


def comparison_report(reference: Path, candidate: Path, tolerance: float, rtol: float) -> bool:
    print_header("Weights Comparison")
    print(f"Reference: {reference.resolve()}")
    print(f"Candidate: {candidate.resolve()}")
    print(f"Tolerances: abs={tolerance:.2e}, rel={rtol:.2e}")

    ok_config, config_mismatches = compare_configs(reference / "config.bin", candidate / "config.bin")
    if ok_config:
        print("\n✓ config.bin matches exactly.")
    else:
        print("\n✗ config.bin mismatches detected:")
        for key, (ref_val, cand_val) in config_mismatches.items():
            print(f"  {key}: reference={ref_val}, candidate={cand_val}")

    stats, errors, deltas = compare_tensors(reference, candidate, tolerance, rtol)

    if errors:
        print("\nDifferences / missing files:")
        for rel, message in sorted(errors.items()):
            print(f"  ✗ {rel}: {message}")
    else:
        print("\n✓ All tensors match within tolerance.")

    print("\nLargest deviations (top 10 by abs_max):")
    sorted_deltas = sorted(
        deltas.items(), key=lambda item: item[1]["abs_max"], reverse=True
    )[:10]
    for rel, metrics in sorted_deltas:
        stat = stats.get(rel)
        if not stat:
            continue
        print(
            f"  {rel}: abs_max={metrics['abs_max']:.3e}, rel_max={metrics['rel_max']:.3e}, "
            f"rms={metrics['rms']:.3e}, range=[{stat.min:.3e}, {stat.max:.3e}]"
        )

    clean = ok_config and not errors
    print("\nSummary:")
    print(f"  Files compared: {len(deltas)}")
    print(f"  Issues detected: {len(errors)}")
    print(f"  Config match: {'yes' if ok_config else 'no'}")
    print("\n" + ("✅ Match confirmed!" if clean else "❌ Differences found!") + "\n")
    return clean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate converted FastSpeech2 weights.")
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("weights"),
        help="Path to the known-good weights directory.",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        help="Path to the newly converted weights directory. "
             "If omitted, only the reference directory is sanity checked.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Absolute tolerance for elementwise comparison.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for elementwise comparison.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    reference = args.reference
    candidate = args.candidate

    if not reference.exists():
        print(f"Reference directory not found: {reference}")
        return 1

    if candidate:
        if not candidate.exists():
            print(f"Candidate directory not found: {candidate}")
            return 1
        success = comparison_report(reference, candidate, args.tolerance, args.rtol)
    else:
        success = single_directory_report(reference)

    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
