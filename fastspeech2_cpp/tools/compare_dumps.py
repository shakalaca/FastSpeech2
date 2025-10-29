#!/usr/bin/env python3
"""Compare PyTorch and C++ intermediate dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


BASE_FLOAT_KEYS = [
    "variance_output",
    "pitch_prediction",
    "energy_prediction",
    "log_duration_prediction",
    "duration_prediction",
    "decoder_output",
    "mel_before",
    "mel_after",
]

INT_KEYS = ["duration_rounded"]


def load_pytorch_dump(directory: Path, key: str) -> np.ndarray:
    path = directory / f"{key}.npy"
    if not path.exists():
        raise FileNotFoundError(f"PyTorch dump missing: {path}")
    return np.load(path)


def load_cpp_dump(directory: Path, key: str, dtype: str) -> np.ndarray:
    path = directory / f"{key}.bin"
    if not path.exists():
        raise FileNotFoundError(f"C++ dump missing: {path}")
    with path.open("rb") as f:
        header = np.frombuffer(f.read(8), dtype=np.int32)
        if header.size != 2:
            raise ValueError(f"Malformed header in {path}")
        rows, cols = int(header[0]), int(header[1])
        array_dtype = np.float32 if dtype == "float" else np.int32
        data = np.frombuffer(f.read(), dtype=array_dtype)
    if data.size != rows * cols:
        raise ValueError(f"Unexpected element count in {path}: {data.size} vs {rows * cols}")
    array = data.reshape(rows, cols)
    if cols == 1:
        array = array.reshape(rows)
    return array


def summarize_diff(ref: np.ndarray, cand: np.ndarray) -> Dict[str, float]:
    diff = cand.astype(np.float64) - ref.astype(np.float64)
    max_abs = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(diff ** 2))) if diff.size > 0 else 0.0
    denom = np.maximum(np.abs(ref.astype(np.float64)), 1e-8)
    rel_max = float(np.max(np.abs(diff) / denom))
    return {"abs_max": max_abs, "rel_max": rel_max, "rms": rms}


def compare_directory(pytorch_dir: Path, cpp_dir: Path, abs_tol: float, rel_tol: float) -> int:
    status = 0

    print("=== Comparing float tensors ===")
    float_keys = ["encoder_embedding"]
    layer_idx = 0
    layer_suffixes = [
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_out",
        "attn_residual",
        "attn_norm",
        "ffn_pre_relu",
        "ffn_post_relu",
        "ffn_out",
        "ffn_residual",
        "",
    ]

    while (pytorch_dir / f"encoder_layer_{layer_idx}.npy").exists() and (
        cpp_dir / f"encoder_layer_{layer_idx}.bin"
    ).exists():
        for suffix in layer_suffixes:
            name = f"encoder_layer_{layer_idx}"
            if suffix:
                name += f"_{suffix}"
            float_keys.append(name)
        layer_idx += 1

    float_keys.append("encoder_output")
    float_keys.extend(BASE_FLOAT_KEYS)

    for key in float_keys:
        pt = load_pytorch_dump(pytorch_dir, key)
        if pt.ndim > 1 and pt.shape[0] == 1:
            pt = np.squeeze(pt, axis=0)
        cpp = load_cpp_dump(cpp_dir, key, dtype="float")
        if pt.shape != cpp.shape:
            print(f"{key}: shape mismatch PyTorch {pt.shape} vs C++ {cpp.shape}")
            status = 1
            continue
        metrics = summarize_diff(pt, cpp)
        print(
            f"{key}: abs_max={metrics['abs_max']:.3e}, rel_max={metrics['rel_max']:.3e}, rms={metrics['rms']:.3e}"
        )
        if metrics["abs_max"] > abs_tol and metrics["rel_max"] > rel_tol:
            status = 1

    print("\n=== Comparing integer tensors ===")
    for key in INT_KEYS:
        pt = load_pytorch_dump(pytorch_dir, key).astype(np.int32)
        if pt.ndim > 1 and pt.shape[0] == 1:
            pt = np.squeeze(pt, axis=0)
        cpp = load_cpp_dump(cpp_dir, key, dtype="int").astype(np.int32)
        if pt.shape != cpp.shape:
            print(f"{key}: shape mismatch PyTorch {pt.shape} vs C++ {cpp.shape}")
            status = 1
            continue
        equal = np.array_equal(pt, cpp)
        print(f"{key}: {'match' if equal else 'mismatch'}")
        if not equal:
            diff = pt - cpp
            print(f"  max abs diff: {np.max(np.abs(diff))}")
            status = 1

    try:
        meta_path = pytorch_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            print("\nPyTorch metadata:", meta)
    except Exception:  # noqa: BLE001
        pass

    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PyTorch and C++ FastSpeech2 dumps")
    parser.add_argument("--pytorch_dir", type=Path, required=True)
    parser.add_argument("--cpp_dir", type=Path, required=True)
    parser.add_argument("--abs_tol", type=float, default=5e-4)
    parser.add_argument("--rel_tol", type=float, default=5e-3)
    args = parser.parse_args()

    status = compare_directory(args.pytorch_dir, args.cpp_dir, args.abs_tol, args.rel_tol)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
