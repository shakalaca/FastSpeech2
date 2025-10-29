#!/usr/bin/env python3
"""Compare FastSpeech2 PyTorch inference against the C++ runtime.

Given a checkpoint and converted weight directory, this script runs a single
utterance through both implementations and reports the numerical differences
between the mel-spectrogram outputs.

Example:
    python3 tools/parity/compare_runtime.py \
        --checkpoint ../output/ckpt/LJSpeech/900000.pth.tar \
        --preprocess_config ../config/LJSpeech/preprocess.yaml \
        --model_config ../config/LJSpeech/model.yaml \
        --weights_dir weights \
        --runtime ./fastspeech2 \
        --phonemes "23,15,8,32,45,12"
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
import yaml


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "model").is_dir():
            return path
    raise RuntimeError(f"Could not locate project root from {start}")


ROOT = find_repo_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.fastspeech2 import FastSpeech2  # noqa: E402


@contextmanager
def working_directory(path: Path):
    """Temporarily switch the current working directory."""
    prev_cwd = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(prev_cwd)


@contextmanager
def materialize_checkpoint(path: Path):
    """Yield a checkpoint path, extracting archives if necessary."""
    temp_dir = None
    target_path = path

    def pick_tar_member(members):
        files = [m for m in members if m.isfile()]
        if not files:
            return None
        preferred = [m for m in files if m.name.endswith((".pth", ".pt", ".bin"))]
        return preferred[0] if preferred else files[0]

    if tarfile.is_tarfile(path):
        temp_dir = tempfile.TemporaryDirectory()
        with tarfile.open(path, "r:*") as archive:
            member = pick_tar_member(archive.getmembers())
            if member is None:
                raise RuntimeError(f"No checkpoint file found inside tar archive: {path}")
            archive.extract(member, temp_dir.name)
            target_path = Path(temp_dir.name) / member.name
    elif zipfile.is_zipfile(path):
        temp_dir = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            candidate = None
            for name in names:
                if name.endswith((".pth", ".pt", ".bin")):
                    candidate = name
                    break
            if candidate is None:
                candidate = names[0]
            archive.extract(candidate, temp_dir.name)
            target_path = Path(temp_dir.name) / candidate

    try:
        yield target_path
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

def parse_ids(raw: str) -> List[int]:
    tokens = raw.replace(",", " ").replace(";", " ").split()
    return [int(tok) for tok in tokens]


def load_ids_from_file(path: Path) -> List[int]:
    if path.suffix == ".bin":
        data = np.fromfile(path, dtype=np.int32)
        return data.astype(int).tolist()
    contents = path.read_text(encoding="utf-8")
    return parse_ids(contents)


def ids_from_text(
    text: str,
    preprocess_cfg: dict,
    language: str,
) -> List[int]:
    from synthesize import preprocess_english, preprocess_mandarin  # noqa: WPS433

    if language == "english":
        sequence = preprocess_english(text, preprocess_cfg)
    elif language == "mandarin":
        sequence = preprocess_mandarin(text, preprocess_cfg)
    else:
        raise ValueError(f"Unsupported language: {language}")
    return sequence.astype(int).tolist()


def run_cpp_runtime(
    runtime: Path,
    config_bin: Path,
    weights_dir: Path,
    phoneme_ids: Iterable[int],
) -> Tuple[np.ndarray, int, int]:
    phoneme_str = ",".join(str(v) for v in phoneme_ids)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        output_path = Path(tmp.name)
    try:
        cmd = [
            str(runtime),
            "--config",
            str(config_bin),
            "--weights",
            str(weights_dir),
            "--phonemes",
            phoneme_str,
            "--output",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        with output_path.open("rb") as f:
            header = f.read(8)
            if len(header) != 8:
                raise RuntimeError("Output file truncated (missing shape header)")
            mel_len, n_mels = struct.unpack("<2i", header)
            payload = f.read()
        mel = np.frombuffer(payload, dtype=np.float32)
        if mel.size != mel_len * n_mels:
            raise RuntimeError(
                f"Malformed mel payload: expected {mel_len * n_mels} floats, "
                f"found {mel.size}"
            )
        mel = mel.reshape(mel_len, n_mels)
        return mel, mel_len, n_mels
    finally:
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass


def run_pytorch_model(
    checkpoint: Path,
    preprocess_cfg_path: Path,
    model_cfg_path: Path,
    phoneme_ids: Iterable[int],
    speaker_id: int,
    device: torch.device,
) -> Tuple[np.ndarray, int, int]:
    preprocess_cfg = yaml.safe_load(preprocess_cfg_path.read_text())
    model_cfg = yaml.safe_load(model_cfg_path.read_text())

    with working_directory(ROOT), materialize_checkpoint(checkpoint) as ckpt_path:
        model = FastSpeech2(preprocess_cfg, model_cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()

        ids = torch.LongTensor(list(phoneme_ids)).unsqueeze(0).to(device)
        src_lens = torch.LongTensor([ids.size(1)]).to(device)
        max_src_len = int(src_lens.max().item())

        speakers = torch.LongTensor([speaker_id]).to(device)

        with torch.no_grad():
            (
                _,
                mel_after,
                _,
                _,
                log_duration_predictions,
                duration_rounded,
                _,
                mel_masks,
                _,
                mel_lens,
            ) = model(
                speakers,
                ids,
                src_lens,
                max_src_len,
                p_control=1.0,
                e_control=1.0,
                d_control=1.0,
            )

    mel_len = int(mel_lens.squeeze(0).item())
    mel = mel_after[0, :mel_len].cpu().numpy().astype(np.float32)
    durations = duration_rounded[0, : ids.size(1)].cpu().numpy().astype(np.float32)
    log_durations = log_duration_predictions[0, : ids.size(1)].cpu().numpy().astype(np.float32)
    return mel, mel_len, mel.shape[1], durations, log_durations


def summarize_difference(
    ref: np.ndarray,
    cand: np.ndarray,
) -> Tuple[float, float, float]:
    diff = cand.astype(np.float64) - ref.astype(np.float64)
    abs_max = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(diff ** 2)))
    denom = np.maximum(np.abs(ref), 1e-8)
    rel_max = float(np.max(np.abs(diff) / denom))
    return abs_max, rel_max, rms


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate C++ FastSpeech2 runtime against PyTorch."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preprocess_config", type=Path, required=True)
    parser.add_argument("--model_config", type=Path, required=True)
    parser.add_argument("--weights_dir", type=Path, default=Path("weights"))
    parser.add_argument("--config_bin", type=Path, default=None)
    parser.add_argument("--runtime", type=Path, default=Path("./fastspeech2"))
    parser.add_argument("--phonemes", type=str, help="Comma-separated phoneme IDs.")
    parser.add_argument("--phoneme_file", type=Path, help="File containing phoneme IDs.")
    parser.add_argument("--text", type=str, help="Raw text to synthesize.")
    parser.add_argument(
        "--language",
        type=str,
        default="english",
        choices=["english", "mandarin"],
        help="Language used when converting raw text to phoneme IDs.",
    )
    parser.add_argument("--speaker_id", type=int, default=0)
    parser.add_argument(
        "--abs_tolerance",
        type=float,
        default=5e-4,
        help="Maximum allowed absolute difference.",
    )
    parser.add_argument(
        "--rel_tolerance",
        type=float,
        default=5e-3,
        help="Maximum allowed relative difference.",
    )
    args = parser.parse_args()

    # Normalize paths relative to current working directory before we change dirs.
    cwd = Path.cwd()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else (cwd / args.checkpoint)
    preprocess_config = args.preprocess_config if args.preprocess_config.is_absolute() else (cwd / args.preprocess_config)
    model_config = args.model_config if args.model_config.is_absolute() else (cwd / args.model_config)
    weights_dir = args.weights_dir if args.weights_dir.is_absolute() else (cwd / args.weights_dir)
    runtime = args.runtime if args.runtime.is_absolute() else (cwd / args.runtime)
    config_bin = args.config_bin
    if config_bin is None:
        config_bin = weights_dir / "config.bin"
    elif not config_bin.is_absolute():
        config_bin = cwd / config_bin

    if not runtime.exists():
        parser.error(f"C++ runtime not found: {runtime}")

    if not checkpoint.exists():
        parser.error(f"Checkpoint not found: {checkpoint}")
    if not preprocess_config.exists():
        parser.error(f"Preprocess config not found: {preprocess_config}")
    if not model_config.exists():
        parser.error(f"Model config not found: {model_config}")
    if not weights_dir.exists():
        parser.error(f"Weights directory not found: {weights_dir}")
    if not config_bin.exists():
        parser.error(f"config.bin not found: {config_bin}")

    sources = [bool(args.phonemes), bool(args.phoneme_file), bool(args.text)]
    if sum(sources) != 1:
        parser.error("Specify exactly one of --phonemes, --phoneme_file, or --text.")

    preprocess_cfg = yaml.safe_load(preprocess_config.read_text())

    if args.phonemes:
        phoneme_ids = parse_ids(args.phonemes)
    elif args.phoneme_file:
        phoneme_file = args.phoneme_file if args.phoneme_file.is_absolute() else (cwd / args.phoneme_file)
        if not phoneme_file.exists():
            parser.error(f"Phoneme file not found: {phoneme_file}")
        phoneme_ids = load_ids_from_file(phoneme_file)
    else:
        phoneme_ids = ids_from_text(args.text, preprocess_cfg, args.language)

    if not phoneme_ids:
        parser.error("No phoneme IDs resolved from the provided input.")

    print("============================================")
    print("FastSpeech2 Runtime Comparison")
    print("============================================")
    print(f"Checkpoint:      {checkpoint.resolve()}")
    print(f"Weights dir:     {weights_dir.resolve()}")
    print(f"C++ binary:      {runtime.resolve()}")
    print(f"Phoneme IDs:     {phoneme_ids}")
    print("--------------------------------------------")

    device = torch.device("cpu")
    torch_mel, torch_len, torch_bins, torch_durations, torch_log_durations = run_pytorch_model(
        checkpoint,
        preprocess_config,
        model_config,
        phoneme_ids,
        args.speaker_id,
        device,
    )

    cpp_mel, cpp_len, cpp_bins = run_cpp_runtime(
        runtime, config_bin, weights_dir, phoneme_ids
    )

    print(f"PyTorch mel shape: {torch_mel.shape} (len={torch_len}, bins={torch_bins})")
    print(f"C++ mel shape:     {cpp_mel.shape} (len={cpp_len}, bins={cpp_bins})")

    if torch_mel.shape != cpp_mel.shape:
        print("❌ Shape mismatch between runtimes.")
        print(f"PyTorch durations: {[int(d) for d in torch_durations.tolist()]}")
        print(f"PyTorch log durations: {torch_log_durations.tolist()}")
        print(f"Total frames (PyTorch): {torch_len}")
        return 2

    abs_max, rel_max, rms = summarize_difference(torch_mel, cpp_mel)

    print("--------------------------------------------")
    print(f"Max abs diff:     {abs_max:.6e}")
    print(f"Max rel diff:     {rel_max:.6e}")
    print(f"RMS diff:         {rms:.6e}")
    print("--------------------------------------------")

    thresholds_met = abs_max <= args.abs_tolerance and rel_max <= args.rel_tolerance

    if thresholds_met:
        print("✅ Outputs match within tolerance.")
        return 0

    print(
        "❌ Differences exceed tolerance "
        f"(abs>{args.abs_tolerance:.1e} or rel>{args.rel_tolerance:.1e})."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
