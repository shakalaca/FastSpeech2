#!/usr/bin/env python3
"""
Dump intermediate tensors from the PyTorch FastSpeech2 implementation.

This script mirrors the execution order of the C++ runtime so we can
establish component-level parity. It saves tensor snapshots as .npy
files under the specified output directory.

Example:
    python3 tools/parity/dump_pytorch_intermediates.py \
        --checkpoint ../output/ckpt/LJSpeech/900000.pth.tar \
        --preprocess_config ../config/LJSpeech/preprocess.yaml \
        --model_config ../config/LJSpeech/model.yaml \
        --output_dir test_data/py_ref \
        --phonemes "23,15,8,32,45,12"
"""

from __future__ import annotations

import argparse
import sys
import os
import json
import tarfile
import zipfile
import tempfile
from pathlib import Path
from contextlib import contextmanager
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "model").is_dir():
            return path
    raise RuntimeError(f"Could not locate project root from {start}")


ROOT = find_repo_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@contextmanager
def working_directory(path: Path):
    prev_cwd = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(prev_cwd)


def parse_ids(raw: str) -> List[int]:
    tokens = raw.replace(",", " ").replace(";", " ").split()
    return [int(tok) for tok in tokens if tok.strip()]


def load_ids_from_file(path: Path) -> List[int]:
    if path.suffix == ".bin":
        return np.fromfile(path, dtype=np.int32).astype(int).tolist()
    return parse_ids(path.read_text(encoding="utf-8"))


def ids_from_text(text: str, preprocess_cfg: dict, language: str) -> List[int]:
    from synthesize import preprocess_english, preprocess_mandarin  # noqa: WPS433

    if language == "english":
        seq = preprocess_english(text, preprocess_cfg)
    elif language == "mandarin":
        seq = preprocess_mandarin(text, preprocess_cfg)
    else:
        raise ValueError(f"Unsupported language: {language}")
    return seq.astype(int).tolist()


def materialize_checkpoint(path: Path) -> Tuple[Path, List[tempfile.TemporaryDirectory]]:
    temp_dirs: List[tempfile.TemporaryDirectory] = []
    current = path

    while True:
        if tarfile.is_tarfile(current):
            temp_dir = tempfile.TemporaryDirectory()
            temp_dirs.append(temp_dir)
            with tarfile.open(current, "r:*") as archive:
                members = [m for m in archive.getmembers() if m.isfile()]
                if not members:
                    raise RuntimeError(f"No files found inside checkpoint archive: {current}")
                archive.extract(members[0], temp_dir.name)
                current = Path(temp_dir.name) / members[0].name
            continue

        if zipfile.is_zipfile(current):
            with zipfile.ZipFile(current, "r") as zf:
                names = [name for name in zf.namelist() if not name.endswith("/")]
                if len(names) != 1:
                    break
                temp_dir = tempfile.TemporaryDirectory()
                temp_dirs.append(temp_dir)
                target = names[0]
                zf.extract(target, temp_dir.name)
                current = Path(temp_dir.name) / target
            continue

        break

    return current, temp_dirs


def write_tensor(path: Path, tensor: torch.Tensor) -> None:
    np.save(path, tensor.detach().cpu().numpy())


def write_array(path: Path, array: np.ndarray) -> None:
    np.save(path, array)


def dump_intermediates(
    checkpoint: Path,
    preprocess_cfg: Path,
    model_cfg: Path,
    phoneme_ids: Iterable[int],
    speaker_id: int,
    output_dir: Path,
) -> None:
    checkpoint = checkpoint.resolve()
    preprocess_cfg = preprocess_cfg.resolve()
    model_cfg = model_cfg.resolve()
    output_dir = output_dir.resolve()

    preprocess = yaml.safe_load(preprocess_cfg.read_text())
    model_config = yaml.safe_load(model_cfg.read_text())

    ckpt_path, temp_dirs = materialize_checkpoint(checkpoint)

    try:
        device = torch.device("cpu")

        from model.fastspeech2 import FastSpeech2  # noqa: WPS433
        from utils.tools import get_mask_from_lengths  # noqa: WPS433

        with working_directory(ROOT):
            model = FastSpeech2(preprocess, model_config).to(device)
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state["model"], strict=True)
            model.eval()

            ids = torch.LongTensor(list(phoneme_ids)).unsqueeze(0).to(device)
            src_lens = torch.LongTensor([ids.size(1)]).to(device)
            max_src_len = int(src_lens.max().item())
            src_masks = get_mask_from_lengths(src_lens, max_src_len)

            # 1. Encoder embedding + positional encoding
            embedding = model.encoder.src_word_emb(ids) + model.encoder.position_enc[
                :, :max_src_len, :
            ].expand(ids.size(0), -1, -1)
            write_tensor(output_dir / "encoder_embedding.npy", embedding)

            current = embedding.clone()
            if model.speaker_emb is not None:
                speaker = torch.LongTensor([speaker_id]).to(device)
                current = current + model.speaker_emb(speaker).unsqueeze(1)
            write_tensor(output_dir / "encoder_output_with_speaker.npy", current)

            slf_attn_mask = src_masks.unsqueeze(1).expand(-1, max_src_len, -1)
            for layer_idx, enc_layer in enumerate(model.encoder.layer_stack):
                prefix = output_dir / f"encoder_layer_{layer_idx}"

                q_linear = F.linear(current, enc_layer.slf_attn.w_qs.weight, enc_layer.slf_attn.w_qs.bias)
                k_linear = F.linear(current, enc_layer.slf_attn.w_ks.weight, enc_layer.slf_attn.w_ks.bias)
                v_linear = F.linear(current, enc_layer.slf_attn.w_vs.weight, enc_layer.slf_attn.w_vs.bias)
                write_tensor(prefix.with_name(prefix.name + "_attn_q.npy"), q_linear)
                write_tensor(prefix.with_name(prefix.name + "_attn_k.npy"), k_linear)
                write_tensor(prefix.with_name(prefix.name + "_attn_v.npy"), v_linear)

                n_head = enc_layer.slf_attn.n_head
                d_k = enc_layer.slf_attn.d_k
                d_v = enc_layer.slf_attn.d_v

                q = q_linear.view(1, -1, n_head, d_k).permute(0, 2, 1, 3).contiguous().view(-1, q_linear.size(1), d_k)
                k = k_linear.view(1, -1, n_head, d_k).permute(0, 2, 1, 3).contiguous().view(-1, k_linear.size(1), d_k)
                v = v_linear.view(1, -1, n_head, d_v).permute(0, 2, 1, 3).contiguous().view(-1, v_linear.size(1), d_v)

                attn_mask = slf_attn_mask.repeat(n_head, 1, 1)
                scores = torch.matmul(q, k.transpose(1, 2)) / (d_k ** 0.5)
                scores = scores.masked_fill(attn_mask, float('-inf'))
                attn_weights = torch.softmax(scores, dim=-1)
                context = torch.matmul(attn_weights, v)
                context = context.view(n_head, 1, q_linear.size(1), d_v).permute(1, 2, 0, 3).contiguous().view(1, q_linear.size(1), -1)

                attn_out = F.linear(context, enc_layer.slf_attn.fc.weight, enc_layer.slf_attn.fc.bias)
                attn_out = enc_layer.slf_attn.dropout(attn_out)
                write_tensor(prefix.with_name(prefix.name + "_attn_out.npy"), attn_out)

                attn_residual = attn_out + current
                write_tensor(prefix.with_name(prefix.name + "_attn_residual.npy"), attn_residual)

                attn_norm = enc_layer.slf_attn.layer_norm(attn_residual)
                write_tensor(prefix.with_name(prefix.name + "_attn_norm.npy"), attn_norm)

                ffn_pre = enc_layer.pos_ffn.w_1(attn_norm.transpose(1, 2)).transpose(1, 2)
                write_tensor(prefix.with_name(prefix.name + "_ffn_pre_relu.npy"), ffn_pre)
                ffn_post = torch.relu(ffn_pre)
                write_tensor(prefix.with_name(prefix.name + "_ffn_post_relu.npy"), ffn_post)
                ffn_out = enc_layer.pos_ffn.w_2(ffn_post.transpose(1, 2)).transpose(1, 2)
                ffn_out = enc_layer.pos_ffn.dropout(ffn_out)
                write_tensor(prefix.with_name(prefix.name + "_ffn_out.npy"), ffn_out)
                ffn_residual = ffn_out + attn_norm
                write_tensor(prefix.with_name(prefix.name + "_ffn_residual.npy"), ffn_residual)

                current = enc_layer.pos_ffn.layer_norm(ffn_residual)
                write_tensor(prefix.with_name(prefix.name + ".npy"), current)

            encoder_output = current
            write_tensor(output_dir / "encoder_output.npy", encoder_output)

            # 3. Variance adaptor
            (
                variance_out,
                pitch_predictions,
                energy_predictions,
                log_duration_prediction,
                duration_rounded,
                mel_lens,
                mel_masks,
            ) = model.variance_adaptor(
                encoder_output,
                src_masks,
                mel_mask=None,
                max_len=None,
                pitch_target=None,
                energy_target=None,
                duration_target=None,
                p_control=1.0,
                e_control=1.0,
                d_control=1.0,
            )

            write_tensor(output_dir / "variance_output.npy", variance_out)
            write_tensor(output_dir / "pitch_prediction.npy", pitch_predictions)
            write_tensor(output_dir / "energy_prediction.npy", energy_predictions)
            write_tensor(output_dir / "log_duration_prediction.npy", log_duration_prediction)
            duration_prediction = torch.exp(log_duration_prediction) - 1.0
            write_tensor(output_dir / "duration_prediction.npy", duration_prediction)
            write_array(
                output_dir / "duration_rounded.npy",
                duration_rounded.detach().squeeze(0).cpu().numpy().astype(np.int32),
            )
            write_array(
                output_dir / "mel_lens.npy",
                mel_lens.detach().cpu().numpy().astype(np.int32),
            )

            # 4. Decoder
            decoder_output, _ = model.decoder(variance_out, mel_masks)
            write_tensor(output_dir / "decoder_output.npy", decoder_output)

            # 5. Mel projection + PostNet
            mel_before = model.mel_linear(decoder_output)
            write_tensor(output_dir / "mel_before.npy", mel_before)

            postnet_out = model.postnet(mel_before)
            mel_after = postnet_out + mel_before
            write_tensor(output_dir / "mel_after.npy", mel_after)

            # Metadata for convenience
            meta = {
                "phoneme_ids": list(phoneme_ids),
                "speaker_id": speaker_id,
                "mel_length": int(mel_lens.item()),
                "n_mels": mel_before.size(-1),
            }
            (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    finally:
        for temp_dir in temp_dirs:
            temp_dir.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump intermediate tensors from PyTorch FastSpeech2.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preprocess_config", type=Path, required=True)
    parser.add_argument("--model_config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--phonemes", type=str, help="Comma-separated phoneme IDs.")
    parser.add_argument("--phoneme_file", type=Path, help="File containing phoneme IDs.")
    parser.add_argument("--text", type=str, help="Raw text to synthesize.")
    parser.add_argument(
        "--language",
        type=str,
        default="english",
        choices=["english", "mandarin"],
        help="Language used when converting raw text to phonemes.",
    )
    parser.add_argument("--speaker_id", type=int, default=0)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocess_cfg = args.preprocess_config
    model_cfg = args.model_config
    checkpoint = args.checkpoint

    sources = [bool(args.phonemes), bool(args.phoneme_file), bool(args.text)]
    if sum(sources) != 1:
        parser.error("Specify exactly one of --phonemes, --phoneme_file, or --text.")

    preprocess_data = yaml.safe_load(preprocess_cfg.read_text())

    if args.phonemes:
        phoneme_ids = parse_ids(args.phonemes)
    elif args.phoneme_file:
        phoneme_file = args.phoneme_file
        if not phoneme_file.exists():
            parser.error(f"Phoneme file not found: {phoneme_file}")
        phoneme_ids = load_ids_from_file(phoneme_file)
    else:
        phoneme_ids = ids_from_text(args.text, preprocess_data, args.language)

    if not phoneme_ids:
        parser.error("Resolved phoneme ID list is empty.")

    dump_intermediates(
        checkpoint=checkpoint,
        preprocess_cfg=preprocess_cfg,
        model_cfg=model_cfg,
        phoneme_ids=phoneme_ids,
        speaker_id=args.speaker_id,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
