#!/usr/bin/env python3
"""Minimal FastSpeech2 weight converter implemented without PyTorch runtime."""

import argparse
import json
import os
import pickle
import struct
import tarfile
import tempfile
import zipfile
from array import array
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import yaml


STORAGE_DTYPES = {
    "torch.FloatStorage": ("f", 4, "float32"),
    "torch.LongStorage": ("q", 8, "int64"),
}


class FakeNumpyDtype:
    """Lightweight stand-in for numpy.dtype used during unpickling."""

    def __init__(self, descriptor, *args, **kwargs):
        self.descriptor = descriptor
        self.args = args
        self.kwargs = kwargs
        self.state = None

    def __setstate__(self, state):
        self.state = state


class Storage:
    """Represents a tensor storage block stored on disk."""

    def __init__(self, file_path: Path, storage_type: str, size: int):
        self.file_path = Path(file_path)
        self.storage_type = storage_type
        self.typecode, self.itemsize, self.dtype = STORAGE_DTYPES[storage_type]
        self.size = size
        self._array: array | None = None

    def load(self) -> array:
        """Materialize the storage into memory."""
        if self._array is None:
            data = self.file_path.read_bytes()
            arr = array(self.typecode)
            arr.frombytes(data)
            if len(arr) != self.size:
                raise ValueError(
                    f"Storage size mismatch for {self.file_path.name}: expected "
                    f"{self.size}, found {len(arr)}"
                )
            self._array = arr
        return self._array


class TensorData:
    """Minimal tensor handle holding metadata and storage reference."""

    def __init__(
        self,
        storage: Storage,
        offset: int,
        size: Iterable[int],
        stride: Iterable[int],
    ):
        self.storage = storage
        self.offset = offset
        self.size = tuple(int(dim) for dim in size)
        self.stride = tuple(int(step) for step in stride)
        self.dtype = storage.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.size

    def numel(self) -> int:
        product = 1
        for dim in self.size:
            product *= dim
        return product

    def is_contiguous(self) -> bool:
        expected = 1
        for dim, stride in zip(reversed(self.size), reversed(self.stride)):
            if dim == 0:
                return True
            if stride != expected:
                return False
            expected *= dim
        return True

    def ensure_materialized(self) -> None:
        self.storage.load()

    def to_float_array(self) -> array:
        """Return tensor flattened as float32 array."""
        if not self.is_contiguous():
            raise ValueError("Non-contiguous tensors are not supported in this converter.")

        data = self.storage.load()
        start = self.offset
        end = start + self.numel()
        segment = data[start:end]

        if self.dtype == "float32":
            return segment

        if self.dtype == "int64":
            return array("f", (float(x) for x in segment))

        raise ValueError(f"Unsupported dtype encountered: {self.dtype}")


class TorchCheckpointUnpickler(pickle.Unpickler):
    """Unpickler that recreates tensors without requiring torch or numpy."""

    def __init__(self, file_obj, storage_dir: Path):
        super().__init__(file_obj)
        self.storage_dir = Path(storage_dir)

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return self._rebuild_tensor_v2

        if module == "torch" and name in {"FloatStorage", "LongStorage"}:
            # The actual storage objects are supplied via persistent IDs; return dummy.
            stub = lambda *args, **kwargs: None  # noqa: E731
            stub.__module__ = module
            stub.__name__ = name
            return stub

        if module == "numpy" and name == "dtype":
            return FakeNumpyDtype

        if module == "numpy.core.multiarray" and name == "scalar":
            return lambda dtype, value, state=None: value

        return super().find_class(module, name)

    def persistent_load(self, pid):
        if isinstance(pid, tuple) and pid and pid[0] == "storage":
            _, storage_type, key, _location, size = pid
            if isinstance(storage_type, str):
                storage_key = storage_type
            else:
                storage_key = f"{getattr(storage_type, '__module__', '')}.{getattr(storage_type, '__name__', '')}"
            if storage_key not in STORAGE_DTYPES:
                raise ValueError(f"Unsupported storage type: {storage_type}")
            return Storage(self.storage_dir / key, storage_key, int(size))
        raise ValueError(f"Unsupported persistent load id: {pid!r}")

    @staticmethod
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, hooks):
        del requires_grad, hooks
        return TensorData(storage, int(storage_offset), size, stride)


def extract_checkpoint(checkpoint_path: Path) -> Tuple[tempfile.TemporaryDirectory, Path]:
    """Extract the checkpoint archive into a temporary directory and return archive root."""
    temp_dir = tempfile.TemporaryDirectory()
    base_path = Path(temp_dir.name)

    path = Path(checkpoint_path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(base_path)
    else:
        with tarfile.open(path, "r:*") as archive:
            archive.extractall(base_path)

    # Some checkpoints nest another tarball; unwrap until data.pkl is found.
    for depth in range(5):
        data_pkl_candidates = list(base_path.rglob("data.pkl"))
        if data_pkl_candidates:
            return temp_dir, data_pkl_candidates[0].parent

        zip_candidates = [
            candidate
            for candidate in base_path.iterdir()
            if candidate.is_file() and zipfile.is_zipfile(candidate)
        ]
        if zip_candidates:
            nested_dir = base_path / f"nested_zip_{depth}"
            nested_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_candidates[0]) as nested_zip:
                nested_zip.extractall(nested_dir)
            base_path = nested_dir
            continue

        tar_candidates = [
            candidate for candidate in base_path.iterdir() if candidate.is_file() and candidate.suffix == ".tar"
        ]
        if tar_candidates:
            nested_dir = base_path / f"nested_tar_{depth}"
            nested_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_candidates[0], "r:*") as nested_tar:
                nested_tar.extractall(nested_dir)
            base_path = nested_dir
            continue

        break

    raise FileNotFoundError("Could not locate data.pkl in the checkpoint archive.")


def load_state_dict(checkpoint_path: Path) -> OrderedDict:
    """Load the model state dict using the custom unpickler."""
    temp_dir, archive_root = extract_checkpoint(checkpoint_path)
    checkpoint = None

    try:
        data_file = archive_root / "data.pkl"
        storage_dir = archive_root / "data"
        if not storage_dir.is_dir():
            raise FileNotFoundError("Missing storage directory in checkpoint archive.")

        with data_file.open("rb") as f:
            unpickler = TorchCheckpointUnpickler(f, storage_dir)
            checkpoint = unpickler.load()
    finally:
        # Ensure storages are loaded before cleaning up the temporary directory.
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model_state = checkpoint["model"]
        elif isinstance(checkpoint, OrderedDict):
            model_state = checkpoint
        else:
            model_state = None

        if isinstance(model_state, OrderedDict):
            for tensor in model_state.values():
                if isinstance(tensor, TensorData):
                    tensor.ensure_materialized()

        temp_dir.cleanup()

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]

    return checkpoint


def save_tensor(tensor: TensorData, path: Path) -> None:
    """Dump a tensor to float32 binary format."""
    if tensor is None:
        return
    flat = tensor.to_float_array()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        flat.tofile(f)


def write_config(preprocess_cfg, model_cfg, stats, out_dir: Path, overrides: Dict[str, int | float | None]):
    transformer = model_cfg["transformer"]
    variance_predictor = model_cfg["variance_predictor"]
    variance_embedding = model_cfg["variance_embedding"]
    postnet_cfg = model_cfg.get("postnet", {})

    config = {
        "dim": transformer["encoder_hidden"],
        "n_enc_layers": transformer["encoder_layer"],
        "n_dec_layers": transformer["decoder_layer"],
        "n_heads": transformer["encoder_head"],
        "head_dim": transformer["encoder_hidden"] // transformer["encoder_head"],
        "ffn_hidden": transformer["conv_filter_size"],
        "vocab_size": model_cfg.get("vocab_size", 300),
        "n_mels": preprocess_cfg["preprocessing"]["mel"]["n_mel_channels"],
        "max_seq_len": model_cfg["max_seq_len"],
        "var_pred_filter_size": variance_predictor["filter_size"],
        "var_pred_kernel_size": variance_predictor["kernel_size"],
        "var_pred_dropout": variance_predictor["dropout"],
        "n_bins": variance_embedding["n_bins"],
        "postnet_embedding_dim": postnet_cfg.get("postnet_embedding_dim", transformer["encoder_hidden"]),
        "postnet_kernel_size": postnet_cfg.get("postnet_kernel_size", 5),
        "postnet_n_convolutions": postnet_cfg.get("postnet_n_convolutions", 5),
        "pitch_min": float(stats["pitch"][0]),
        "pitch_max": float(stats["pitch"][1]),
        "energy_min": float(stats["energy"][0]),
        "energy_max": float(stats["energy"][1]),
    }

    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    int_keys_1 = [
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
    float_keys = ["var_pred_dropout", "pitch_min", "pitch_max", "energy_min", "energy_max"]
    int_keys_2 = ["n_bins", "postnet_embedding_dim", "postnet_kernel_size", "postnet_n_convolutions"]

    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.bin"
    with config_path.open("wb") as f:
        f.write(struct.pack("<" + "i" * len(int_keys_1), *[int(config[k]) for k in int_keys_1]))
        f.write(struct.pack("<" + "f" * len(float_keys), *[float(config[k]) for k in float_keys]))
        f.write(struct.pack("<" + "i" * len(int_keys_2), *[int(config[k]) for k in int_keys_2]))

    print(f"Saved config: {config_path}")
    return config


def dump_variance_predictor(state_dict: OrderedDict, prefix: str, out_dir: Path):
    files = {
        "conv1_weight.bin": f"{prefix}.conv_layer.conv1d_1.conv.weight",
        "conv1_bias.bin": f"{prefix}.conv_layer.conv1d_1.conv.bias",
        "conv2_weight.bin": f"{prefix}.conv_layer.conv1d_2.conv.weight",
        "conv2_bias.bin": f"{prefix}.conv_layer.conv1d_2.conv.bias",
        "ln1_gamma.bin": f"{prefix}.conv_layer.layer_norm_1.weight",
        "ln1_beta.bin": f"{prefix}.conv_layer.layer_norm_1.bias",
        "ln2_gamma.bin": f"{prefix}.conv_layer.layer_norm_2.weight",
        "ln2_beta.bin": f"{prefix}.conv_layer.layer_norm_2.bias",
        "linear_weight.bin": f"{prefix}.linear_layer.weight",
        "linear_bias.bin": f"{prefix}.linear_layer.bias",
    }

    for filename, key in files.items():
        save_tensor(state_dict[key], out_dir / filename)


def dump_fft_layer(state_dict: OrderedDict, prefix: str, out_dir: Path, index: int):
    files = {
        "attn_q_weight.bin": f"{prefix}.{index}.slf_attn.w_qs.weight",
        "attn_q_bias.bin": f"{prefix}.{index}.slf_attn.w_qs.bias",
        "attn_k_weight.bin": f"{prefix}.{index}.slf_attn.w_ks.weight",
        "attn_k_bias.bin": f"{prefix}.{index}.slf_attn.w_ks.bias",
        "attn_v_weight.bin": f"{prefix}.{index}.slf_attn.w_vs.weight",
        "attn_v_bias.bin": f"{prefix}.{index}.slf_attn.w_vs.bias",
        "attn_out_weight.bin": f"{prefix}.{index}.slf_attn.fc.weight",
        "attn_out_bias.bin": f"{prefix}.{index}.slf_attn.fc.bias",
        "attn_norm_gamma.bin": f"{prefix}.{index}.slf_attn.layer_norm.weight",
        "attn_norm_beta.bin": f"{prefix}.{index}.slf_attn.layer_norm.bias",
        "ffn_w1.bin": f"{prefix}.{index}.pos_ffn.w_1.weight",
        "ffn_b1.bin": f"{prefix}.{index}.pos_ffn.w_1.bias",
        "ffn_w2.bin": f"{prefix}.{index}.pos_ffn.w_2.weight",
        "ffn_b2.bin": f"{prefix}.{index}.pos_ffn.w_2.bias",
        "ffn_norm_gamma.bin": f"{prefix}.{index}.pos_ffn.layer_norm.weight",
        "ffn_norm_beta.bin": f"{prefix}.{index}.pos_ffn.layer_norm.bias",
    }

    layer_dir = out_dir / f"layer_{index}"
    for filename, key in files.items():
        save_tensor(state_dict[key], layer_dir / filename)


def dump_postnet_layer(state_dict: OrderedDict, prefix: str, out_dir: Path, index: int):
    files = {
        "conv_weight.bin": f"{prefix}.{index}.0.conv.weight",
        "conv_bias.bin": f"{prefix}.{index}.0.conv.bias",
        "bn_gamma.bin": f"{prefix}.{index}.1.weight",
        "bn_beta.bin": f"{prefix}.{index}.1.bias",
        "bn_mean.bin": f"{prefix}.{index}.1.running_mean",
        "bn_var.bin": f"{prefix}.{index}.1.running_var",
    }

    layer_dir = out_dir / f"layer_{index}"
    for filename, key in files.items():
        save_tensor(state_dict[key], layer_dir / filename)


def convert_model(checkpoint_path: Path, preprocess_cfg, model_cfg, output_dir: Path):
    state_dict = load_state_dict(checkpoint_path)

    stats_path = Path(preprocess_cfg["path"]["preprocessed_path"]) / "stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
    else:
        stats = {"pitch": [0.0, 0.0], "energy": [0.0, 0.0]}

    out_dir = Path(output_dir)
    encoder_dir = out_dir / "encoder"
    variance_dir = out_dir / "variance_adaptor"
    decoder_dir = out_dir / "decoder"
    postnet_dir = out_dir / "postnet"

    for folder in (out_dir, encoder_dir, variance_dir, decoder_dir, postnet_dir):
        folder.mkdir(parents=True, exist_ok=True)

    vocab_size = state_dict["encoder.src_word_emb.weight"].shape[0]

    postnet_layers = [
        key
        for key in state_dict.keys()
        if key.startswith("postnet.convolutions.") and key.endswith(".0.conv.weight")
    ]
    postnet_count = len(postnet_layers)
    postnet_kernel = None
    postnet_dim = None
    if postnet_layers:
        sample = state_dict[postnet_layers[0]]
        postnet_kernel = sample.shape[-1]
        postnet_dim = sample.shape[0] if postnet_count == 1 else state_dict[postnet_layers[1]].shape[0]

    pitch_embedding = state_dict.get("variance_adaptor.pitch_embedding.weight")
    n_bins = pitch_embedding.shape[0] if pitch_embedding else None

    overrides = {
        "vocab_size": vocab_size,
        "postnet_n_convolutions": postnet_count if postnet_count else None,
        "postnet_embedding_dim": postnet_dim,
        "postnet_kernel_size": postnet_kernel,
        "n_bins": n_bins,
    }

    config = write_config(preprocess_cfg, model_cfg, stats, out_dir, overrides)

    save_tensor(state_dict["encoder.src_word_emb.weight"], encoder_dir / "embedding.bin")

    for i in range(config["n_enc_layers"]):
        dump_fft_layer(state_dict, "encoder.layer_stack", encoder_dir, i)

    dump_variance_predictor(state_dict, "variance_adaptor.duration_predictor", variance_dir / "duration_predictor")
    dump_variance_predictor(state_dict, "variance_adaptor.pitch_predictor", variance_dir / "pitch_predictor")
    dump_variance_predictor(state_dict, "variance_adaptor.energy_predictor", variance_dir / "energy_predictor")

    save_tensor(state_dict["variance_adaptor.pitch_embedding.weight"], variance_dir / "pitch_embedding.bin")
    save_tensor(state_dict["variance_adaptor.energy_embedding.weight"], variance_dir / "energy_embedding.bin")

    for i in range(config["n_dec_layers"]):
        dump_fft_layer(state_dict, "decoder.layer_stack", decoder_dir, i)

    save_tensor(state_dict["mel_linear.weight"], out_dir / "mel_linear_weight.bin")
    save_tensor(state_dict["mel_linear.bias"], out_dir / "mel_linear_bias.bin")

    for i in range(config["postnet_n_convolutions"]):
        dump_postnet_layer(state_dict, "postnet.convolutions", postnet_dir, i)

    print(f"Conversion finished. Output directory: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert FastSpeech2 checkpoint to binary weights.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--preprocess_config", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="fastspeech2_cpp/weights")
    args = parser.parse_args()

    with open(args.preprocess_config) as f:
        preprocess_cfg = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.model_config) as f:
        model_cfg = yaml.load(f, Loader=yaml.FullLoader)

    convert_model(Path(args.checkpoint), preprocess_cfg, model_cfg, Path(args.output_dir))


if __name__ == "__main__":
    main()
