"""FPP-Net: independent reference implementation of the final paper model.

Dependencies: numpy, scipy, torch. No project-specific imports or data paths.
Raw input: (batch, channels, samples), floating-point original measurements.
KAIST: channels=1, samples=5120, sampling_rate=25600 Hz.
CEITEC: channels=3, samples=1710, sampling_rate=10000 Hz.
The caller performs dataset selection, speed screening and record/domain splits.

Example (training):
    features = spectral_features(raw_windows, sampling_rate=25600)
    model = FPPNet(raw_channels=1)
    logits = model(torch.from_numpy(features))
    loss = nn.functional.cross_entropy(logits, labels)  # labels: 0 healthy, 1 fault
    loss.backward()

Example (trained inference):
    models = [load_checkpoint(path) for path in checkpoint_paths]
    probabilities = predict_windows(models, raw_windows, sampling_rate=25600)
    record_ids, record_probabilities = aggregate_records(probabilities, records)

CLI examples:
    python fpp_net.py --demo
    python fpp_net.py --input windows.npy --fs 25600 --checkpoint model.pt
    python fpp_net.py --input windows.npy --fs 10000 --checkpoint a.pt b.pt c.pt

The demo uses RANDOM weights and only checks execution. Published accuracy
requires the trained checkpoints and the paper's exact evaluation partitions.
Core provenance: research/frontend_binary_v25.py::spectral and
research/position_context_v26.py::Compact with head_mode='position'.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
from torch import nn

__all__ = ["FPPNet", "spectral_features", "load_checkpoint",
           "predict_windows", "aggregate_records"]


def spectral_features(raw: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Return float32 relative log power + local residual, shape (B, 2*C, 1024).

    Preserve the source array's floating-point precision through mean removal
    to match the original preprocessing. FFT/interpolation use NumPy precision.
    Gaussian smoothing: 50 Hz standard deviation, reflective boundaries.
    This NumPy/SciPy frontend is not differentiable; train the PyTorch network
    using its precomputed output. No population/target statistics are fitted.
    """
    raw = np.asarray(raw)
    if raw.ndim != 3 or min(raw.shape) < 1 or raw.shape[-1] < 3:
        raise ValueError("raw must have shape (batch, channels, samples>=3)")
    if not np.issubdtype(raw.dtype, np.floating) or not np.isfinite(raw).all():
        raise ValueError("raw must contain finite real floating-point samples")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be finite and positive")
    centered = raw - raw.mean(axis=-1, keepdims=True)
    n = centered.shape[-1]
    power = abs(np.fft.rfft(centered * np.hanning(n), axis=-1)) ** 2
    frequencies = np.fft.rfftfreq(n, 1 / sampling_rate)
    grid = np.linspace(sampling_rate / (2 * 1024), sampling_rate / 2, 1024)
    interpolated = np.array([
        np.interp(grid, frequencies, row)
        for row in power.reshape(-1, power.shape[-1])
    ]).reshape(*power.shape[:2], 1024)
    interpolated /= interpolated.mean((1, 2), keepdims=True).clip(1e-20)
    log_power = np.log(interpolated.clip(1e-8))
    residual = log_power - gaussian_filter1d(
        log_power, 50 / (sampling_rate / 2048), axis=-1, mode="reflect"
    )
    features = np.clip(np.concatenate([log_power, residual], axis=1) / 5, -8, 8)
    if not np.isfinite(features).all():
        raise ValueError("Spectral computation overflowed; check input magnitudes")
    return features.astype(np.float32)


class FPPNet(nn.Module):
    """Two convolutions and ordered readout; returns two-class logits.

    raw_channels is the number of SENSOR channels (1 or 3), not the number
    of spectral channels (2 or 6). Input to forward is (B, 2*C, 1024).
    Parameters: 17098 + 112*C; 17,210 for KAIST, 17,434 for CEITEC.
    State-dict keys match the original position-head Compact implementation.
    """

    def __init__(self, raw_channels: int = 1):
        super().__init__()
        if not isinstance(raw_channels, int) or raw_channels < 1:
            raise ValueError("raw_channels must be a positive integer")
        self.raw_channels = raw_channels
        self.encoder = nn.Sequential(
            nn.Conv1d(2 * raw_channels, 8, kernel_size=7, stride=2, padding=3),
            nn.SiLU(),
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.AvgPool1d(4),
        )
        self.head = nn.Sequential(
            nn.Linear(1024, 16), nn.SiLU(), nn.Dropout(0.2), nn.Linear(16, 2)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1:] != (2 * self.raw_channels, 1024):
            raise ValueError(f"Expected (batch, {2 * self.raw_channels}, 1024)")
        return self.head(self.encoder(features).flatten(1))


def load_checkpoint(path: str | Path, device: str = "cpu") -> FPPNet:
    """Load a trusted original checkpoint or a bare FPPNet state_dict.

    Original checkpoints also contain mean/std for unused descriptors; the
    final FPP-Net consumes only spectral features and needs neither value.
    Only full, position-head model weights are accepted (strict loading).
    Sampling rate/window length are supplied separately by the caller.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    channels = state["encoder.0.weight"].shape[1]
    if channels not in (2, 6):
        raise ValueError("Expected full KAIST/CEITEC FPP-Net (2/6 spectral channels)")
    model = FPPNet(raw_channels=channels // 2)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def predict_windows(
    models: FPPNet | Sequence[FPPNet], raw: np.ndarray,
    sampling_rate: float, batch_size: int = 256,
) -> np.ndarray:
    """Return (B, 2) probabilities [healthy, fault], averaging model scores.

    Models execute serially on their existing devices. Preprocessing is shared
    across ensemble members. Averages probabilities, never logits. Original
    train/eval flags are restored on exit. Use argmax(1) for predictions; ties
    belong to the healthy class, matching the experimental implementation.
    """
    members = [models] if isinstance(models, FPPNet) else list(models)
    raw = np.asarray(raw)
    if not members or batch_size < 1:
        raise ValueError("Provide at least one model and a positive batch_size")
    if raw.ndim != 3 or len(raw) == 0:
        raise ValueError("raw must be a nonempty (batch, channels, samples) array")
    if any(m.raw_channels != raw.shape[1] for m in members):
        raise ValueError("Sensor channel count differs from the model")
    flags = [m.training for m in members]
    output = []
    try:
        for model in members:
            model.eval()
        with torch.inference_mode():
            for start in range(0, len(raw), batch_size):
                features = torch.from_numpy(spectral_features(
                    raw[start:start + batch_size], sampling_rate
                ))
                probabilities = []
                for model in members:
                    device = next(model.parameters()).device
                    probabilities.append(model(features.to(device)).softmax(1).cpu().numpy())
                output.append(np.mean(probabilities, axis=0))
    finally:
        for model, training in zip(members, flags):
            model.train(training)
    return np.concatenate(output)


def aggregate_records(probabilities: np.ndarray, record_ids: np.ndarray):
    """Average window probabilities per record; return sorted IDs and scores.

    Supply windows from ONE evaluation fold. Do not merge acquisition records
    across domains/folds. This computes full-record, not fixed-duration scores.
    """
    p, records = np.asarray(probabilities), np.asarray(record_ids)
    if p.ndim != 2 or p.shape[1] != 2 or records.shape != (len(p),) or len(p) == 0:
        raise ValueError("Expected probabilities (B,2) and record_ids (B,)")
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Expected normalized finite class probabilities")
    unique = np.unique(records)
    return unique, np.stack([p[records == r].mean(axis=0) for r in unique])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", action="store_true", help="Random-weight execution check, not diagnosis")
    parser.add_argument("--input", type=Path, help="Raw floating-point .npy array (B,C,N)")
    parser.add_argument("--fs", type=float, help="Sampling frequency in Hz")
    parser.add_argument("--checkpoint", type=Path, nargs="+", help="One or more trained .pt files")
    parser.add_argument("--output", type=Path, default=Path("fpp_predictions.npz"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    # CLI uses one CPU thread; importing this module has no global side effects.
    torch.set_num_threads(1)
    if args.demo:
        torch.manual_seed(0)
        for channels, n, fs in [(1, 5120, 25600), (3, 1710, 10000)]:
            raw = np.random.default_rng(0).normal(size=(2, channels, n))
            model = FPPNet(channels).to(args.device)
            p = predict_windows(model, raw, fs)
            print(f"RANDOM-WEIGHT DEMO: C={channels}, parameters={sum(v.numel() for v in model.parameters())}, probabilities={p.shape}")
        return
    if args.input is None or args.fs is None or not args.checkpoint:
        parser.error("Supply --input, --fs and --checkpoint, or use --demo")
    models = [load_checkpoint(p, args.device) for p in args.checkpoint]
    raw = np.load(args.input, mmap_mode="r", allow_pickle=False)
    p = predict_windows(models, raw, args.fs, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, probabilities=p, predictions=p.argmax(1))
    print(f"Saved {len(p)} window predictions to {args.output}")


if __name__ == "__main__":
    main()
