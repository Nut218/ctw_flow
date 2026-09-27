from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F

from .tw import canonicalize_factor_state


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def load_observations(
    data_path: str,
    srf_path: str,
    device: torch.device,
    *,
    crop_size: int = 0,
    scale: int = 4,
):
    data = sio.loadmat(data_path)
    gt = (
        np.asarray(data["HRHSI"], dtype=np.float32)
        if "HRHSI" in data
        else None
    )
    lr_hsi = np.asarray(data["LRHSI"], dtype=np.float32)
    hr_msi = np.asarray(data["HRMSI"], dtype=np.float32)

    if crop_size > 0:
        if crop_size % scale != 0:
            raise ValueError("crop_size must be divisible by scale")
        low = crop_size // scale
        if gt is not None:
            gt = gt[:crop_size, :crop_size]
        lr_hsi = lr_hsi[:low, :low]
        hr_msi = hr_msi[:crop_size, :crop_size]

    lr_tensor = (
        torch.from_numpy(lr_hsi)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    msi_tensor = (
        torch.from_numpy(hr_msi)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    srf = torch.from_numpy(
        np.transpose(sio.loadmat(srf_path)["srf"])
    ).float().to(device)
    return gt, lr_tensor, msi_tensor, srf


def make_synthetic_observations(
    device: torch.device,
    *,
    height: int = 64,
    width: int = 64,
    channels: int = 8,
    scale: int = 4,
):
    torch.manual_seed(7)
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height),
        torch.linspace(-1.0, 1.0, width),
        indexing="ij",
    )
    base = torch.exp(-((ys - 0.2).square() + (xs + 0.1).square()) * 3.0)
    spectral_curve = 0.5 + 0.5 * torch.sin(torch.arange(1, channels + 1) * 1.7)
    gt = torch.stack(
        [base * spectral_curve[band] for band in range(channels)],
        dim=-1,
    )
    gt = (gt - gt.min()) / (gt.max() - gt.min())

    lr_hsi = F.avg_pool2d(
        gt.permute(2, 0, 1).unsqueeze(0),
        kernel_size=scale,
        stride=scale,
    )
    srf = torch.zeros(channels, 3)
    srf[:3, :3] = torch.eye(3)
    if channels > 3:
        srf[3:, 0] = 0.2
        srf[3:, 1] = 0.1
        srf[3:, 2] = 0.1
    hr_msi = torch.einsum(
        "nchw,cs->nshw",
        gt.permute(2, 0, 1).unsqueeze(0),
        srf,
    )
    return (
        gt.numpy().astype(np.float32),
        lr_hsi.to(device),
        hr_msi.to(device),
        srf.to(device),
    )


def load_factor_corpus(
    code_path: str,
    *,
    ring_ranks: Sequence[int],
    core_ranks: Sequence[int],
    dataset_prefix: str = "CAVE/",
    max_items: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    store = torch.load(code_path, map_location="cpu")
    ring_tag = "r" + "-".join(str(int(value)) for value in ring_ranks)
    core_tag = "l" + "-".join(str(int(value)) for value in core_ranks)
    suffix = f"{ring_tag}_{core_tag}"
    keys = sorted(
        key
        for key in store
        if suffix in key and key.startswith(dataset_prefix)
    )
    if max_items > 0:
        keys = keys[:max_items]
    if not keys:
        raise ValueError(f"no factor entries with suffix {suffix} in {code_path}")

    g1_values = []
    g2_values = []
    g3_values = []
    core_values = []
    for key in keys:
        item = store[key]
        factors = item["factors"]
        state = canonicalize_factor_state(
            factors[0].float(),
            factors[1].float(),
            factors[2].float(),
            item["core"].float(),
        )
        g1_values.append(state[0])
        g2_values.append(state[1])
        g3_values.append(state[2])
        core_values.append(state[3])

    return (
        torch.stack(g1_values),
        torch.stack(g2_values),
        torch.stack(g3_values),
        torch.stack(core_values),
    )
