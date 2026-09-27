from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

import tensorly as tl
from tensorly.decomposition import tensor_ring

from .resizer import Resizer


tl.set_backend("pytorch")

FactorState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
FACTOR_NAMES = ("G1", "G2", "G3")
_RESIZER_CACHE = {}


def canonicalize_factor_state(
    g1: torch.Tensor,
    g2: torch.Tensor,
    g3: torch.Tensor,
    core: torch.Tensor,
) -> FactorState:
    g1 = g1.clone()
    g2 = g2.clone()
    g3 = g3.clone()
    core = core.clone()

    for index in range(core.shape[0]):
        component = g1[..., index]
        scale = component.norm().clamp_min(1e-8)
        pivot = component.flatten()[component.flatten().abs().argmax()]
        sign = -1.0 if pivot < 0 else 1.0
        g1[..., index] = component / scale * sign
        core[index, :, :] = core[index, :, :] * (scale * sign)

    for index in range(core.shape[1]):
        component = g2[..., index]
        scale = component.norm().clamp_min(1e-8)
        pivot = component.flatten()[component.flatten().abs().argmax()]
        sign = -1.0 if pivot < 0 else 1.0
        g2[..., index] = component / scale * sign
        core[:, index, :] = core[:, index, :] * (scale * sign)

    for index in range(core.shape[2]):
        component = g3[..., index]
        scale = component.norm().clamp_min(1e-8)
        pivot = component.flatten()[component.flatten().abs().argmax()]
        sign = -1.0 if pivot < 0 else 1.0
        g3[..., index] = component / scale * sign
        core[:, :, index] = core[:, :, index] * (scale * sign)

    return g1, g2, g3, core


def tw_block(
    g1_block: torch.Tensor,
    g2: torch.Tensor,
    g3: torch.Tensor,
    core: torch.Tensor,
) -> torch.Tensor:
    """Storage convention:

    G1: (R1, H, R2, L1)
    G2: (R2, W, R3, L2)
    G3: (R3, C, R1, L3)
    C : (L1, L2, L3)
    """
    p = torch.einsum("ahbu,bwcv->ahwcuv", g1_block, g2)
    q = torch.einsum("csat,uvt->acuvs", g3, core)
    return torch.einsum("ahwcuv,acuvs->hws", p, q)


def reconstruct_tw(
    g1: torch.Tensor,
    g2: torch.Tensor,
    g3: torch.Tensor,
    core: torch.Tensor,
    block_size: int = 16,
) -> torch.Tensor:
    blocks = []
    for start in range(0, g1.shape[1], block_size):
        stop = min(start + block_size, g1.shape[1])
        blocks.append(tw_block(g1[:, start:stop], g2, g3, core))
    return torch.cat(blocks, dim=0)


def _batched_probe_reconstruction(
    factors: FactorState,
    core: torch.Tensor,
    *,
    scale: int,
    grid_size: int,
):
    """Reconstruct a small, regularly sampled set of scale x scale blocks."""
    g1, g2, g3 = factors[:3]
    if g1.dim() == 4:
        g1, g2, g3 = (value.unsqueeze(0) for value in (g1, g2, g3))
    if core.dim() == 3:
        core = core.unsqueeze(0)
    batch = core.shape[0]
    if g1.shape[0] == 1 and batch > 1:
        g1, g2, g3 = (value.expand(batch, *value.shape[1:]) for value in (g1, g2, g3))
    low_h = g1.shape[2] // int(scale)
    low_w = g2.shape[2] // int(scale)
    count = max(1, min(int(grid_size), low_h, low_w))
    low_rows = torch.linspace(0, low_h - 1, count, device=core.device).round().long()
    low_cols = torch.linspace(0, low_w - 1, count, device=core.device).round().long()
    offsets = torch.arange(int(scale), device=core.device)
    rows = (low_rows[:, None] * int(scale) + offsets[None]).reshape(-1)
    cols = (low_cols[:, None] * int(scale) + offsets[None]).reshape(-1)
    g1_probe = g1.index_select(2, rows)
    g2_probe = g2.index_select(2, cols)
    spatial = torch.einsum("bahdu,bdwcv->bahwcuv", g1_probe, g2_probe)
    spectral = torch.einsum("bcsat,buvt->bacuvs", g3, core)
    recon = torch.einsum("bahwcuv,bacuvs->bhws", spatial, spectral)
    return recon, low_rows, low_cols


def _probe_observations(
    factors: FactorState,
    core: torch.Tensor,
    srf: torch.Tensor,
    *,
    scale: int,
    grid_size: int,
):
    recon, low_rows, low_cols = _batched_probe_reconstruction(
        factors,
        core,
        scale=scale,
        grid_size=grid_size,
    )
    count = low_rows.numel()
    batch, _, _, channels = recon.shape
    lr_probe = recon.reshape(
        batch, count, scale, count, scale, channels
    ).mean(dim=(2, 4))
    msi_probe = torch.einsum("bhwc,cs->bhws", recon, srf)
    return recon, lr_probe, msi_probe, low_rows, low_cols


def _normalized_residual_statistics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    spatial_dims,
):
    error = prediction - target
    target_scale = target.square().mean(dim=spatial_dims).sqrt().clamp_min(1e-2)
    signed = error.mean(dim=spatial_dims) / target_scale
    rms = error.square().mean(dim=spatial_dims).sqrt() / target_scale
    return torch.cat([signed, rms], dim=1)


def factor_observation_residual_features(
    factors: FactorState,
    current_core: torch.Tensor,
    target_core: torch.Tensor,
    srf: torch.Tensor,
    *,
    scale: int,
    grid_size: int = 4,
):
    """Residual summary used while training from clean factor pairs."""
    _, lr_current, msi_current, _, _ = _probe_observations(
        factors, current_core, srf, scale=scale, grid_size=grid_size
    )
    _, lr_target, msi_target, _, _ = _probe_observations(
        factors, target_core, srf, scale=scale, grid_size=grid_size
    )
    lr_features = _normalized_residual_statistics(
        lr_current, lr_target, spatial_dims=(1, 2)
    )
    msi_features = _normalized_residual_statistics(
        msi_current, msi_target, spatial_dims=(1, 2)
    )
    return torch.tanh(torch.nan_to_num(torch.cat([lr_features, msi_features], dim=1)))


def observation_residual_features(
    factors: FactorState,
    current_core: torch.Tensor,
    lr_hsi: torch.Tensor,
    hr_msi: torch.Tensor,
    srf: torch.Tensor,
    *,
    scale: int,
    grid_size: int = 4,
):
    """Residual summary against the actual LRHSI and HRMSI observations."""
    _, lr_current, msi_current, low_rows, low_cols = _probe_observations(
        factors, current_core, srf, scale=scale, grid_size=grid_size
    )
    lr_target = lr_hsi.index_select(2, low_rows).index_select(3, low_cols)
    lr_target = lr_target.permute(0, 2, 3, 1)
    offsets = torch.arange(int(scale), device=current_core.device)
    rows = (low_rows[:, None] * int(scale) + offsets[None]).reshape(-1)
    cols = (low_cols[:, None] * int(scale) + offsets[None]).reshape(-1)
    msi_target = hr_msi.index_select(2, rows).index_select(3, cols)
    msi_target = msi_target.permute(0, 2, 3, 1)
    lr_features = _normalized_residual_statistics(
        lr_current, lr_target, spatial_dims=(1, 2)
    )
    msi_features = _normalized_residual_statistics(
        msi_current, msi_target, spatial_dims=(1, 2)
    )
    return torch.tanh(torch.nan_to_num(torch.cat([lr_features, msi_features], dim=1)))


def probe_reconstruction_losses(
    factors: FactorState,
    predicted_core: torch.Tensor,
    target_core: torch.Tensor,
    srf: torch.Tensor,
    *,
    scale: int,
    grid_size: int = 4,
):
    predicted, lr_pred, msi_pred, _, _ = _probe_observations(
        factors, predicted_core, srf, scale=scale, grid_size=grid_size
    )
    target, lr_target, msi_target, _, _ = _probe_observations(
        factors, target_core, srf, scale=scale, grid_size=grid_size
    )

    def relative_mse(left, right):
        return F.mse_loss(left, right) / right.square().mean().clamp_min(1e-4)

    reconstruction = relative_mse(predicted, target)
    observation = relative_mse(lr_pred, lr_target) + relative_mse(
        msi_pred, msi_target
    )
    return reconstruction, observation


def factor_to_signal(raw: torch.Tensor, name: str) -> torch.Tensor:
    if name not in FACTOR_NAMES:
        raise ValueError(f"unknown factor {name}")
    if raw.dim() == 4:
        raw = raw.unsqueeze(0)
    if raw.dim() != 5:
        raise ValueError(f"{name} must be a 4-D factor or a batched 5-D factor")
    return raw.permute(0, 1, 3, 4, 2).reshape(raw.shape[0], -1, raw.shape[2])


def signal_to_factor(
    signal: torch.Tensor,
    shape: Sequence[int],
    name: str,
) -> torch.Tensor:
    if name == "G1":
        r1, h_dim, r2, l1 = shape
        return signal.view(r1, r2, l1, h_dim).permute(0, 3, 1, 2)
    if name == "G2":
        r2, w_dim, r3, l2 = shape
        return signal.view(r2, r3, l2, w_dim).permute(0, 3, 1, 2)
    if name == "G3":
        r3, c_dim, r1, l3 = shape
        return signal.view(r3, r1, l3, c_dim).permute(0, 3, 1, 2)
    raise ValueError(f"unknown factor {name}")


def factor_shapes(
    height: int,
    width: int,
    channels: int,
    ring_ranks: Sequence[int],
    core_ranks: Sequence[int],
) -> Dict[str, Tuple[int, ...]]:
    r1, r2, r3 = (int(value) for value in ring_ranks)
    l1, l2, l3 = (int(value) for value in core_ranks)
    return {
        "G1": (r1, height, r2, l1),
        "G2": (r2, width, r3, l2),
        "G3": (r3, channels, r1, l3),
        "core": (l1, l2, l3),
    }


def init_tw_from_tensor(
    target: torch.Tensor,
    ring_ranks: Sequence[int],
    core_ranks: Sequence[int],
    *,
    seed: int,
) -> FactorState:
    if target.dim() == 4 and target.shape[0] == 1:
        target = target[0]
    if target.dim() != 3:
        raise ValueError("target must have shape (H, W, C)")

    device = target.device
    dtype = target.dtype
    r1, r2, r3 = (int(value) for value in ring_ranks)
    l1, l2, l3 = (int(value) for value in core_ranks)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    g1 = torch.empty((r1, target.shape[0], r2, l1), dtype=dtype)
    g2 = torch.empty((r2, target.shape[1], r3, l2), dtype=dtype)
    g3 = torch.empty((r3, target.shape[2], r1, l3), dtype=dtype)
    try:
        factors = tensor_ring(
            target.detach().float().cpu().contiguous(),
            rank=[r1, r2, r3, r1],
            svd="randomized_svd",
        )
        g1[:, :, :, 0] = factors[0].to(dtype=dtype)
        g2[:, :, :, 0] = factors[1].to(dtype=dtype)
        g3[:, :, :, 0] = factors[2].to(dtype=dtype)
        for index in range(1, l1):
            g1[..., index] = 0
        for index in range(1, l2):
            g2[..., index] = 0
        for index in range(1, l3):
            g3[..., index] = 0
    except Exception:
        g1 = torch.randn(g1.shape, generator=generator, dtype=dtype) * 0.01
        g2 = torch.randn(g2.shape, generator=generator, dtype=dtype) * 0.01
        g3 = torch.randn(g3.shape, generator=generator, dtype=dtype) * 0.01

    core = torch.zeros((l1, l2, l3), dtype=dtype)
    for index in range(min(l1, l2, l3)):
        core[index, index, index] = 1.0
    core = core + torch.randn(core.shape, generator=generator, dtype=dtype) * 1e-3
    return (
        g1.to(device=device),
        g2.to(device=device),
        g3.to(device=device),
        core.to(device=device),
    )


def downsample_spatial(image: torch.Tensor, scale: int) -> torch.Tensor:
    if scale < 1:
        raise ValueError("scale must be positive")
    cache_key = (
        tuple(image.shape),
        int(scale),
        image.device.type,
        image.device.index,
        image.dtype,
    )
    resizer = _RESIZER_CACHE.get(cache_key)
    if resizer is None:
        scale_factor = [1.0] * image.dim()
        scale_factor[-2:] = [1.0 / float(scale), 1.0 / float(scale)]
        resizer = Resizer(image.shape, scale_factor).to(
            device=image.device, dtype=image.dtype
        )
        _RESIZER_CACHE[cache_key] = resizer
    return resizer(image)


def apply_srf(image: torch.Tensor, srf: torch.Tensor) -> torch.Tensor:
    if srf.dim() != 2:
        raise ValueError("srf must have shape (num_hsi_bands, num_msi_bands)")
    if image.shape[1] != srf.shape[0]:
        raise ValueError("srf input dimension must match image channels")
    return torch.einsum("nchw,cs->nshw", image, srf)


def coupled_loss_from_factors(
    factors: FactorState,
    lr_hsi: torch.Tensor,
    hr_msi: torch.Tensor,
    srf: torch.Tensor,
    *,
    scale: int,
    block_size: int,
    lambda_msi: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    full = reconstruct_tw(*factors, block_size=block_size)
    full_chw = full.permute(2, 0, 1).unsqueeze(0)
    lr_hat = downsample_spatial(full_chw, scale)
    msi_hat = apply_srf(full_chw, srf)
    lr_loss = F.mse_loss(lr_hat, lr_hsi)
    msi_loss = F.mse_loss(msi_hat, hr_msi)
    total = lr_loss + float(lambda_msi) * msi_loss
    return total, {
        "lr": float(lr_loss.detach()),
        "msi": float(msi_loss.detach()),
    }


def init_from_observations(
    lr_hsi: torch.Tensor,
    hr_msi: torch.Tensor,
    srf: torch.Tensor,
    *,
    ring_ranks: Sequence[int],
    core_ranks: Sequence[int],
    scale: int,
    projection_steps: int,
    projection_lr: float,
    block_size: int,
    lambda_msi: float,
    seed: int,
) -> FactorState:
    hr_hwc = hr_msi[0].permute(1, 2, 0).contiguous()
    lr_hwc = lr_hsi[0].permute(1, 2, 0).contiguous()
    state_z = init_tw_from_tensor(
        hr_hwc,
        ring_ranks,
        core_ranks,
        seed=seed,
    )
    state_y = init_tw_from_tensor(
        lr_hwc,
        ring_ranks,
        core_ranks,
        seed=seed + 1000,
    )
    initial = canonicalize_factor_state(
        state_z[0],
        state_z[1],
        state_y[2],
        (state_z[3] + state_y[3]) * 0.5,
    )
    return joint_projection(
        initial,
        lr_hsi,
        hr_msi,
        srf,
        scale=scale,
        steps=projection_steps,
        lr=projection_lr,
        block_size=block_size,
        lambda_msi=lambda_msi,
    )


def joint_projection(
    factors: FactorState,
    lr_hsi: torch.Tensor,
    hr_msi: torch.Tensor,
    srf: torch.Tensor,
    *,
    scale: int,
    steps: int,
    lr: float,
    block_size: int,
    lambda_msi: float,
) -> FactorState:
    state = [tensor.detach().clone().requires_grad_(True) for tensor in factors]
    optimizer = torch.optim.Adam(state, lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = coupled_loss_from_factors(
            (state[0], state[1], state[2], state[3]),
            lr_hsi,
            hr_msi,
            srf,
            scale=scale,
            block_size=block_size,
            lambda_msi=lambda_msi,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(state, max_norm=1.0)
        optimizer.step()
    return canonicalize_factor_state(*[tensor.detach() for tensor in state])


def condition_factors_on_core(
    factors: FactorState,
    core: torch.Tensor,
    lr_hsi: torch.Tensor,
    hr_msi: torch.Tensor,
    srf: torch.Tensor,
    *,
    scale: int,
    steps: int,
    lr: float,
    block_size: int,
    lambda_msi: float,
    anchor_weight: float,
) -> FactorState:
    g1 = factors[0].detach().clone().requires_grad_(True)
    g2 = factors[1].detach().clone().requires_grad_(True)
    g3 = factors[2].detach().clone().requires_grad_(True)
    anchors = [factors[index].detach() for index in range(3)]
    core = core.detach()
    optimizer = torch.optim.Adam([g1, g2, g3], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        data_loss, _ = coupled_loss_from_factors(
            (g1, g2, g3, core),
            lr_hsi,
            hr_msi,
            srf,
            scale=scale,
            block_size=block_size,
            lambda_msi=lambda_msi,
        )
        anchor_loss = (
            F.mse_loss(g1, anchors[0])
            + F.mse_loss(g2, anchors[1])
            + F.mse_loss(g3, anchors[2])
        )
        loss = data_loss + float(anchor_weight) * anchor_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([g1, g2, g3], max_norm=1.0)
        optimizer.step()

    return canonicalize_factor_state(g1.detach(), g2.detach(), g3.detach(), core)


def condition_spatial_on_core_g3(
    factors: FactorState,
    core: torch.Tensor,
    g3: torch.Tensor,
    lr_hsi: torch.Tensor,
    hr_msi: torch.Tensor,
    srf: torch.Tensor,
    *,
    scale: int,
    steps: int,
    lr: float,
    block_size: int,
    lambda_msi: float,
    anchor_weight: float,
) -> FactorState:
    """Fit only G1/G2 while the jointly controlled C/G3 pair stays fixed."""
    g1 = factors[0].detach().clone().requires_grad_(True)
    g2 = factors[1].detach().clone().requires_grad_(True)
    anchors = (g1.detach().clone(), g2.detach().clone())
    core = core.detach()
    g3 = g3.detach()
    optimizer = torch.optim.Adam([g1, g2], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        data_loss, _ = coupled_loss_from_factors(
            (g1, g2, g3, core),
            lr_hsi,
            hr_msi,
            srf,
            scale=scale,
            block_size=block_size,
            lambda_msi=lambda_msi,
        )
        anchor_loss = F.mse_loss(g1, anchors[0]) + F.mse_loss(g2, anchors[1])
        loss = data_loss + float(anchor_weight) * anchor_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([g1, g2], max_norm=1.0)
        optimizer.step()

    return canonicalize_factor_state(g1.detach(), g2.detach(), g3, core)


def project_core_g3(
    factors: FactorState,
    lr_hsi: torch.Tensor,
    hr_msi: torch.Tensor,
    srf: torch.Tensor,
    *,
    scale: int,
    steps: int,
    lr: float,
    block_size: int,
    lambda_msi: float,
    core_prior_weight: float,
    g3_prior_weight: float,
    g3_anchor: torch.Tensor,
) -> FactorState:
    """Final self-supervised C/G3 projection with G1/G2 held fixed."""
    g1, g2 = factors[0].detach(), factors[1].detach()
    g3 = factors[2].detach().clone().requires_grad_(True)
    core = factors[3].detach().clone().requires_grad_(True)
    core_prior = core.detach().clone()
    g3_prior = g3_anchor.detach()
    optimizer = torch.optim.Adam([g3, core], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        data_loss, _ = coupled_loss_from_factors(
            (g1, g2, g3, core),
            lr_hsi,
            hr_msi,
            srf,
            scale=scale,
            block_size=block_size,
            lambda_msi=lambda_msi,
        )
        prior_loss = (
            float(core_prior_weight) * F.mse_loss(core, core_prior)
            + float(g3_prior_weight) * F.mse_loss(g3, g3_prior)
        )
        (data_loss + prior_loss).backward()
        torch.nn.utils.clip_grad_norm_([g3, core], max_norm=1.0)
        optimizer.step()

    return canonicalize_factor_state(g1, g2, g3.detach(), core.detach())


def refine_g2_with_holdout_gate(
    factors: FactorState,
    lr_hsi: torch.Tensor,
    hr_msi: torch.Tensor,
    srf: torch.Tensor,
    *,
    scale: int,
    steps: int,
    lr: float,
    block_size: int,
    lambda_msi: float,
    latent_dim: int,
    anchor_weight: float,
    gate_scales: Sequence[float],
    holdout_period: int,
    seed: int,
):
    """Fit a low-frequency G2 residual and accept it on held-out observations."""
    anchor = factors[1].detach()
    retained = max(1, min(int(latent_dim), anchor.shape[1]))
    positions = torch.arange(
        anchor.shape[1], device=anchor.device, dtype=anchor.dtype
    ).add_(0.5)
    frequencies = torch.arange(retained, device=anchor.device, dtype=anchor.dtype)
    basis = torch.cos(
        torch.pi * positions[:, None] * frequencies[None, :] / anchor.shape[1]
    )
    basis = torch.linalg.qr(basis, mode="reduced").Q
    coefficients = torch.zeros(
        anchor.shape[0], retained, anchor.shape[2], anchor.shape[3],
        device=anchor.device, dtype=anchor.dtype, requires_grad=True,
    )
    optimizer = torch.optim.Adam([coefficients], lr=lr)
    fixed = tuple(value.detach() for value in factors)

    def mask_like(value, offset):
        height, width = value.shape[-2:]
        mask = (
            torch.arange(height * width, device=value.device) % int(holdout_period)
            == int(offset)
        ).reshape(1, 1, height, width)
        return mask.expand_as(value)

    lr_validation = mask_like(lr_hsi, seed % holdout_period)
    msi_validation = mask_like(hr_msi, (seed + 7) % holdout_period)

    def losses(g2):
        full = reconstruct_tw(fixed[0], g2, fixed[2], fixed[3], block_size)
        full = full.permute(2, 0, 1).unsqueeze(0)
        lr_error = (downsample_spatial(full, scale) - lr_hsi).square()
        msi_error = (apply_srf(full, srf) - hr_msi).square()
        train = lr_error.masked_select(~lr_validation).mean() + float(
            lambda_msi
        ) * msi_error.masked_select(~msi_validation).mean()
        validation = lr_error.masked_select(lr_validation).mean() + float(
            lambda_msi
        ) * msi_error.masked_select(msi_validation).mean()
        return train, validation

    for _ in range(max(0, int(steps))):
        optimizer.zero_grad(set_to_none=True)
        residual = torch.einsum("pk,akbl->apbl", basis, coefficients)
        train, _ = losses(anchor + residual)
        (train + float(anchor_weight) * coefficients.square().mean()).backward()
        torch.nn.utils.clip_grad_norm_([coefficients], max_norm=1.0)
        optimizer.step()

    residual = torch.einsum("pk,akbl->apbl", basis, coefficients.detach())
    candidates = []
    with torch.no_grad():
        for candidate_scale in gate_scales:
            _, validation = losses(anchor + float(candidate_scale) * residual)
            candidates.append((float(validation), float(candidate_scale)))
    best_loss, best_scale = min(candidates, key=lambda item: item[0])
    result = list(fixed)
    result[1] = anchor + best_scale * residual
    report = {
        "selected_scale": best_scale,
        "validation_loss": best_loss,
        "candidates": [
            {"scale": candidate_scale, "validation_loss": validation}
            for validation, candidate_scale in candidates
        ],
    }
    return tuple(result), report


def relative_factor_change(
    previous: FactorState,
    current: FactorState,
) -> float:
    numerator = sum(
        float((current[index] - previous[index]).norm())
        for index in range(4)
    )
    denominator = sum(
        float(previous[index].norm().clamp_min(1e-8))
        for index in range(4)
    )
    return numerator / max(denominator, 1e-8)
