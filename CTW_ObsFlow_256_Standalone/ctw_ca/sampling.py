from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .tw import (
    FactorState,
    canonicalize_factor_state,
    condition_factors_on_core,
    condition_spatial_on_core_g3,
    coupled_loss_from_factors,
    observation_residual_features,
    joint_projection,
    project_core_g3,
    refine_g2_with_holdout_gate,
    reconstruct_tw,
    relative_factor_change,
)


def linear_schedule(steps: int, device: torch.device):
    betas = np.linspace(1e-4, 0.02, steps, dtype=np.float64)
    alphas_cumprod = np.cumprod(1.0 - betas)
    sqrt_ac = torch.from_numpy(np.sqrt(alphas_cumprod)).float().to(device)
    sqrt_1_ac = torch.from_numpy(np.sqrt(1.0 - alphas_cumprod)).float().to(device)
    return sqrt_ac, sqrt_1_ac


def core_to_signal(
    core: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return (core.reshape(1, -1) - mean) / std


def signal_to_core(
    signal: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    shape: Sequence[int],
) -> torch.Tensor:
    return (signal * std + mean).reshape(shape)


class ConditionalFactorOptimizer:
    def __init__(
        self,
        factors: FactorState,
        anchors: FactorState,
        *,
        lr: float,
    ):
        self.g1 = factors[0].detach().clone().requires_grad_(True)
        self.g2 = factors[1].detach().clone().requires_grad_(True)
        self.g3 = factors[2].detach().clone().requires_grad_(True)
        self.anchor_g1 = anchors[0].detach().clone()
        self.anchor_g2 = anchors[1].detach().clone()
        self.anchor_g3 = anchors[2].detach().clone()
        self.optimizer = torch.optim.Adam([self.g1, self.g2, self.g3], lr=lr)

    def state(
        self,
        core: torch.Tensor,
        *,
        detach_core: bool = True,
    ) -> FactorState:
        return (
            self.g1.detach(),
            self.g2.detach(),
            self.g3.detach(),
            core.detach() if detach_core else core,
        )

    def step(
        self,
        core: torch.Tensor,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        srf: torch.Tensor,
        *,
        scale: int,
        block_size: int,
        lambda_msi: float,
        anchor_weight: float,
    ) -> Dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        data_loss, errors = coupled_loss_from_factors(
            (self.g1, self.g2, self.g3, core.detach()),
            lr_hsi,
            hr_msi,
            srf,
            scale=scale,
            block_size=block_size,
            lambda_msi=lambda_msi,
        )
        anchor_loss = (
            F.mse_loss(self.g1, self.anchor_g1)
            + F.mse_loss(self.g2, self.anchor_g2)
            + F.mse_loss(self.g3, self.anchor_g3)
        )
        loss = data_loss + float(anchor_weight) * anchor_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [self.g1, self.g2, self.g3],
            max_norm=1.0,
        )
        self.optimizer.step()
        return errors


class ConditionalSpatialOptimizer:
    """Observation-conditioned G1/G2 optimizer for joint C/G3 sampling."""

    def __init__(self, factors: FactorState, anchors: FactorState, *, lr: float):
        self.g1 = factors[0].detach().clone().requires_grad_(True)
        self.g2 = factors[1].detach().clone().requires_grad_(True)
        self.anchor_g1 = anchors[0].detach().clone()
        self.anchor_g2 = anchors[1].detach().clone()
        self.optimizer = torch.optim.Adam([self.g1, self.g2], lr=lr)

    def state(
        self,
        g3: torch.Tensor,
        core: torch.Tensor,
        *,
        detach_g3: bool = True,
        detach_core: bool = True,
    ) -> FactorState:
        return (
            self.g1.detach(),
            self.g2.detach(),
            g3.detach() if detach_g3 else g3,
            core.detach() if detach_core else core,
        )

    def step(
        self,
        g3: torch.Tensor,
        core: torch.Tensor,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        srf: torch.Tensor,
        *,
        scale: int,
        block_size: int,
        lambda_msi: float,
        anchor_weight: float,
    ) -> Dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        data_loss, errors = coupled_loss_from_factors(
            (self.g1, self.g2, g3.detach(), core.detach()),
            lr_hsi,
            hr_msi,
            srf,
            scale=scale,
            block_size=block_size,
            lambda_msi=lambda_msi,
        )
        anchor_loss = F.mse_loss(self.g1, self.anchor_g1) + F.mse_loss(
            self.g2, self.anchor_g2
        )
        (data_loss + float(anchor_weight) * anchor_loss).backward()
        torch.nn.utils.clip_grad_norm_([self.g1, self.g2], max_norm=1.0)
        self.optimizer.step()
        return errors


def make_schedule(t_start: int, steps: int):
    schedule = []
    for value in torch.linspace(t_start, 0, steps).round().long().tolist():
        if not schedule or schedule[-1] != value:
            schedule.append(value)
    if schedule[-1] != 0:
        schedule.append(0)
    return schedule


def coarse_to_fine_weight(step_index: int, num_steps: int, args):
    """Return a smooth coarse-to-detail blend weight for observation guidance."""
    if not bool(getattr(args, "coarse_to_fine_guidance", False)):
        return None
    progress = float(step_index) / float(max(1, num_steps - 1))
    start = float(args.coarse_transition_start)
    end = float(args.coarse_transition_end)
    if not 0.0 <= start < end <= 1.0:
        raise ValueError(
            "coarse-to-fine transition must satisfy 0 <= start < end <= 1"
        )
    blend = min(1.0, max(0.0, (progress - start) / (end - start)))
    return blend * blend * (3.0 - 2.0 * blend)


def guided_coupled_loss(
    factors,
    lr_hsi,
    hr_msi,
    srf,
    *,
    step_index: int,
    num_steps: int,
    args,
):
    return coupled_loss_from_factors(
        factors,
        lr_hsi,
        hr_msi,
        srf,
        scale=args.scale,
        block_size=args.block_size,
        lambda_msi=args.lambda_msi,
        coarse_to_fine_weight=coarse_to_fine_weight(
            step_index, num_steps, args
        ),
        lowpass_kernel=getattr(args, "coarse_lowpass_kernel", 5),
        full_loss_weight=getattr(args, "coarse_full_loss_weight", 0.2),
        detail_gradient_weight=getattr(
            args, "coarse_detail_gradient_weight", 0.25
        ),
    )


def calibrate_factor_adapter(
    *,
    model: torch.nn.Module,
    objective: str,
    initial_factors: FactorState,
    stats: Dict[str, object],
    lr_hsi: torch.Tensor,
    hr_msi: torch.Tensor,
    srf: torch.Tensor,
    args,
) -> Dict[str, object]:
    """Self-supervise the small factor adapter while freezing the core prior."""
    parameters = list(model.trainable_parameters())
    if not parameters or args.adapter_tta_steps <= 0:
        return {"accepted": False, "reason": "disabled"}
    snapshot = [parameter.detach().clone() for parameter in parameters]
    mean = stats["mean"].to(lr_hsi.device)
    std = stats["std"].to(lr_hsi.device)
    core_shape = tuple(stats["shape"])
    clean_signal = core_to_signal(initial_factors[3], mean, std)
    sqrt_ac, sqrt_1_ac = linear_schedule(1000, lr_hsi.device)
    generator = torch.Generator(device=lr_hsi.device).manual_seed(args.seed + 2718)
    fixed_noise = torch.randn(
        clean_signal.shape,
        generator=generator,
        device=clean_signal.device,
        dtype=clean_signal.dtype,
    )

    def loss_at(timestep: int, *, with_grad: bool):
        time = torch.full(
            (1,), timestep, device=lr_hsi.device, dtype=torch.long
        )
        noisy = sqrt_ac[timestep] * clean_signal + sqrt_1_ac[timestep] * fixed_noise
        prediction = model(noisy, time, initial_factors)
        if objective == "pred_x0":
            clean_prediction = prediction
        else:
            clean_prediction = (
                noisy - sqrt_1_ac[timestep] * prediction
            ) / sqrt_ac[timestep]
        core = signal_to_core(clean_prediction, mean, std, core_shape)
        data_loss, _ = coupled_loss_from_factors(
            (initial_factors[0], initial_factors[1], initial_factors[2], core),
            lr_hsi,
            hr_msi,
            srf,
            scale=args.scale,
            block_size=args.block_size,
            lambda_msi=args.lambda_msi,
        )
        with torch.no_grad():
            base_prediction = model.base(noisy, time)
        prior = F.mse_loss(prediction, base_prediction)
        return data_loss + float(args.adapter_tta_prior) * prior

    validation_times = (
        min(args.t_start, 400),
        min(args.t_start, 200),
        min(args.t_start, 50),
    )
    with torch.no_grad():
        before = float(torch.stack([loss_at(t, with_grad=False) for t in validation_times]).mean())
    optimizer = torch.optim.Adam(parameters, lr=args.adapter_tta_lr)
    model.train()
    for step in range(args.adapter_tta_steps):
        optimizer.zero_grad(set_to_none=True)
        fraction = step / max(1, args.adapter_tta_steps - 1)
        timestep = int(round(args.t_start * (1.0 - fraction)))
        loss = loss_at(timestep, with_grad=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        after = float(torch.stack([loss_at(t, with_grad=False) for t in validation_times]).mean())
    accepted = after < before
    if not accepted:
        with torch.no_grad():
            for parameter, original in zip(parameters, snapshot):
                parameter.copy_(original)
    return {"accepted": accepted, "before": before, "after": after}


def sample_core_with_cross_attention(
    *,
    model: torch.nn.Module,
    objective: str,
    initial_factors: FactorState,
    stats: Dict[str, object],
    lr_hsi: torch.Tensor,
    hr_msi: torch.Tensor,
    srf: torch.Tensor,
    args,
) -> Dict[str, object]:
    device = lr_hsi.device
    mean = stats["mean"].to(device)
    std = stats["std"].to(device)
    core_shape = tuple(stats["shape"])
    anchors = tuple(tensor.detach().clone() for tensor in initial_factors)
    joint_core_g3 = bool(getattr(args, "joint_core_g3", False))
    conditional = (
        ConditionalSpatialOptimizer(initial_factors, anchors, lr=args.factor_lr)
        if joint_core_g3
        else ConditionalFactorOptimizer(initial_factors, anchors, lr=args.factor_lr)
    )
    g3 = initial_factors[2].detach().clone()
    g3_anchor = g3.detach().clone()

    core_signal = core_to_signal(initial_factors[3], mean, std)
    sqrt_ac, sqrt_1_ac = linear_schedule(1000, device)
    c_t = (
        sqrt_ac[args.t_start] * core_signal
        + sqrt_1_ac[args.t_start] * torch.randn_like(core_signal)
    )
    schedule = make_schedule(args.t_start, args.ddim_steps)

    history = []
    previous_factors = anchors
    last_core = initial_factors[3]
    model.eval()
    for step_index, t in enumerate(schedule[:-1]):
        s = schedule[step_index + 1]
        batched_t = torch.full((1,), t, device=device, dtype=torch.long)

        g3_t = g3.detach().requires_grad_(joint_core_g3)
        conditioning_factors = (
            conditional.state(
                g3_t,
                previous_factors[3],
                detach_g3=not joint_core_g3,
            )
            if joint_core_g3
            else conditional.state(previous_factors[3])
        )
        c_t = c_t.detach().requires_grad_(True)
        prediction = model(c_t, batched_t, conditioning_factors)
        if objective == "pred_x0":
            c0_signal = prediction
        else:
            c0_signal = (c_t - sqrt_1_ac[t] * prediction) / sqrt_ac[t]
        core = signal_to_core(c0_signal, mean, std, core_shape)

        update_factors = (
            args.factor_interval > 0
            and step_index % args.factor_interval == 0
            and t <= args.factor_start_t
        )
        if update_factors:
            if joint_core_g3:
                conditional.step(
                    g3_t.detach(),
                    core.detach(),
                    lr_hsi,
                    hr_msi,
                    srf,
                    scale=args.scale,
                    block_size=args.block_size,
                    lambda_msi=args.lambda_msi,
                    anchor_weight=args.factor_anchor,
                )
            else:
                conditional.step(
                    core.detach(),
                    lr_hsi,
                    hr_msi,
                    srf,
                    scale=args.scale,
                    block_size=args.block_size,
                    lambda_msi=args.lambda_msi,
                    anchor_weight=args.factor_anchor,
                )

        current_factors = (
            conditional.state(
                g3_t,
                core,
                detach_g3=False,
                detach_core=False,
            )
            if joint_core_g3
            else conditional.state(core, detach_core=False)
        )
        loss, errors = guided_coupled_loss(
            current_factors,
            lr_hsi,
            hr_msi,
            srf,
            step_index=step_index,
            num_steps=len(schedule) - 1,
            args=args,
        )
        guided_loss = loss
        if joint_core_g3:
            g3_anchor_loss = float(args.g3_anchor) * F.mse_loss(
                g3_t, g3_anchor
            )
            if (
                bool(getattr(args, "coarse_to_fine_guidance", False))
                and bool(getattr(args, "coarse_to_fine_core_only", False))
            ):
                full_observation_loss, _ = coupled_loss_from_factors(
                    current_factors,
                    lr_hsi,
                    hr_msi,
                    srf,
                    scale=args.scale,
                    block_size=args.block_size,
                    lambda_msi=args.lambda_msi,
                )
                grad_core = torch.autograd.grad(
                    guided_loss, c_t, retain_graph=True
                )[0]
                grad_g3 = torch.autograd.grad(
                    full_observation_loss + g3_anchor_loss, g3_t
                )[0]
            else:
                guided_loss = guided_loss + g3_anchor_loss
                grad_core, grad_g3 = torch.autograd.grad(
                    guided_loss, (c_t, g3_t)
                )
        else:
            grad_core = torch.autograd.grad(guided_loss, c_t)[0]
            grad_g3 = None
        if args.normalize_guidance:
            grad_core = (
                grad_core
                / grad_core.norm().clamp_min(1e-8)
                * c_t.detach().norm().clamp_min(1e-6)
            )

        if objective == "pred_x0":
            eps = (c_t - sqrt_ac[t] * c0_signal) / sqrt_1_ac[t]
        else:
            eps = prediction
        noise_term = 0.0 if s == 0 else sqrt_1_ac[s] * eps
        guidance = (
            args.rho_core
            * float(args.rho_decay) ** float(t)
            * float(max(1, t - s))
            * grad_core
        )
        c_s = (sqrt_ac[s] * c0_signal + noise_term - guidance).detach()

        if joint_core_g3:
            if args.normalize_guidance:
                grad_g3 = (
                    grad_g3
                    / grad_g3.norm().clamp_min(1e-8)
                    * g3_t.detach().norm().clamp_min(1e-6)
                )
            g3_step = (
                args.rho_g3
                * float(args.rho_decay) ** float(t)
                * float(max(1, t - s))
            )
            g3 = (g3_t - g3_step * grad_g3).detach()
            max_change = float(args.g3_max_relative_change) * g3_anchor.norm()
            delta = g3 - g3_anchor
            if max_change > 0 and delta.norm() > max_change:
                g3 = g3_anchor + delta * (max_change / delta.norm())

        updated_factors = current_factors
        if update_factors:
            updated_factors = (
                conditional.state(g3, core.detach())
                if joint_core_g3
                else conditional.state(core.detach())
            )
        change = relative_factor_change(previous_factors, updated_factors)
        history.append(
            {
                "step": step_index + 1,
                "t": int(t),
                "next_t": int(s),
                "loss": float(loss.detach()),
                "lr_loss": errors["lr"],
                "msi_loss": errors["msi"],
                "skeleton_loss": errors.get("skeleton", 0.0),
                "detail_loss": errors.get("detail", 0.0),
                "detail_weight": errors.get("detail_weight", -1.0),
                "grad_core": float(grad_core.norm()),
                "grad_g3": float(grad_g3.norm()) if grad_g3 is not None else 0.0,
                "g3_relative_change": float(
                    (g3 - g3_anchor).norm() / g3_anchor.norm().clamp_min(1e-8)
                ),
                "factor_updated": bool(update_factors),
                "relative_factor_change": change,
            }
        )
        if step_index == 0 or (step_index + 1) % max(1, len(schedule) // 5) == 0:
            print(
                f"cross-attn core {step_index + 1:03d}/{len(schedule) - 1} "
                f"t={t} loss={float(loss.detach()):.5f} "
                f"lr={errors['lr']:.5f} msi={errors['msi']:.5f}"
            )
        c_t = c_s
        last_core = core.detach()
        previous_factors = updated_factors

    if joint_core_g3:
        final_factors = condition_spatial_on_core_g3(
            conditional.state(g3, last_core),
            last_core,
            g3,
            lr_hsi,
            hr_msi,
            srf,
            scale=args.scale,
            steps=args.final_factor_steps,
            lr=args.factor_lr,
            block_size=args.block_size,
            lambda_msi=args.lambda_msi,
            anchor_weight=args.factor_anchor,
        )
    else:
        final_factors = condition_factors_on_core(
            conditional.state(last_core),
            last_core,
            lr_hsi,
            hr_msi,
            srf,
            scale=args.scale,
            steps=args.final_factor_steps,
            lr=args.factor_lr,
            block_size=args.block_size,
            lambda_msi=args.lambda_msi,
            anchor_weight=args.factor_anchor,
        )
    if args.final_joint_steps > 0:
        if joint_core_g3:
            final_factors = project_core_g3(
                final_factors,
                lr_hsi,
                hr_msi,
                srf,
                scale=args.scale,
                steps=args.final_joint_steps,
                lr=args.projection_lr,
                block_size=args.block_size,
                lambda_msi=args.lambda_msi,
                core_prior_weight=args.core_prior_weight,
                g3_prior_weight=args.g3_prior_weight,
                g3_anchor=g3_anchor,
            )
        else:
            final_factors = joint_projection(
                final_factors,
                lr_hsi,
                hr_msi,
                srf,
                scale=args.scale,
                steps=args.final_joint_steps,
                lr=args.projection_lr,
                block_size=args.block_size,
                lambda_msi=args.lambda_msi,
            )
    g2_gate_report = None
    if joint_core_g3 and args.g2_gate_enabled:
        final_factors, g2_gate_report = refine_g2_with_holdout_gate(
            final_factors,
            lr_hsi,
            hr_msi,
            srf,
            scale=args.scale,
            steps=args.g2_gate_steps,
            lr=args.factor_lr,
            block_size=args.block_size,
            lambda_msi=args.lambda_msi,
            latent_dim=args.g2_gate_dim,
            anchor_weight=args.g2_gate_anchor,
            gate_scales=tuple(
                float(value) for value in args.g2_gate_scales.split(",")
            ),
            holdout_period=args.g2_gate_holdout_period,
            seed=args.seed,
        )

    final_recon = reconstruct_tw(*final_factors, block_size=args.block_size)
    recon = np.clip(
        final_recon.permute(2, 0, 1).unsqueeze(0)[0].permute(1, 2, 0).cpu().numpy(),
        0.0,
        1.0,
    )
    return {
        "final_factors": final_factors,
        "recon": recon,
        "history": history,
        "diffusion_mode": "core_g3_self_supervised" if joint_core_g3 else "core_only",
        "g2_gate": g2_gate_report,
    }


def sample_core_with_flow(
    *,
    model: torch.nn.Module,
    initial_factors: FactorState,
    stats: Dict[str, object],
    lr_hsi: torch.Tensor,
    hr_msi: torch.Tensor,
    srf: torch.Tensor,
    args,
    g3_pca: Dict[str, object] = None,
) -> Dict[str, object]:
    """Integrate a factor-conditioned core flow with observation guidance."""
    device = lr_hsi.device
    mean = stats["mean"].to(device)
    std = stats["std"].to(device)
    core_shape = tuple(stats["shape"])
    anchors = tuple(tensor.detach().clone() for tensor in initial_factors)
    joint_core_g3 = bool(getattr(args, "joint_core_g3", False))
    conditional = (
        ConditionalSpatialOptimizer(initial_factors, anchors, lr=args.factor_lr)
        if joint_core_g3
        else ConditionalFactorOptimizer(initial_factors, anchors, lr=args.factor_lr)
    )
    g3 = initial_factors[2].detach().clone()
    g3_anchor = g3.detach().clone()
    joint_latent = g3_pca is not None
    core_dim = int(mean.numel())
    if joint_latent:
        pca_mean = g3_pca["mean"].to(device)
        pca_basis = g3_pca["basis"].to(device)
        code_mean = g3_pca["code_mean"].to(device)
        code_std = g3_pca["code_std"].to(device)
        g3_shape = tuple(g3_pca["shape"])

        def encode_g3(value):
            code = (value.reshape(1, -1) - pca_mean) @ pca_basis.T
            return (code - code_mean) / code_std

        def decode_state(value):
            core_value = signal_to_core(
                value[:, :core_dim], mean, std, core_shape
            )
            normalized_code = value[:, core_dim:]
            code = normalized_code * code_std + code_mean
            g3_value = (pca_mean + code @ pca_basis).reshape(g3_shape)
            max_change = float(args.g3_max_relative_change) * g3_anchor.norm()
            delta = g3_value - g3_anchor
            if max_change > 0:
                ratio = (max_change / delta.norm().clamp_min(1e-8)).clamp(max=1.0)
                g3_value = g3_anchor + ratio * delta
            return core_value, g3_value

    clean_initial = core_to_signal(initial_factors[3], mean, std)
    if joint_latent:
        clean_initial = torch.cat([clean_initial, encode_g3(g3_anchor)], dim=1)
    generator = torch.Generator(device=device).manual_seed(args.seed + 31415)
    noise = torch.randn(
        clean_initial.shape,
        generator=generator,
        device=device,
        dtype=clean_initial.dtype,
    )
    disable_flow = bool(getattr(args, "disable_flow", False))
    start = min(0.95, max(0.0, float(args.flow_start)))
    state = clean_initial.detach().clone() if disable_flow else (
        (1.0 - start) * noise + start * clean_initial
    )
    times = torch.linspace(start, 1.0, int(args.flow_steps) + 1, device=device)

    history = []
    previous_factors = anchors
    last_core = initial_factors[3]
    model.eval()
    for step_index in range(int(args.flow_steps)):
        time = times[step_index]
        next_time = times[step_index + 1]
        dt = float(next_time - time)
        diffusion_t = int(round((1.0 - float(time)) * 1000.0))
        next_diffusion_t = int(round((1.0 - float(next_time)) * 1000.0))

        g3_t = g3.detach().requires_grad_(joint_core_g3 and not joint_latent)
        conditioning_factors = (
            conditional.state(
                g3_t,
                previous_factors[3],
                detach_g3=not joint_core_g3,
            )
            if joint_core_g3
            else conditional.state(previous_factors[3])
        )
        state = state.detach().requires_grad_(True)
        batched_time = time.reshape(1)
        observation_features = None
        if int(getattr(model, "observation_dim", 0)) > 0:
            current_core = signal_to_core(
                state[:, :core_dim], mean, std, core_shape
            )
            observation_features = observation_residual_features(
                conditioning_factors,
                current_core,
                lr_hsi,
                hr_msi,
                srf,
                scale=args.scale,
                grid_size=int(getattr(model, "observation_probe_grid", 4)),
            ).detach()
        velocity = (
            torch.zeros_like(state)
            if disable_flow
            else model(
                state,
                batched_time,
                conditioning_factors,
                observation_features,
            )
        )
        endpoint_signal = state + (1.0 - time) * velocity
        if joint_latent:
            core, predicted_g3 = decode_state(endpoint_signal)
        else:
            core = signal_to_core(endpoint_signal, mean, std, core_shape)
            predicted_g3 = g3_t

        update_factors = (
            args.factor_interval > 0
            and step_index % args.factor_interval == 0
            and diffusion_t <= args.factor_start_t
        )
        if update_factors:
            if joint_core_g3:
                conditional.step(
                    predicted_g3.detach(),
                    core.detach(),
                    lr_hsi,
                    hr_msi,
                    srf,
                    scale=args.scale,
                    block_size=args.block_size,
                    lambda_msi=args.lambda_msi,
                    anchor_weight=args.factor_anchor,
                )
            else:
                conditional.step(
                    core.detach(),
                    lr_hsi,
                    hr_msi,
                    srf,
                    scale=args.scale,
                    block_size=args.block_size,
                    lambda_msi=args.lambda_msi,
                    anchor_weight=args.factor_anchor,
                )

        current_factors = (
            conditional.state(
                predicted_g3,
                core,
                detach_g3=False,
                detach_core=False,
            )
            if joint_core_g3
            else conditional.state(core, detach_core=False)
        )
        loss, errors = guided_coupled_loss(
            current_factors,
            lr_hsi,
            hr_msi,
            srf,
            step_index=step_index,
            num_steps=int(args.flow_steps),
            args=args,
        )
        guided_loss = loss
        if joint_core_g3:
            guided_loss = guided_loss + float(args.g3_anchor) * F.mse_loss(
                predicted_g3, g3_anchor
            )
            if joint_latent:
                grad_core = torch.autograd.grad(guided_loss, state)[0]
                grad_g3 = grad_core[:, core_dim:]
            else:
                grad_core, grad_g3 = torch.autograd.grad(
                    guided_loss,
                    (state, g3_t),
                )
        else:
            grad_core = torch.autograd.grad(guided_loss, state)[0]
            grad_g3 = None
        if args.normalize_guidance:
            grad_core = (
                grad_core
                / grad_core.norm().clamp_min(1e-8)
                * state.detach().norm().clamp_min(1e-6)
            )
        diffusion_delta = float(max(1, diffusion_t - next_diffusion_t))
        guidance = (
            args.rho_core
            * float(args.rho_decay) ** float(diffusion_t)
            * diffusion_delta
            * grad_core
        )
        next_state = (state + dt * velocity - guidance).detach()

        if joint_core_g3 and not joint_latent:
            if args.normalize_guidance:
                grad_g3 = (
                    grad_g3
                    / grad_g3.norm().clamp_min(1e-8)
                    * g3_t.detach().norm().clamp_min(1e-6)
                )
            g3_step = (
                args.rho_g3
                * float(args.rho_decay) ** float(diffusion_t)
                * diffusion_delta
            )
            g3 = (g3_t - g3_step * grad_g3).detach()
            max_change = float(args.g3_max_relative_change) * g3_anchor.norm()
            delta = g3 - g3_anchor
            if max_change > 0 and delta.norm() > max_change:
                g3 = g3_anchor + delta * (max_change / delta.norm())
        elif joint_latent:
            g3 = predicted_g3.detach()

        updated_factors = current_factors
        if update_factors:
            updated_factors = (
                conditional.state(g3, core.detach())
                if joint_core_g3
                else conditional.state(core.detach())
            )
        history.append(
            {
                "step": step_index + 1,
                "flow_time": float(time),
                "next_flow_time": float(next_time),
                "t": diffusion_t,
                "next_t": next_diffusion_t,
                "loss": float(loss.detach()),
                "lr_loss": errors["lr"],
                "msi_loss": errors["msi"],
                "skeleton_loss": errors.get("skeleton", 0.0),
                "detail_loss": errors.get("detail", 0.0),
                "detail_weight": errors.get("detail_weight", -1.0),
                "grad_core": float(grad_core.norm()),
                "grad_g3": float(grad_g3.norm()) if grad_g3 is not None else 0.0,
                "g3_relative_change": float(
                    (g3 - g3_anchor).norm() / g3_anchor.norm().clamp_min(1e-8)
                ),
                "factor_updated": bool(update_factors),
                "relative_factor_change": relative_factor_change(
                    previous_factors,
                    updated_factors,
                ),
            }
        )
        if step_index == 0 or (step_index + 1) % max(1, int(args.flow_steps) // 5) == 0:
            print(
                f"flow core {step_index + 1:03d}/{int(args.flow_steps)} "
                f"t={float(time):.3f} loss={float(loss.detach()):.5f} "
                f"lr={errors['lr']:.5f} msi={errors['msi']:.5f}"
            )
        state = next_state
        last_core = core.detach()
        previous_factors = updated_factors

    if joint_latent:
        last_core, g3 = decode_state(state)
        last_core = last_core.detach()
        g3 = g3.detach()
    else:
        last_core = signal_to_core(state, mean, std, core_shape).detach()
    if joint_core_g3:
        final_factors = condition_spatial_on_core_g3(
            conditional.state(g3, last_core),
            last_core,
            g3,
            lr_hsi,
            hr_msi,
            srf,
            scale=args.scale,
            steps=args.final_factor_steps,
            lr=args.factor_lr,
            block_size=args.block_size,
            lambda_msi=args.lambda_msi,
            anchor_weight=args.factor_anchor,
        )
    else:
        final_factors = condition_factors_on_core(
            conditional.state(last_core),
            last_core,
            lr_hsi,
            hr_msi,
            srf,
            scale=args.scale,
            steps=args.final_factor_steps,
            lr=args.factor_lr,
            block_size=args.block_size,
            lambda_msi=args.lambda_msi,
            anchor_weight=args.factor_anchor,
        )
    if args.final_joint_steps > 0:
        if joint_core_g3:
            final_factors = project_core_g3(
                final_factors,
                lr_hsi,
                hr_msi,
                srf,
                scale=args.scale,
                steps=args.final_joint_steps,
                lr=args.projection_lr,
                block_size=args.block_size,
                lambda_msi=args.lambda_msi,
                core_prior_weight=args.core_prior_weight,
                g3_prior_weight=args.g3_prior_weight,
                g3_anchor=g3_anchor,
                structure_ssim_weight=float(
                    getattr(args, "structure_ssim_weight", 0.0)
                ),
                structure_gradient_weight=float(
                    getattr(args, "structure_gradient_weight", 0.0)
                ),
                structure_sam_weight=float(
                    getattr(args, "structure_sam_weight", 0.0)
                ),
                structure_last_steps=int(
                    getattr(args, "structure_last_steps", 0)
                ),
                alternating_updates=bool(
                    getattr(args, "factor_alternating_updates", False)
                ),
            )
        else:
            final_factors = joint_projection(
                final_factors,
                lr_hsi,
                hr_msi,
                srf,
                scale=args.scale,
                steps=args.final_joint_steps,
                lr=args.projection_lr,
                block_size=args.block_size,
                lambda_msi=args.lambda_msi,
            )
    g2_gate_report = None
    if joint_core_g3 and args.g2_gate_enabled:
        final_factors, g2_gate_report = refine_g2_with_holdout_gate(
            final_factors,
            lr_hsi,
            hr_msi,
            srf,
            scale=args.scale,
            steps=args.g2_gate_steps,
            lr=args.factor_lr,
            block_size=args.block_size,
            lambda_msi=args.lambda_msi,
            latent_dim=args.g2_gate_dim,
            anchor_weight=args.g2_gate_anchor,
            gate_scales=tuple(
                float(value) for value in args.g2_gate_scales.split(",")
            ),
            holdout_period=args.g2_gate_holdout_period,
            seed=args.seed,
        )
    final_recon = reconstruct_tw(*final_factors, block_size=args.block_size)
    recon = np.clip(
        final_recon.permute(2, 0, 1).unsqueeze(0)[0].permute(1, 2, 0).cpu().numpy(),
        0.0,
        1.0,
    )
    return {
        "final_factors": final_factors,
        "recon": recon,
        "history": history,
        "diffusion_mode": (
            "core_g3_latent_flow"
            if joint_latent
            else ("core_flow_g3_guided" if joint_core_g3 else "core_flow")
        ),
        "g2_gate": g2_gate_report,
    }
