from __future__ import annotations

import argparse
import csv
import json
import os
from types import SimpleNamespace

import numpy as np
import scipy.io as sio
import torch

from ctw_ca.data import load_observations, make_synthetic_observations
from ctw_ca.flow import ConditionalCoreFlow
from ctw_ca.metrics import calculate_metrics
from ctw_ca.model import CrossAttentionCoreDenoiser
from ctw_ca.sampling import (
    calibrate_factor_adapter,
    sample_core_with_cross_attention,
    sample_core_with_flow,
)
from ctw_ca.tw import factor_shapes, init_from_observations, reconstruct_tw


def parse_int_list(value: str):
    return tuple(int(item) for item in value.replace(" ", "").split(","))


def load_json(path: str):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_arg(cli, config, name, default=None):
    value = getattr(cli, name, None)
    if value is not None:
        return value
    if name in config:
        return config[name]
    return default


def build_config(cli):
    config = load_json(cli.config)
    cross = config.get("cross_attention", {})
    flow = config.get("flow", {})
    joint = config.get("joint_core_g3", {})
    coarse = config.get("coarse_to_fine_guidance", {})
    return SimpleNamespace(
        data=resolve_arg(cli, config, "data", "data/cave/cave.mat"),
        srf=resolve_arg(cli, config, "srf", "data/cave/srf_est.mat"),
        model_dir=resolve_arg(
            cli,
            config,
            "model_dir",
            "result/cave/factor_diffusion_ctw_v2_r888_l222",
        ),
        metadata=resolve_arg(cli, config, "metadata", ""),
        adapter_checkpoint=resolve_arg(cli, config, "adapter_checkpoint", ""),
        base_checkpoint=resolve_arg(cli, config, "base_checkpoint", ""),
        initial_factors=resolve_arg(cli, config, "initial_factors", ""),
        out_dir=resolve_arg(
            cli,
            config,
            "out_dir",
            "result/cave/ctw_cross_attention",
        ),
        device=resolve_arg(cli, config, "device", "auto"),
        seed=int(resolve_arg(cli, config, "seed", 0)),
        smoke=bool(resolve_arg(cli, config, "smoke", False)),
        crop=int(resolve_arg(cli, config, "crop", 0)),
        scale=int(resolve_arg(cli, config, "scale", 4)),
        ring_ranks=resolve_arg(cli, config, "ring_ranks", "8,8,8"),
        core_ranks=resolve_arg(cli, config, "core_ranks", "2,2,2"),
        sampler=resolve_arg(cli, config, "sampler", "diffusion"),
        flow_checkpoint=resolve_arg(cli, flow, "flow_checkpoint", ""),
        flow_steps=int(resolve_arg(cli, flow, "flow_steps", 20)),
        flow_start=float(resolve_arg(cli, flow, "flow_start", 0.4)),
        t_start=int(resolve_arg(cli, config, "t_start", 600)),
        ddim_steps=int(resolve_arg(cli, config, "ddim_steps", 100)),
        init_projection_steps=int(
            resolve_arg(cli, config, "init_projection_steps", 100)
        ),
        block_size=int(resolve_arg(cli, config, "block_size", 16)),
        lambda_msi=float(resolve_arg(cli, config, "lambda_msi", 1.0)),
        projection_lr=float(resolve_arg(cli, config, "projection_lr", 2e-3)),
        factor_lr=float(resolve_arg(cli, config, "factor_lr", 2e-3)),
        factor_interval=int(resolve_arg(cli, config, "factor_interval", 2)),
        factor_start_t=int(resolve_arg(cli, config, "factor_start_t", 400)),
        factor_anchor=float(resolve_arg(cli, config, "factor_anchor", 1e-3)),
        final_factor_steps=int(
            resolve_arg(cli, config, "final_factor_steps", 200)
        ),
        final_joint_steps=int(
            resolve_arg(cli, config, "final_joint_steps", 300)
        ),
        rho_core=float(resolve_arg(cli, config, "rho_core", 0.01)),
        rho_decay=float(resolve_arg(cli, config, "rho_decay", 0.998)),
        normalize_guidance=bool(
            resolve_arg(cli, config, "normalize_guidance", True)
        ),
        coarse_to_fine_guidance=bool(
            resolve_arg(
                cli,
                coarse,
                "coarse_to_fine_guidance",
                coarse.get("enabled", False),
            )
        ),
        coarse_to_fine_core_only=bool(
            resolve_arg(
                cli,
                coarse,
                "coarse_to_fine_core_only",
                coarse.get("core_only", False),
            )
        ),
        coarse_lowpass_kernel=int(
            resolve_arg(
                cli,
                coarse,
                "coarse_lowpass_kernel",
                coarse.get("lowpass_kernel", 5),
            )
        ),
        coarse_transition_start=float(
            resolve_arg(
                cli,
                coarse,
                "coarse_transition_start",
                coarse.get("transition_start", 0.4),
            )
        ),
        coarse_transition_end=float(
            resolve_arg(
                cli,
                coarse,
                "coarse_transition_end",
                coarse.get("transition_end", 0.75),
            )
        ),
        coarse_full_loss_weight=float(
            resolve_arg(
                cli,
                coarse,
                "coarse_full_loss_weight",
                coarse.get("full_loss_weight", 0.2),
            )
        ),
        coarse_detail_gradient_weight=float(
            resolve_arg(
                cli,
                coarse,
                "coarse_detail_gradient_weight",
                coarse.get("detail_gradient_weight", 0.25),
            )
        ),
        joint_core_g3=bool(
            resolve_arg(cli, joint, "joint_core_g3", joint.get("enabled", False))
        ),
        rho_g3=float(resolve_arg(cli, joint, "rho_g3", 5e-5)),
        g3_anchor=float(resolve_arg(cli, joint, "g3_anchor", 0.1)),
        g3_max_relative_change=float(
            resolve_arg(cli, joint, "g3_max_relative_change", 0.25)
        ),
        core_prior_weight=float(
            resolve_arg(cli, joint, "core_prior_weight", 0.1)
        ),
        g3_prior_weight=float(
            resolve_arg(cli, joint, "g3_prior_weight", 0.1)
        ),
        structure_ssim_weight=float(
            resolve_arg(cli, joint, "structure_ssim_weight", 0.0)
        ),
        structure_gradient_weight=float(
            resolve_arg(cli, joint, "structure_gradient_weight", 0.0)
        ),
        structure_sam_weight=float(
            resolve_arg(cli, joint, "structure_sam_weight", 0.0)
        ),
        structure_last_steps=int(
            resolve_arg(cli, joint, "structure_last_steps", 0)
        ),
        factor_alternating_updates=bool(
            resolve_arg(cli, joint, "factor_alternating_updates", False)
        ),
        g2_gate_enabled=bool(resolve_arg(cli, joint, "g2_gate_enabled", False)),
        g2_gate_dim=int(resolve_arg(cli, joint, "g2_gate_dim", 16)),
        g2_gate_steps=int(resolve_arg(cli, joint, "g2_gate_steps", 50)),
        g2_gate_anchor=float(resolve_arg(cli, joint, "g2_gate_anchor", 0.1)),
        g2_gate_scales=resolve_arg(cli, joint, "g2_gate_scales", "0,0.25,0.5,0.75,1"),
        g2_gate_holdout_period=int(
            resolve_arg(cli, joint, "g2_gate_holdout_period", 20)
        ),
        d_model=int(resolve_arg(cli, cross, "d_model", 128)),
        num_heads=int(resolve_arg(cli, cross, "num_heads", 4)),
        tokens_per_factor=int(
            resolve_arg(cli, cross, "tokens_per_factor", 4)
        ),
        gate_init=float(resolve_arg(cli, cross, "gate_init", 0.0)),
        interaction_tokens=bool(
            resolve_arg(cli, cross, "interaction_tokens", False)
        ),
        adapter_tta=bool(resolve_arg(cli, cross, "adapter_tta", False)),
        adapter_tta_steps=int(resolve_arg(cli, cross, "adapter_tta_steps", 20)),
        adapter_tta_lr=float(resolve_arg(cli, cross, "adapter_tta_lr", 1e-4)),
        adapter_tta_prior=float(
            resolve_arg(cli, cross, "adapter_tta_prior", 0.1)
        ),
    )


def build_core_stats(metadata):
    core = metadata["stats"]["core"]
    return {
        "mean": core["mean"].float(),
        "std": core["std"].float().clamp_min(1e-6),
        "shape": tuple(core["shape"]),
    }


def save_outputs(result, metrics, out_dir, args):
    os.makedirs(out_dir, exist_ok=True)
    factors = result["final_factors"]
    payload = {
        "recon": result["recon"],
        "recon_initial": result["initial_recon"],
        "G1": factors[0].cpu().numpy(),
        "G2": factors[1].cpu().numpy(),
        "G3": factors[2].cpu().numpy(),
        "C": factors[3].cpu().numpy(),
    }
    if result["gt"] is not None:
        payload["HRHSI"] = result["gt"]
    payload.update(metrics)
    sio.savemat(os.path.join(out_dir, "recon_info.mat"), payload)
    torch.save(
        {
            "initial_factors": [
                tensor.cpu() for tensor in result["initial_factors"]
            ],
            "final_factors": [tensor.cpu() for tensor in factors],
            "metrics": metrics,
            "history": result["history"],
            "g2_gate": result.get("g2_gate"),
        },
        os.path.join(out_dir, "factors.pt"),
    )
    if result["history"]:
        with open(
            os.path.join(out_dir, "core_reverse_history.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(result["history"][0]),
            )
            writer.writeheader()
            writer.writerows(result["history"])
    with open(
        os.path.join(out_dir, "run_config.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "args": vars(args),
                "metrics": metrics,
                "g2_gate": result.get("g2_gate"),
                "adapter_tta": result.get("adapter_tta"),
            },
            handle,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Cross-attention core-tensor diffusion for CTW fusion"
    )
    parser.add_argument("--config", default="")
    parser.add_argument("--data", default=None)
    parser.add_argument("--srf", default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--adapter_checkpoint", default=None)
    parser.add_argument("--base_checkpoint", default=None)
    parser.add_argument("--initial_factors", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", default=None)
    parser.add_argument("--crop", type=int, default=None)
    parser.add_argument("--scale", type=int, default=None)
    parser.add_argument("--ring_ranks", default=None)
    parser.add_argument("--core_ranks", default=None)
    parser.add_argument("--sampler", choices=("diffusion", "flow"), default=None)
    parser.add_argument("--flow_checkpoint", default=None)
    parser.add_argument("--flow_steps", type=int, default=None)
    parser.add_argument("--flow_start", type=float, default=None)
    parser.add_argument("--t_start", type=int, default=None)
    parser.add_argument("--ddim_steps", type=int, default=None)
    parser.add_argument("--init_projection_steps", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--lambda_msi", type=float, default=None)
    parser.add_argument("--projection_lr", type=float, default=None)
    parser.add_argument("--factor_lr", type=float, default=None)
    parser.add_argument("--factor_interval", type=int, default=None)
    parser.add_argument("--factor_start_t", type=int, default=None)
    parser.add_argument("--factor_anchor", type=float, default=None)
    parser.add_argument("--final_factor_steps", type=int, default=None)
    parser.add_argument("--final_joint_steps", type=int, default=None)
    parser.add_argument("--rho_core", type=float, default=None)
    parser.add_argument("--rho_decay", type=float, default=None)
    parser.add_argument(
        "--coarse_to_fine_guidance", action="store_true", default=None
    )
    parser.add_argument(
        "--no_coarse_to_fine_guidance",
        dest="coarse_to_fine_guidance",
        action="store_false",
    )
    parser.add_argument(
        "--coarse_to_fine_core_only", action="store_true", default=None
    )
    parser.add_argument(
        "--no_coarse_to_fine_core_only",
        dest="coarse_to_fine_core_only",
        action="store_false",
    )
    parser.add_argument("--coarse_lowpass_kernel", type=int, default=None)
    parser.add_argument("--coarse_transition_start", type=float, default=None)
    parser.add_argument("--coarse_transition_end", type=float, default=None)
    parser.add_argument("--coarse_full_loss_weight", type=float, default=None)
    parser.add_argument(
        "--coarse_detail_gradient_weight", type=float, default=None
    )
    parser.add_argument("--normalize_guidance", action="store_true", default=None)
    parser.add_argument(
        "--no_normalize_guidance",
        dest="normalize_guidance",
        action="store_false",
    )
    parser.add_argument("--joint_core_g3", action="store_true", default=None)
    parser.add_argument(
        "--core_only", dest="joint_core_g3", action="store_false"
    )
    parser.add_argument("--rho_g3", type=float, default=None)
    parser.add_argument("--g3_anchor", type=float, default=None)
    parser.add_argument("--g3_max_relative_change", type=float, default=None)
    parser.add_argument("--core_prior_weight", type=float, default=None)
    parser.add_argument("--g3_prior_weight", type=float, default=None)
    parser.add_argument("--structure_ssim_weight", type=float, default=None)
    parser.add_argument("--structure_gradient_weight", type=float, default=None)
    parser.add_argument("--structure_sam_weight", type=float, default=None)
    parser.add_argument("--structure_last_steps", type=int, default=None)
    parser.add_argument(
        "--factor_alternating_updates", action="store_true", default=None
    )
    parser.add_argument(
        "--no_factor_alternating_updates",
        dest="factor_alternating_updates",
        action="store_false",
    )
    parser.add_argument("--g2_gate_enabled", action="store_true", default=None)
    parser.add_argument("--no_g2_gate", dest="g2_gate_enabled", action="store_false")
    parser.add_argument("--g2_gate_dim", type=int, default=None)
    parser.add_argument("--g2_gate_steps", type=int, default=None)
    parser.add_argument("--g2_gate_anchor", type=float, default=None)
    parser.add_argument("--g2_gate_scales", default=None)
    parser.add_argument("--g2_gate_holdout_period", type=int, default=None)
    parser.add_argument("--adapter_tta", action="store_true", default=None)
    parser.add_argument("--no_adapter_tta", dest="adapter_tta", action="store_false")
    parser.add_argument("--adapter_tta_steps", type=int, default=None)
    parser.add_argument("--adapter_tta_lr", type=float, default=None)
    parser.add_argument("--adapter_tta_prior", type=float, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--tokens_per_factor", type=int, default=None)
    parser.add_argument("--gate_init", type=float, default=None)
    parser.add_argument("--interaction_tokens", action="store_true", default=None)
    parser.add_argument(
        "--no_interaction_tokens",
        dest="interaction_tokens",
        action="store_false",
    )
    cli = parser.parse_args()
    args = build_config(cli)

    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ring_ranks = parse_int_list(args.ring_ranks)
    core_ranks = parse_int_list(args.core_ranks)
    if len(ring_ranks) != 3 or len(core_ranks) != 3:
        raise ValueError("ring_ranks and core_ranks need three values")

    if args.smoke:
        gt, lr_hsi, hr_msi, srf = make_synthetic_observations(device)
        ring_ranks = (2, 2, 2)
        core_ranks = (2, 2, 2)
        args.t_start = 200
        args.ddim_steps = 8
        args.init_projection_steps = 4
        args.factor_interval = 2
        args.factor_start_t = 200
        args.final_factor_steps = 4
        args.final_joint_steps = 4
        args.rho_core = 0.005
        args.rho_g3 = 1e-4
        metadata = {
            "stats": {
                "core": {
                    "mean": torch.zeros(1, 8),
                    "std": torch.ones(1, 8),
                    "shape": (2, 2, 2),
                }
            },
            "args": {"objective": "pred_x0"},
        }
        base_checkpoint = ""
        adapter_checkpoint = ""
    else:
        gt, lr_hsi, hr_msi, srf = load_observations(
            args.data,
            args.srf,
            device,
            crop_size=args.crop,
            scale=args.scale,
        )
        metadata_path = args.metadata or os.path.join(
            args.model_dir,
            "training_metadata.pt",
        )
        metadata = torch.load(metadata_path, map_location="cpu")
        base_checkpoint = args.base_checkpoint or os.path.join(
            args.model_dir,
            "model_core.pt",
        )
        adapter_checkpoint = args.adapter_checkpoint

    if gt is not None:
        height, width, channels = gt.shape
    else:
        height, width = hr_msi.shape[2:]
        channels = lr_hsi.shape[1]

    shapes = factor_shapes(
        height,
        width,
        channels,
        ring_ranks,
        core_ranks,
    )
    flow_checkpoint = None
    if args.sampler == "flow":
        if not args.flow_checkpoint:
            raise ValueError("flow sampler requires --flow_checkpoint")
        flow_checkpoint = torch.load(args.flow_checkpoint, map_location="cpu")
        stats = {
            "mean": flow_checkpoint["stats"]["mean"].float(),
            "std": flow_checkpoint["stats"]["std"].float().clamp_min(1e-6),
            "shape": tuple(flow_checkpoint["stats"]["shape"]),
        }
    else:
        stats = build_core_stats(metadata)
    if tuple(stats["shape"]) != tuple(core_ranks):
        raise ValueError(
            f"checkpoint core shape {tuple(stats['shape'])} "
            f"does not match core_ranks {core_ranks}"
        )

    core_dim = int(stats["mean"].numel())
    model_state_dim = core_dim
    if args.sampler == "flow" and flow_checkpoint.get("g3_pca") is not None:
        model_state_dim += int(flow_checkpoint["g3_pca"]["latent_dim"])
    if args.sampler == "flow":
        flow_args = flow_checkpoint.get("args", {})
        model = ConditionalCoreFlow(
            model_state_dim,
            shapes,
            d_model=int(flow_args.get("d_model", args.d_model)),
            num_heads=int(flow_args.get("num_heads", args.num_heads)),
            tokens_per_factor=int(
                flow_args.get("tokens_per_factor", args.tokens_per_factor)
            ),
            interaction_tokens=bool(
                flow_args.get("interaction_tokens", args.interaction_tokens)
            ),
            observation_dim=int(flow_args.get("observation_dim", 0)),
        ).to(device)
        model.observation_probe_grid = int(
            flow_args.get("observation_probe_grid", 4)
        )
        model.load_state_dict(flow_checkpoint["model"])
    else:
        model = CrossAttentionCoreDenoiser(
            core_dim,
            shapes,
            d_model=args.d_model,
            num_heads=args.num_heads,
            tokens_per_factor=args.tokens_per_factor,
            gate_init=args.gate_init,
            interaction_tokens=args.interaction_tokens,
        ).to(device)
        if base_checkpoint:
            model.load_base_checkpoint(base_checkpoint, map_location=device)
        if adapter_checkpoint:
            model.load_adapter_checkpoint(adapter_checkpoint, map_location=device)
    model.eval()

    if args.initial_factors and os.path.exists(args.initial_factors):
        stored = torch.load(args.initial_factors, map_location=device)
        values = stored.get("initial_factors", stored) if isinstance(stored, dict) else stored
        initial_factors = tuple(value.to(device) for value in values)
        if tuple(initial_factors[3].shape) != tuple(core_ranks):
            raise ValueError("shared initialization core shape does not match core_ranks")
    else:
        initial_factors = init_from_observations(
            lr_hsi,
            hr_msi,
            srf,
            ring_ranks=ring_ranks,
            core_ranks=core_ranks,
            scale=args.scale,
            projection_steps=args.init_projection_steps,
            projection_lr=args.projection_lr,
            block_size=args.block_size,
            lambda_msi=args.lambda_msi,
            seed=args.seed,
        )
        if args.initial_factors:
            os.makedirs(os.path.dirname(os.path.abspath(args.initial_factors)), exist_ok=True)
            torch.save(
                {
                    "initial_factors": [value.detach().cpu() for value in initial_factors],
                    "seed": args.seed,
                    "ring_ranks": ring_ranks,
                    "core_ranks": core_ranks,
                },
                args.initial_factors,
            )
    initial_recon = reconstruct_tw(
        *initial_factors,
        block_size=args.block_size,
    )
    initial_recon_np = np.clip(
        initial_recon.permute(2, 0, 1)
        .unsqueeze(0)[0]
        .permute(1, 2, 0)
        .cpu()
        .numpy(),
        0.0,
        1.0,
    )

    objective = metadata.get("args", {}).get("objective", "pred_x0")
    adapter_tta_report = None
    if args.adapter_tta and args.sampler == "diffusion":
        adapter_tta_report = calibrate_factor_adapter(
            model=model,
            objective=objective,
            initial_factors=initial_factors,
            stats=stats,
            lr_hsi=lr_hsi,
            hr_msi=hr_msi,
            srf=srf,
            args=args,
        )
        print("adapter_tta", adapter_tta_report)
    if args.sampler == "flow":
        sampled = sample_core_with_flow(
            model=model,
            initial_factors=initial_factors,
            stats=stats,
            lr_hsi=lr_hsi,
            hr_msi=hr_msi,
            srf=srf,
            args=args,
            g3_pca=flow_checkpoint.get("g3_pca"),
        )
    else:
        sampled = sample_core_with_cross_attention(
            model=model,
            objective=objective,
            initial_factors=initial_factors,
            stats=stats,
            lr_hsi=lr_hsi,
            hr_msi=hr_msi,
            srf=srf,
            args=args,
        )
    metrics = {}
    if gt is not None:
        metrics = calculate_metrics(
            gt,
            sampled["recon"],
            args.scale,
        )

    result = {
        "gt": gt,
        "initial_factors": initial_factors,
        "initial_recon": initial_recon_np,
        "adapter_tta": adapter_tta_report,
        **sampled,
    }
    save_outputs(result, metrics, args.out_dir, args)
    print("metrics", metrics)
    print("saved:", args.out_dir)


if __name__ == "__main__":
    main()
