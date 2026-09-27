from __future__ import annotations

import argparse
import os

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F

from ctw_ca.data import load_factor_corpus
from ctw_ca.flow import ConditionalCoreFlow
from ctw_ca.tw import (
    factor_observation_residual_features,
    probe_reconstruction_losses,
)


def parse_int_list(value: str):
    return tuple(int(item) for item in value.replace(" ", "").split(","))


def main():
    parser = argparse.ArgumentParser(
        description="Train a factor-conditioned rectified flow for the TW core"
    )
    parser.add_argument(
        "--code_pt",
        default="result/cave/interaction_factor_corpus/tw_codes.pt",
    )
    parser.add_argument(
        "--out_checkpoint",
        default="result/cave/core_flow_multisource/model.pt",
    )
    parser.add_argument("--dataset_prefix", default="TRAIN/")
    parser.add_argument("--val_prefix", default="VAL/")
    parser.add_argument("--ring_ranks", default="8,8,8")
    parser.add_argument("--core_ranks", default="2,2,2")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--factor_noise", type=float, default=0.005)
    parser.add_argument("--condition_dropout", type=float, default=0.1)
    parser.add_argument("--endpoint_weight", type=float, default=0.1)
    parser.add_argument("--match_weight", type=float, default=0.05)
    parser.add_argument("--match_margin", type=float, default=0.02)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--tokens_per_factor", type=int, default=4)
    parser.add_argument("--interaction_tokens", action="store_true")
    parser.add_argument("--observation_residuals", action="store_true")
    parser.add_argument("--observation_probe_grid", type=int, default=4)
    parser.add_argument("--srf", default="data/cave/srf_est.mat")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--tw_recon_weight", type=float, default=0.0)
    parser.add_argument("--observation_weight", type=float, default=0.0)
    parser.add_argument(
        "--g3_latent_dim",
        type=int,
        default=0,
        help="PCA dimensions appended to the core flow state; 0 keeps C-only flow",
    )
    parser.add_argument("--validate_every", type=int, default=100)
    parser.add_argument("--validation_batches", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ring_ranks = parse_int_list(args.ring_ranks)
    core_ranks = parse_int_list(args.core_ranks)
    train = load_factor_corpus(
        args.code_pt,
        ring_ranks=ring_ranks,
        core_ranks=core_ranks,
        dataset_prefix=args.dataset_prefix,
    )
    validation = load_factor_corpus(
        args.code_pt,
        ring_ranks=ring_ranks,
        core_ranks=core_ranks,
        dataset_prefix=args.val_prefix,
    )
    train_core_flat = train[3].reshape(train[3].shape[0], -1)
    core_mean = train_core_flat.mean(dim=0, keepdim=True)
    core_std = train_core_flat.std(dim=0, keepdim=True).clamp_min(1e-6)
    g3_mean = None
    g3_basis = None
    g3_code_mean = None
    g3_code_std = None
    if args.g3_latent_dim > 0:
        g3_flat = train[2].reshape(train[2].shape[0], -1)
        g3_mean = g3_flat.mean(dim=0, keepdim=True)
        centered_g3 = g3_flat - g3_mean
        latent_dim = min(
            int(args.g3_latent_dim),
            centered_g3.shape[0] - 1,
            centered_g3.shape[1],
        )
        _, _, basis_columns = torch.pca_lowrank(
            centered_g3,
            q=latent_dim,
            center=False,
        )
        g3_basis = basis_columns.T.contiguous()
        train_codes = centered_g3 @ basis_columns
        g3_code_mean = train_codes.mean(dim=0, keepdim=True)
        g3_code_std = train_codes.std(dim=0, keepdim=True).clamp_min(1e-6)
    shapes = {
        "G1": tuple(train[0].shape[1:]),
        "G2": tuple(train[1].shape[1:]),
        "G3": tuple(train[2].shape[1:]),
        "core": tuple(train[3].shape[1:]),
    }
    state_dim = train_core_flat.shape[1] + (
        0 if g3_basis is None else int(g3_basis.shape[0])
    )
    observation_dim = 0
    srf = None
    if (
        args.observation_residuals
        or args.tw_recon_weight > 0
        or args.observation_weight > 0
    ):
        srf = torch.from_numpy(np.transpose(sio.loadmat(args.srf)["srf"]))
        srf = srf.float().to(device)
    if args.observation_residuals:
        observation_dim = 2 * (int(train[2].shape[2]) + int(srf.shape[1]))
    args.observation_dim = observation_dim
    model = ConditionalCoreFlow(
        state_dim,
        shapes,
        d_model=args.d_model,
        num_heads=args.num_heads,
        tokens_per_factor=args.tokens_per_factor,
        interaction_tokens=args.interaction_tokens,
        observation_dim=observation_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    mean = core_mean.to(device)
    std = core_std.to(device)
    if g3_basis is not None:
        g3_mean_device = g3_mean.to(device)
        g3_basis_device = g3_basis.to(device)
        g3_code_mean_device = g3_code_mean.to(device)
        g3_code_std_device = g3_code_std.to(device)

    def draw_batch(values, batch_size):
        indices = torch.randint(0, values[0].shape[0], (batch_size,))
        return tuple(value[indices].to(device) for value in values)

    def objective(factors, *, training):
        batch_size = factors[3].shape[0]
        core_clean = (factors[3].reshape(batch_size, -1) - mean) / std
        if g3_basis is not None:
            g3_flat = factors[2].reshape(batch_size, -1)
            g3_codes = (g3_flat - g3_mean_device) @ g3_basis_device.T
            g3_clean = (
                g3_codes - g3_code_mean_device
            ) / g3_code_std_device
            clean = torch.cat([core_clean, g3_clean], dim=1)
        else:
            clean = core_clean
        noise = torch.randn_like(clean)
        time = torch.rand(batch_size, 1, device=device)
        state = (1.0 - time) * noise + time * clean
        target_velocity = clean - noise
        conditioned = factors
        if training and args.condition_dropout > 0:
            keep = (
                torch.rand(batch_size, 1, 1, 1, 1, device=device)
                >= args.condition_dropout
            )
            conditioned = (
                factors[0] * keep,
                factors[1] * keep,
                factors[2] * keep,
                factors[3],
            )
        observation_features = None
        if args.observation_residuals:
            current_core = (
                state[:, : train_core_flat.shape[1]] * std + mean
            ).reshape_as(factors[3])
            observation_features = factor_observation_residual_features(
                factors,
                current_core,
                factors[3],
                srf,
                scale=args.scale,
                grid_size=args.observation_probe_grid,
            ).detach()
        prediction = model(
            state,
            time[:, 0],
            conditioned,
            observation_features,
        )
        correct_per = (prediction - target_velocity).square().mean(dim=1)
        flow_loss = correct_per.mean()
        endpoint = state + (1.0 - time) * prediction
        endpoint_loss = F.mse_loss(endpoint, clean)

        permutation = torch.roll(torch.arange(batch_size, device=device), 1)
        shuffled = (
            factors[0][permutation],
            factors[1][permutation],
            factors[2][permutation],
            factors[3],
        )
        shuffled_prediction = model(
            state,
            time[:, 0],
            shuffled,
            observation_features,
        )
        shuffled_per = (
            shuffled_prediction - target_velocity
        ).square().mean(dim=1)
        match_loss = F.relu(
            float(args.match_margin) + correct_per - shuffled_per
        ).mean()
        reconstruction_loss = state.new_zeros(())
        observation_loss = state.new_zeros(())
        if args.tw_recon_weight > 0 or args.observation_weight > 0:
            predicted_core = (
                endpoint[:, : train_core_flat.shape[1]] * std + mean
            ).reshape_as(factors[3])
            reconstruction_loss, observation_loss = probe_reconstruction_losses(
                factors,
                predicted_core,
                factors[3],
                srf,
                scale=args.scale,
                grid_size=args.observation_probe_grid,
            )
        total = (
            flow_loss
            + float(args.endpoint_weight) * endpoint_loss
            + float(args.match_weight) * match_loss
            + float(args.tw_recon_weight) * reconstruction_loss
            + float(args.observation_weight) * observation_loss
        )
        return total, {
            "flow": float(flow_loss.detach()),
            "endpoint": float(endpoint_loss.detach()),
            "match": float(match_loss.detach()),
            "reconstruction": float(reconstruction_loss.detach()),
            "observation": float(observation_loss.detach()),
        }

    losses = []
    validation_history = []
    best_validation = float("inf")
    best_state = None
    model.train()
    for step in range(1, args.steps + 1):
        factors = draw_batch(train, args.batch_size)
        if args.factor_noise > 0:
            factors = (
                factors[0] + args.factor_noise * torch.randn_like(factors[0]),
                factors[1] + args.factor_noise * torch.randn_like(factors[1]),
                factors[2] + args.factor_noise * torch.randn_like(factors[2]),
                factors[3],
            )
        loss, details = objective(factors, training=True)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                f"step {step:05d} loss={float(loss.detach()):.6f} "
                f"flow={details['flow']:.6f} endpoint={details['endpoint']:.6f} "
                f"match={details['match']:.6f} "
                f"recon={details['reconstruction']:.6f} "
                f"obs={details['observation']:.6f}",
                flush=True,
            )
        if step % max(1, args.validate_every) == 0 or step == args.steps:
            model.eval()
            values = []
            with torch.no_grad():
                for _ in range(max(1, args.validation_batches)):
                    factors = draw_batch(validation, args.batch_size)
                    value, _ = objective(factors, training=False)
                    values.append(float(value))
            val_loss = float(np.mean(values))
            validation_history.append({"step": step, "loss": val_loss})
            print(f"validation step {step:05d} loss={val_loss:.6f}", flush=True)
            if val_loss < best_validation:
                best_validation = val_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(args.out_checkpoint) or ".", exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "stats": {
                "mean": core_mean,
                "std": core_std,
                "shape": tuple(train[3].shape[1:]),
            },
            "g3_pca": None
            if g3_basis is None
            else {
                "mean": g3_mean,
                "basis": g3_basis,
                "code_mean": g3_code_mean,
                "code_std": g3_code_std,
                "shape": tuple(train[2].shape[1:]),
                "latent_dim": int(g3_basis.shape[0]),
            },
            "losses": losses,
            "validation": validation_history,
            "best_validation": best_validation,
        },
        args.out_checkpoint,
    )
    print(f"saved: {args.out_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
