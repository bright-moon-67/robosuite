"""PyTorch RSSM reference for 60-D robosuite multi-block defense replay.

This is a deliberately SMALL Gaussian RSSM starter (not exact DreamerV3).
No dependency on robosuite at training time. No future observations during imagination.
Observation layout specified by the user:
  0:14 joint q/dq, 14:20 eef xyz/velocity, 20:50 (5,6) block rel+flags,
  50:55 categorical status, 55 clock, 56:60 eef quaternion.
"""

import argparse
import hashlib
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .dataset import load_episode as load_validated_episode, observation_schema
from .worldmodel_eval import openloop_report, world_positions

OBS_DIM, ACTION_DIM, NUM_BLOCKS = 60, 7, 5
CONTINUOUS = list(range(20)) + [20 + 6 * j + k for j in range(5) for k in range(3)]


def symlog(x):
    return x.sign() * torch.log1p(torch.abs(x))


def load_episode(path):
    """Validate the local replay contract and retain only learning channels."""
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    expected = observation_schema(NUM_BLOCKS)
    schema = metadata["schema"]
    if any(schema.get(key) != expected[key] for key in ("size", "version", "fields", "eef_quat")):
        raise ValueError(f"{path}: incompatible observation schema")
    if (
        metadata["action_dim"] != ACTION_DIM
        or metadata["status_codes"] != {"attacking": 0, "cleared": 1, "breached": 2}
        or metadata["control_freq"] != 20
    ):
        raise ValueError(f"{path}: incompatible actions, status labels or control frequency")
    ep = load_validated_episode(path)
    if ep["obs"].shape[1] != OBS_DIM:
        raise ValueError(f"{path}: expected {OBS_DIM}-D observation")
    states = np.concatenate([ep["obs"][:1], ep["next_obs"]]).astype("float32")
    if not np.allclose(np.linalg.norm(states[:, 56:60], axis=-1), 1, atol=1e-3):
        raise ValueError(f"{path}: expected unit xyzw quaternion")
    return {"states": states, **{key: ep[key].astype("float32") for key in ("action", "r_disc", "continuation")}}


def split_episodes(paths, seed=42):
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate episode paths")
    if len(paths) < 3:
        raise ValueError("Need at least 3 episodes for train/val/test")
    perm = np.random.default_rng(seed).permutation(len(paths))
    nv = max(1, round(0.15 * len(paths)))
    nt = max(1, round(0.15 * len(paths)))
    ntrain = len(paths) - nv - nt
    if ntrain < 1:
        raise ValueError("Not enough training episodes")
    return (
        [paths[i] for i in perm[:ntrain]],
        [paths[i] for i in perm[ntrain : ntrain + nv]],
        [paths[i] for i in perm[ntrain + nv :]],
    )


def fit_stats(episodes):
    x = np.concatenate([ep["states"] for ep in episodes], axis=0)
    mean = np.zeros(OBS_DIM, dtype="float32")
    std = np.ones(OBS_DIM, dtype="float32")
    mean[CONTINUOUS] = x[:, CONTINUOUS].mean(axis=0)
    std[CONTINUOUS] = np.maximum(x[:, CONTINUOUS].std(axis=0), 0.01)
    pos = world_positions(x).reshape(-1, 3)
    pmean = pos.mean(axis=0).astype("float32")
    pstd = np.maximum(pos.std(axis=0), 0.02).astype("float32")
    return {"mean": mean, "std": std, "pmean": pmean, "pstd": pstd}


def sample_batch(episodes, batch_size, length, rng, burn=0):
    if batch_size < 1 or length < 1 or burn < 0:
        raise ValueError("invalid batch, length or burn-in")
    length += burn
    eligible = [e for e in episodes if len(e["action"]) >= length]
    if len(eligible) != len(episodes):
        raise ValueError("Sequence plus burn-in exceeds a short episode; reduce length/burn to retain failure episodes")
    if not eligible:
        raise ValueError(f"No episode has >= {length} transitions")
    out = {"states": [], "actions": [], "reward": [], "cont": []}
    # Draw uniformly over ALL valid windows; short episodes are not given
    # disproportionate probability merely because they are short.
    counts = np.array([len(ep["action"]) - length + 1 for ep in eligible], dtype=float)
    probs = counts / counts.sum()
    for _ in range(batch_size):
        ep = eligible[int(rng.choice(len(eligible), p=probs))]
        start = int(rng.integers(0, len(ep["action"]) - length + 1))
        end = start + length
        out["states"].append(ep["states"][start : end + 1])
        out["actions"].append(ep["action"][start:end])
        out["reward"].append(ep["r_disc"][start:end])
        out["cont"].append(ep["continuation"][start:end])
    return {k: torch.from_numpy(np.stack(v)).float() for k, v in out.items()}


def diagonal_kl(mq, lq, mp, lp):
    """KL[q || p] with diag Normal, last dim is summed."""
    vq, vp = torch.exp(2 * lq), torch.exp(2 * lp)
    return (0.5 * (2 * (lp - lq) + (vq + (mq - mp) ** 2) / vp - 1)).sum(-1)


class DefenseRSSM(nn.Module):
    """h deterministic GRU memory; z Gaussian stochastic latent.

    Posterior sees o_(t+1); prior never does. Only external state is decoded.
    """

    def __init__(self, deter=128, latent=24, embed=128):
        super().__init__()
        self.deter, self.latent = deter, latent
        self.encoder = nn.Sequential(nn.Linear(OBS_DIM, embed), nn.SiLU(), nn.Linear(embed, embed), nn.SiLU())
        self.rnn = nn.GRUCell(latent + ACTION_DIM, deter)
        self.prior = nn.Linear(deter, 2 * latent)
        self.posterior = nn.Sequential(nn.Linear(deter + embed, embed), nn.SiLU(), nn.Linear(embed, 2 * latent))
        self.feature_dim = deter + latent
        self.position_head = nn.Sequential(nn.Linear(self.feature_dim, 128), nn.SiLU(), nn.Linear(128, 3 * NUM_BLOCKS))
        self.status_head = nn.Linear(self.feature_dim, 3 * NUM_BLOCKS)
        self.reward_head = nn.Linear(self.feature_dim, 1)
        self.cont_head = nn.Linear(self.feature_dim, 1)

    @staticmethod
    def dist(params):
        mean, raw_logstd = params.chunk(2, -1)
        logstd = torch.clamp(raw_logstd, -5.0, 1.0)
        return mean, logstd

    def initialize(self, obs0):
        b = obs0.shape[0]
        h = obs0.new_zeros(b, self.deter)
        # Initial posterior based on the first real observation.
        zmean, zlog = self.dist(self.posterior(torch.cat([h, self.encoder(obs0)], dim=-1)))
        z = zmean + torch.randn_like(zmean) * torch.exp(zlog) if self.training else zmean
        return h, z

    def transition(self, h, z, action, next_obs=None):
        h = self.rnn(torch.cat([z, action], dim=-1), h)
        pm, pl = self.dist(self.prior(h))
        if next_obs is None:
            return h, pm, (pm, pl), None
        qm, ql = self.dist(self.posterior(torch.cat([h, self.encoder(next_obs)], dim=-1)))
        z = qm + torch.randn_like(qm) * torch.exp(ql) if self.training else qm
        return h, z, (pm, pl), (qm, ql)

    def heads(self, h, z):
        f = torch.cat([h, z], -1)
        return (
            self.position_head(f).reshape(-1, NUM_BLOCKS, 3),
            self.status_head(f).reshape(-1, NUM_BLOCKS, 3),
            self.reward_head(f).squeeze(-1),
            self.cont_head(f).squeeze(-1),
        )

    def batch_loss(self, batch, stats, burn=0):
        states, actions = batch["states"], batch["actions"]
        if not 0 <= burn < actions.shape[1]:
            raise ValueError("burn-in must leave at least one loss transition")
        mean, std = stats["mean"], stats["std"]
        pmean, pstd = stats["pmean"], stats["pstd"]
        h, z = self.initialize((states[:, 0] - mean) / std)
        accum = torch.zeros(5, device=states.device)
        for t in range(actions.shape[1]):
            true_o = states[:, t + 1]
            x = (true_o - mean) / std
            if t < burn:
                with torch.no_grad():
                    h, z, _, _ = self.transition(h.detach(), z.detach(), actions[:, t], x)
                continue
            h, z, (pm, pl), (qm, ql) = self.transition(h, z, actions[:, t], x)
            pp, ss, rr, cc = self.heads(h, z)
            # The extra prior reconstruction keeps imagination useful.
            prior_p, _, _, _ = self.heads(h, pm)
            target = (world_positions(true_o) - pmean) / pstd
            status = torch.round(true_o[:, 50:55]).long().clamp(0, 2)
            active = (status == 0).float() + 0.25 * (status != 0).float()
            posloss = (F.smooth_l1_loss(pp, target, reduction="none").mean(-1) * active).mean()
            priorloss = (F.smooth_l1_loss(prior_p, target, reduction="none").mean(-1) * active).mean()
            sloss = F.cross_entropy(
                ss.reshape(-1, 3), status.reshape(-1), weight=torch.tensor([0.5, 1.0, 2.0], device=states.device)
            )
            rloss = F.smooth_l1_loss(rr, symlog(batch["reward"][:, t]))
            closs = F.binary_cross_entropy_with_logits(cc, batch["cont"][:, t], reduction="none")
            closs = (closs * torch.where(batch["cont"][:, t] < 0.5, 15.0, 1.0)).mean()
            # Separate stopped-gradient KL objectives, Dreamer-style.
            dyn = diagonal_kl(qm.detach(), ql.detach(), pm, pl)
            rep = diagonal_kl(qm, ql, pm.detach(), pl.detach())
            kl = 0.5 * torch.clamp(dyn, min=1.0).mean() + 0.1 * torch.clamp(rep, min=1.0).mean()
            accum = accum + torch.stack([posloss + priorloss, sloss, rloss, closs, kl])
        v = accum / (actions.shape[1] - burn)
        loss = v.sum()
        return loss, {
            "pos": v[0].item(),
            "status": v[1].item(),
            "reward": v[2].item(),
            "continue": v[3].item(),
            "kl": v[4].item(),
        }

    @torch.no_grad()
    def forecast(self, states_context, actions_context, future_actions, stats):
        """No future observations; context states shape [B, C+1, 60].
        actions_context shape [B,C,7]; future_actions [B,H,7].
        Returns future block WORLD positions [B,H,5,3] in meters.
        """
        self.eval()
        if states_context.shape != (actions_context.shape[0], actions_context.shape[1] + 1, OBS_DIM):
            raise ValueError("context states must contain one more frame than context actions")
        h, z = self.initialize((states_context[:, 0] - stats["mean"]) / stats["std"])
        for t in range(actions_context.shape[1]):
            x_next = (states_context[:, t + 1] - stats["mean"]) / stats["std"]
            h, z, _, _ = self.transition(h, z, actions_context[:, t], x_next)
        return self.imagine(h, z, future_actions, stats)

    def imagine(self, h, z, future_actions, stats):
        """Roll the prior mean from an existing posterior without modifying it."""
        if (
            future_actions.ndim != 3
            or future_actions.shape[0] != h.shape[0]
            or future_actions.shape[-1] != ACTION_DIM
            or future_actions.shape[1] < 1
        ):
            raise ValueError("expected future actions [batch, horizon >= 1, 7]")
        preds, status_probs, reward_est, continue_probs = [], [], [], []
        for t in range(future_actions.shape[1]):
            h, z, _, _ = self.transition(h, z, future_actions[:, t])
            pred, status, reward, cont = self.heads(h, z)
            preds.append(pred * stats["pstd"] + stats["pmean"])
            status_probs.append(status.softmax(-1))
            reward_est.append(reward.sign() * torch.expm1(torch.abs(reward)))
            continue_probs.append(cont.sigmoid())
        return (
            torch.stack(preds, 1),
            torch.stack(status_probs, 1),
            torch.stack(reward_est, 1),
            torch.stack(continue_probs, 1),
        )

    @torch.no_grad()
    def forecast_positions(self, states_context, actions_context, future_actions, stats):
        return self.forecast(states_context, actions_context, future_actions, stats)[0]


def device_stats(stats, device):
    return {k: torch.as_tensor(v, device=device) for k, v in stats.items()}


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = DefenseRSSM(**checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), device_stats(checkpoint["stats"], device), checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "train", "eval"))
    parser.add_argument("--data", required=True, type=Path, help="directory of paired episode NPZ/JSON")
    parser.add_argument("--out", type=Path, default=Path("artifacts/defense_rssm_v0"))
    parser.add_argument("--ood-data", type=Path, help="eval an independent dataset with new episode seeds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--burn", type=int, default=8, help="posterior warmup transitions before training loss")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--windows-per-episode", type=int, default=2)
    parser.add_argument("--stride", type=int, help="eval all eligible starts at this stride instead of random windows")
    args = parser.parse_args()
    checkpoint = args.out / "world_model.pt"
    if min(args.steps, args.batch, args.length, args.windows_per_episode) < 1 or args.lr <= 0:
        parser.error("steps, batch, length, windows and learning rate must be positive")
    if args.burn < 0:
        parser.error("burn-in must be nonnegative")
    if args.ood_data and args.mode != "eval":
        parser.error("--ood-data is for evaluation")
    paths = sorted(args.data.glob("episode_*.npz"))
    torch.manual_seed(args.seed)
    if args.mode == "eval":
        model, stats, cp = load_checkpoint(checkpoint, args.device)
        if args.ood_data:
            paths = sorted(args.ood_data.glob("episode_*.npz"))
            if not paths:
                parser.error("no OOD episodes found")
            if set(episode_seeds(paths)).intersection(cp["seeds"]):
                parser.error("OOD episode seeds overlap the saved source dataset")
            if set(file_record(p)["npz"] for p in paths).intersection(v["npz"] for v in cp["files"].values()):
                parser.error("OOD data overlaps the saved source dataset")
            report_path = args.out / f"openloop_ood_{args.ood_data.resolve().name}.json"
        else:
            paths = [args.data / name for name in cp["split"]["test"]]
            if episode_seeds(paths) != cp["test_seeds"]:
                parser.error("held-out episode seeds differ from checkpoint")
            for path in paths:
                if file_record(path) != cp["files"][path.name]:
                    parser.error(f"held-out file changed since training: {path.name}")
            report_path = args.out / "openloop_report.json"
        report = openloop_report(
            model,
            [load_episode(p) for p in paths],
            stats,
            windows_per_ep=args.windows_per_episode,
            stride=args.stride,
            device=args.device,
        )
        report["dataset"] = {
            "directory": str(paths[0].parent.resolve()),
            "episodes": [p.name for p in paths],
            "profiles": {p.name: episode_profile(p) for p in paths},
            "context_steps": 20,
            "stride": args.stride,
            "windows_per_episode": args.windows_per_episode,
            "sampling_seed": 777,
            "control_hz": 20,
        }
        report["checkpoint"] = {
            "path": str(checkpoint.resolve()),
            "sha256": file_hash(checkpoint),
            "step": cp["step"],
            "validation_loss": cp["validation_loss"],
        }
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
        print("saved:", report_path)
        return

    groups = dict(zip(("train", "val", "test"), split_episodes(paths, args.seed)))
    seeds = episode_seeds(paths)
    if len(set(seeds)) != len(paths):
        raise ValueError("duplicate episode seeds in source dataset; split independent episodes")
    print("episodes:", {name: len(group) for name, group in groups.items()})
    if args.mode == "inspect":
        for name, group in groups.items():
            sizes = [len(load_episode(p)["action"]) for p in group]
            print(f"{name}: {sum(sizes)} transitions, lengths {min(sizes)}..{max(sizes)}")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise FileExistsError(f"{args.out} is not empty; choose a fresh --out")
    train = [load_episode(p) for p in groups["train"]]
    val = [load_episode(p) for p in groups["val"]]
    stats = device_stats(fit_stats(train), args.device)
    # Fixed validation windows make successive checkpoints comparable.
    validation = {
        k: v.to(args.device)
        for k, v in sample_batch(val, args.batch, args.length, np.random.default_rng(12345), args.burn).items()
    }
    architecture = dict(deter=128, latent=24, embed=128)
    model = DefenseRSSM(**architecture).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed + 1)
    files = {p.name: file_record(p) for p in paths}
    if len({v["npz"] for v in files.values()}) != len(paths):
        raise ValueError("duplicate episode contents in source dataset")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    np.savez(args.out / "normalization.npz", **{k: v.cpu().numpy() for k, v in stats.items()})
    best = np.inf
    for step in range(1, args.steps + 1):
        model.train()
        batch = {k: v.to(args.device) for k, v in sample_batch(train, args.batch, args.length, rng, args.burn).items()}
        opt.zero_grad(set_to_none=True)
        loss, _ = model.batch_loss(batch, stats, args.burn)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 100.0, error_if_nonfinite=True)
        opt.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vloss, parts = model.batch_loss(validation, stats, args.burn)
            if not torch.isfinite(vloss):
                raise RuntimeError(f"Nonfinite validation loss at step {step}")
            print(f"step={step} train={loss.item():.4f} val={vloss.item():.4f} parts={parts}", flush=True)
            with (args.out / "training.jsonl").open("a") as log:
                log.write(json.dumps({"step": step, "train": loss.item(), "validation": vloss.item(), **parts}) + "\n")
            if vloss.item() < best:
                best = vloss.item()
                temp = checkpoint.with_suffix(".tmp")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "stats": stats,
                        "architecture": architecture,
                        "seed": args.seed,
                        "step": step,
                        "validation_loss": best,
                        "seeds": seeds,
                        "files": files,
                        "config": config,
                        "test_seeds": episode_seeds(groups["test"]),
                        "data": str(args.data.resolve()),
                        "split": {k: [p.name for p in group] for k, group in groups.items()},
                    },
                    temp,
                )
                temp.replace(checkpoint)
    print("best saved:", checkpoint)


def episode_seeds(paths):
    return [json.loads(path.with_suffix(".json").read_text())["seed"] for path in paths]


def episode_profile(path):
    metadata = json.loads(path.with_suffix(".json").read_text())
    config = metadata.get("config", {})
    return {key: config.get(key) for key in ("motion", "min_speed", "max_speed", "stagger", "epsilon")}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_record(path):
    return {"npz": file_hash(path), "json": file_hash(path.with_suffix(".json"))}


if __name__ == "__main__":
    main()
