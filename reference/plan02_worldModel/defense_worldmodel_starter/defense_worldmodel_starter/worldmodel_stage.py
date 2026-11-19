"""PyTorch RSSM reference for 60-D robosuite multi-block defense replay.

This is a deliberately SMALL Gaussian RSSM starter (not exact DreamerV3).
No dependency on robosuite at training time. No future observations during imagination.
Observation layout specified by the user:
  0:14 joint q/dq, 14:20 eef xyz/velocity, 20:50 (5,6) block rel+flags,
  50:55 categorical status, 55 clock, 56:60 eef quaternion.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OBS_DIM, ACTION_DIM, NUM_BLOCKS = 60, 7, 5
CONTINUOUS = list(range(20)) + [20+6*j+k for j in range(5) for k in range(3)]


def world_positions(obs):
    """Block world xyz = observed relative-to-EEF xyz + EEF world xyz."""
    blocks = obs[..., 20:50].reshape(*obs.shape[:-1], NUM_BLOCKS, 6)
    return blocks[..., :3] + obs[..., 14:17][..., None, :]


def symlog(x):
    return x.sign() * torch.log1p(torch.abs(x))


def load_episode(path):
    with np.load(path, allow_pickle=False) as f:
        ep = {k: f[k].copy() for k in f.files}
    required = ("obs", "next_obs", "action", "r_disc", "continuation")
    for name in required:
        if name not in ep:
            raise ValueError(f"{path}: missing field {name}")
    n = len(ep["action"])
    if n < 1 or ep["obs"].shape != (n, OBS_DIM) \
            or ep["next_obs"].shape != (n, OBS_DIM) \
            or ep["action"].shape != (n, ACTION_DIM):
        raise ValueError(f"{path}: invalid transition shape")
    for k in required:
        if len(ep[k]) != n or not np.isfinite(ep[k]).all():
            raise ValueError(f"{path}: nonfinite or mismatched {k}")
    if n > 1 and not np.allclose(ep["next_obs"][:-1], ep["obs"][1:], atol=1e-5):
        raise ValueError(f"{path}: transitions are not contiguous")
    if "reward" in ep and "r_diff" in ep:
        if not np.allclose(ep["reward"], ep["r_disc"] + ep["r_diff"], atol=1e-4):
            raise ValueError(f"{path}: inconsistent reward decomposition")
    if "terminated" in ep and "truncated" in ep:
        terminated = np.asarray(ep["terminated"]).astype(bool)
        truncated = np.asarray(ep["truncated"]).astype(bool)
        if np.any(terminated & truncated):
            raise ValueError(f"{path}: terminated and truncated both true")
        expected = 1.0 - terminated.astype(float)
        if not np.allclose(ep["continuation"], expected):
            raise ValueError(f"{path}: continuation is not (1 - terminated)")
    if "done" in ep and np.any(np.asarray(ep["done"])[:-1]):
        raise ValueError(f"{path}: episode has a done flag before last transition")
    if "is_first" in ep and not bool(ep["is_first"][0]):
        raise ValueError(f"{path}: first transition lacks is_first")
    statuses = np.rint(ep["next_obs"][:, 50:55]).astype(int)
    if np.any((statuses < 0) | (statuses > 2)):
        raise ValueError(f"{path}: invalid block status")
    ep["states"] = np.concatenate([ep["obs"][:1], ep["next_obs"]], axis=0).astype("float32")
    ep["action"] = ep["action"].astype("float32")
    ep["r_disc"] = ep["r_disc"].astype("float32")
    ep["continuation"] = ep["continuation"].astype("float32")
    return ep


def split_episodes(paths, seed=42):
    if len(paths) < 3:
        raise ValueError("Need at least 3 episodes for train/val/test")
    perm = np.random.default_rng(seed).permutation(len(paths))
    nv = max(1, round(.15 * len(paths)))
    nt = max(1, round(.15 * len(paths)))
    ntrain = len(paths) - nv - nt
    if ntrain < 1:
        raise ValueError("Not enough training episodes")
    return ([paths[i] for i in perm[:ntrain]],
            [paths[i] for i in perm[ntrain:ntrain+nv]],
            [paths[i] for i in perm[ntrain+nv:]])


def fit_stats(episodes):
    x = np.concatenate([ep["states"] for ep in episodes], axis=0)
    mean = np.zeros(OBS_DIM, dtype="float32")
    std = np.ones(OBS_DIM, dtype="float32")
    mean[CONTINUOUS] = x[:, CONTINUOUS].mean(axis=0)
    std[CONTINUOUS] = np.maximum(x[:, CONTINUOUS].std(axis=0), .01)
    pos = world_positions(x).reshape(-1, 3)
    pmean = pos.mean(axis=0).astype("float32")
    pstd = np.maximum(pos.std(axis=0), .02).astype("float32")
    return {"mean": mean, "std": std, "pmean": pmean, "pstd": pstd}


def sample_batch(episodes, batch_size, length, rng):
    eligible = [e for e in episodes if len(e["action"]) >= length]
    if not eligible:
        raise ValueError(f"No episode has >= {length} transitions")
    out = {"states": [], "actions": [], "reward": [], "cont": []}
    # Draw uniformly over ALL valid windows; short episodes are not given
    # disproportionate probability merely because they are short.
    counts = np.array([len(ep["action"])-length+1 for ep in eligible], dtype=float)
    probs = counts/counts.sum()
    for _ in range(batch_size):
        ep = eligible[int(rng.choice(len(eligible), p=probs))]
        start = int(rng.integers(0, len(ep["action"])-length+1))
        end = start + length
        out["states"].append(ep["states"][start:end+1])
        out["actions"].append(ep["action"][start:end])
        out["reward"].append(ep["r_disc"][start:end])
        out["cont"].append(ep["continuation"][start:end])
    return {k: torch.from_numpy(np.stack(v)).float() for k, v in out.items()}


def diagonal_kl(mq, lq, mp, lp):
    """KL[q || p] with diag Normal, last dim is summed."""
    vq, vp = torch.exp(2*lq), torch.exp(2*lp)
    return (.5 * (2*(lp-lq) + (vq + (mq-mp)**2)/vp - 1)).sum(-1)


class DefenseRSSM(nn.Module):
    """h deterministic GRU memory; z Gaussian stochastic latent.

    Posterior sees o_(t+1); prior never does. Only external state is decoded.
    """

    def __init__(self, deter=128, latent=24, embed=128):
        super().__init__()
        self.deter, self.latent = deter, latent
        self.encoder = nn.Sequential(nn.Linear(OBS_DIM, embed), nn.SiLU(),
                                     nn.Linear(embed, embed), nn.SiLU())
        self.rnn = nn.GRUCell(latent + ACTION_DIM, deter)
        self.prior = nn.Linear(deter, 2*latent)
        self.posterior = nn.Sequential(nn.Linear(deter+embed, embed), nn.SiLU(),
                                       nn.Linear(embed, 2*latent))
        self.feature_dim = deter+latent
        self.position_head = nn.Sequential(nn.Linear(self.feature_dim, 128), nn.SiLU(),
                                           nn.Linear(128, 3*NUM_BLOCKS))
        self.status_head = nn.Linear(self.feature_dim, 3*NUM_BLOCKS)
        self.reward_head = nn.Linear(self.feature_dim, 1)
        self.cont_head = nn.Linear(self.feature_dim, 1)

    @staticmethod
    def dist(params):
        mean, raw_logstd = params.chunk(2, -1)
        logstd = torch.clamp(raw_logstd, -5., 1.)
        return mean, logstd

    def initialize(self, obs0):
        b = obs0.shape[0]
        h = obs0.new_zeros(b, self.deter)
        # Initial posterior based on the first real observation.
        zmean, zlog = self.dist(self.posterior(
            torch.cat([h, self.encoder(obs0)], dim=-1)))
        z = zmean + torch.randn_like(zmean)*torch.exp(zlog) if self.training else zmean
        return h, z

    def transition(self, h, z, action, next_obs=None):
        h = self.rnn(torch.cat([z, action], dim=-1), h)
        pm, pl = self.dist(self.prior(h))
        if next_obs is None:
            return h, pm, (pm, pl), None
        qm, ql = self.dist(self.posterior(
            torch.cat([h, self.encoder(next_obs)], dim=-1)))
        z = qm + torch.randn_like(qm)*torch.exp(ql) if self.training else qm
        return h, z, (pm, pl), (qm, ql)

    def heads(self, h, z):
        f = torch.cat([h, z], -1)
        return (self.position_head(f).reshape(-1, NUM_BLOCKS, 3),
                self.status_head(f).reshape(-1, NUM_BLOCKS, 3),
                self.reward_head(f).squeeze(-1),
                self.cont_head(f).squeeze(-1))

    def batch_loss(self, batch, stats):
        states, actions = batch["states"], batch["actions"]
        mean, std = stats["mean"], stats["std"]
        pmean, pstd = stats["pmean"], stats["pstd"]
        h, z = self.initialize((states[:, 0]-mean)/std)
        accum = torch.zeros(5, device=states.device)
        for t in range(actions.shape[1]):
            true_o = states[:, t+1]
            x = (true_o - mean)/std
            h, z, (pm, pl), (qm, ql) = self.transition(h, z, actions[:, t], x)
            pp, ss, rr, cc = self.heads(h, z)
            # The extra prior reconstruction keeps imagination useful.
            prior_p, _, _, _ = self.heads(h, pm)
            target = (world_positions(true_o)-pmean)/pstd
            status = torch.round(true_o[:, 50:55]).long().clamp(0, 2)
            active = (status == 0).float() + .25*(status != 0).float()
            posloss = (F.smooth_l1_loss(pp, target, reduction="none").mean(-1)*
                       active).mean()
            priorloss = (F.smooth_l1_loss(prior_p,target,reduction="none")
                         .mean(-1)*active).mean()
            sloss = F.cross_entropy(ss.reshape(-1, 3), status.reshape(-1),
                                    weight=torch.tensor([.5, 1., 2.],
                                                        device=states.device))
            rloss = F.smooth_l1_loss(rr, symlog(batch["reward"][:, t]))
            closs = F.binary_cross_entropy_with_logits(
                cc, batch["cont"][:, t], reduction="none")
            closs = (closs * torch.where(batch["cont"][:, t] < .5, 15., 1.)).mean()
            # Separate stopped-gradient KL objectives, Dreamer-style.
            dyn = diagonal_kl(qm.detach(), ql.detach(), pm, pl)
            rep = diagonal_kl(qm, ql, pm.detach(), pl.detach())
            kl = .5*torch.clamp(dyn, min=1.).mean() \
               + .1*torch.clamp(rep, min=1.).mean()
            accum = accum + torch.stack([posloss+priorloss, sloss, rloss,
                                         closs, kl])
        v = accum / actions.shape[1]
        loss = v.sum()
        return loss, {"pos": v[0].item(), "status":v[1].item(),
                      "reward":v[2].item(), "continue":v[3].item(),
                      "kl":v[4].item()}

    @torch.no_grad()
    def forecast(self, states_context, actions_context, future_actions,
                 stats):
        """No future observations; context states shape [B, C+1, 60].
        actions_context shape [B,C,7]; future_actions [B,H,7].
        Returns future block WORLD positions [B,H,5,3] in meters.
        """
        self.eval()
        h, z = self.initialize(
            (states_context[:, 0]-stats["mean"])/stats["std"])
        for t in range(actions_context.shape[1]):
            x_next = (states_context[:, t+1]-stats["mean"])/stats["std"]
            h, z, _, _ = self.transition(h,z,actions_context[:, t], x_next)
        preds, status_probs, reward_est, continue_probs = [], [], [], []
        for t in range(future_actions.shape[1]):
            h, z, _, _ = self.transition(h,z,future_actions[:,t])
            pred, status, reward, cont = self.heads(h,z)
            preds.append(pred*stats["pstd"] + stats["pmean"])
            status_probs.append(status.softmax(-1))
            reward_est.append(reward.sign()*torch.expm1(torch.abs(reward)))
            continue_probs.append(cont.sigmoid())
        return (torch.stack(preds, 1),
                torch.stack(status_probs, 1),
                torch.stack(reward_est, 1),
                torch.stack(continue_probs, 1))

    @torch.no_grad()
    def forecast_positions(self, states_context, actions_context,
                           future_actions, stats):
        return self.forecast(states_context, actions_context,
                             future_actions, stats)[0]


def device_stats(stats, device):
    return {k: torch.as_tensor(v, device=device) for k,v in stats.items()}


def openloop_report(model, episodes, stats, horizons=(10,20,40),
                    burn=20, windows_per_ep=2, seed=777, device="cpu"):
    """Evaluate against CV; ALL predictions use only observed context and known
    recorded future actions (no future observations or future true robot states).
    Primary metric: blocks active throughout the forecast, to isolate motion.
    """
    rng = np.random.default_rng(seed)
    summary = {}
    model.eval()
    with torch.no_grad():
        for horizon in horizons:
            errs, cv_errs, active_errs = [], [], []
            positives, true_positives, false_positives = 0, 0, 0
            for ep in episodes:
                T = len(ep["action"])
                if T < burn+horizon: continue
                for _ in range(windows_per_ep):
                    start = int(rng.integers(0, T-burn-horizon+1))
                    b = start+burn
                    states = ep["states"]
                    s_ctx = torch.from_numpy(states[start:b+1])[None].to(device)
                    a_ctx = torch.from_numpy(ep["action"][start:b])[None].to(device)
                    a_fut = torch.from_numpy(ep["action"][b:b+horizon])[None].to(device)
                    forecast_t, status_t, _, _ = model.forecast(
                        s_ctx,a_ctx,a_fut,stats)
                    forecast = forecast_t[0].cpu().numpy()
                    status_prob = status_t[0].cpu().numpy()
                    truth = world_positions(states[b+1:b+horizon+1])
                    mae = np.linalg.norm(forecast-truth,axis=-1)
                    # Constant-velocity predictor from the two most recent real states.
                    last, previous = world_positions(states[b]), world_positions(states[b-1])
                    velocity = last - previous  # displacement per control step
                    cvel = last[None] + (np.arange(1,horizon+1)[:,None,None])*velocity[None]
                    cverr = np.linalg.norm(cvel-truth,axis=-1)
                    errs.append(mae[-1].mean())
                    cv_errs.append(cverr[-1].mean())
                    start_active = (states[b,50:55] == 0)
                    remained_active = (states[b+1:b+horizon+1,50:55] == 0).all(axis=0)
                    grasped_future = states[b+1:b+horizon+1,20:50].reshape(
                        horizon,5,6)[:,:,4] > .5
                    never_grasped = ~grasped_future.any(axis=0)
                    keep = start_active & remained_active & never_grasped
                    if keep.any():
                        active_errs.append(mae[-1,keep].mean())
                    actual_status = states[b+1:b+horizon+1,50:55]
                    actual_breach = bool(np.any(
                        (actual_status == 2) & start_active[None]))
                    predicted_breach = bool(np.any(
                        (status_prob[:,:,2] > .5) & start_active[None]))
                    positives += int(actual_breach)
                    true_positives += int(actual_breach and predicted_breach)
                    false_positives += int(not actual_breach and predicted_breach)
            summary[str(horizon)] = {
                "wm_m":float(np.mean(errs)) if errs else None,
                "cv_m":float(np.mean(cv_errs)) if cv_errs else None,
                "always_active_wm_m":float(np.mean(active_errs)) if active_errs else None,
                "windows":len(errs),
                "breach_prevalence":positives/len(errs) if errs else None,
                "breach_recall":true_positives/positives if positives else None,
                "breach_precision":true_positives/(true_positives+false_positives)
                    if (true_positives+false_positives) else None,
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect","train","eval"))
    parser.add_argument("--data", required=True, help="directory of episode_*.npz")
    parser.add_argument("--out", default="runs/wm_sine5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    paths = sorted(Path(args.data).glob("*.npz"))
    if len(paths) < 3:
        raise RuntimeError("At least 3 episode npz files required")
    train_paths, val_paths, test_paths = split_episodes(paths,args.seed)
    train = [load_episode(p) for p in train_paths]
    val = [load_episode(p) for p in val_paths]
    test = [load_episode(p) for p in test_paths]
    print(f"train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"transition counts={[sum(len(e['action']) for e in subset) for subset in (train,val,test)]}")
    if args.mode == "inspect":
        print("first episode:",train_paths[0],len(train[0]["action"]),
              "states",train[0]["states"].shape)
        return
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    checkpoint = out/"world_model.pt"
    if args.mode == "train":
        stats_np=fit_stats(train)
        stats=device_stats(stats_np,device)
        model=DefenseRSSM().to(device)
        opt=torch.optim.Adam(model.parameters(),lr=args.lr)
        rng=np.random.default_rng(args.seed+1)
        best=np.inf
        for step in range(1,args.steps+1):
            model.train()
            batch={k:v.to(device) for k,v in
                   sample_batch(train,args.batch,args.length,rng).items()}
            opt.zero_grad(set_to_none=True)
            loss, parts=model.batch_loss(batch,stats)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),100.)
            opt.step()
            if step==1 or step%100==0 or step==args.steps:
                model.eval()
                vb={k:v.to(device) for k,v in
                    sample_batch(val,args.batch,args.length,
                                 np.random.default_rng(12345)).items()}
                with torch.no_grad():
                    vloss,vparts=model.batch_loss(vb,stats)
                print(f"step={step} train={loss.item():.4f} val={vloss.item():.4f}"
                      f" parts={vparts}")
                if vloss.item()<best:
                    best=vloss.item()
                    torch.save({"model":model.state_dict(),"stats":stats_np,
                                "seed":args.seed,"deter":model.deter,
                                "latent":model.latent,
                                "split":{
                                    "train":[p.name for p in train_paths],
                                    "val":[p.name for p in val_paths],
                                    "test":[p.name for p in test_paths],
                                }},checkpoint)
        print("best saved:",checkpoint)
    elif args.mode=="eval":
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        cp=torch.load(checkpoint,map_location=device,weights_only=False)
        # Replay split is LOCKED to the saved experiment, regardless of CLI seed.
        old_test = cp.get("split", {}).get("test", None)
        if old_test is None:
            raise ValueError("Checkpoint missing frozen test split")
        available = {p.name:p for p in paths}
        if any(name not in available for name in old_test):
            raise ValueError("Test episodes used in training are missing")
        test = [load_episode(available[name]) for name in old_test]
        model=DefenseRSSM(deter=cp["deter"],latent=cp["latent"]).to(device)
        model.load_state_dict(cp["model"])
        stats=device_stats(cp["stats"],device)
        report=openloop_report(model,test,stats,device=device)
        print(json.dumps(report,indent=2,ensure_ascii=False))
        (out/"openloop_report.json").write_text(json.dumps(report,indent=2))
    else:
        raise AssertionError("unreachable")


if __name__=="__main__":
    main()
