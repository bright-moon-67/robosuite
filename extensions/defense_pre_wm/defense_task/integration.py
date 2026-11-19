"""Minimal deployment-facing RSSM posterior filter. No actor is included."""

import torch
from .worldmodel_stage import device_stats, DefenseRSSM, OBS_DIM, ACTION_DIM, load_checkpoint


class LatentFilter:
    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        model, stats, _ = load_checkpoint(path, device)
        model.requires_grad_(False)
        return cls(model, stats, device)

    def __init__(self, model: DefenseRSSM, stats: dict, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.stats = device_stats(stats, self.device)
        self.h, self.z = None, None

    def _obs_tensor(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if x.numel() != OBS_DIM or not torch.isfinite(x).all():
            raise ValueError(f"expected {OBS_DIM}-D observation")
        return x.reshape(1, OBS_DIM)

    @torch.no_grad()
    def reset(self, first_obs):
        raw = self._obs_tensor(first_obs)
        norm = (raw - self.stats["mean"]) / self.stats["std"]
        self.h, self.z = self.model.initialize(norm)
        return self.features()

    @torch.no_grad()
    def update(self, previous_action, current_obs):
        if self.h is None:
            raise RuntimeError("call reset(first_obs) before update")
        a = torch.as_tensor(previous_action, dtype=torch.float32, device=self.device)
        if a.numel() != ACTION_DIM or not torch.isfinite(a).all() or (a.abs() > 1.000001).any():
            raise ValueError(f"expected {ACTION_DIM}-D previous action")
        raw = self._obs_tensor(current_obs)
        norm = (raw - self.stats["mean"]) / self.stats["std"]
        self.h, self.z, _, _ = self.model.transition(self.h, self.z, a.reshape(1, ACTION_DIM), norm)
        return self.features()

    def features(self):
        """[1, deter+latent] -> Actor latent input; detach for DiffDreamer."""
        if self.h is None:
            raise RuntimeError("filter was not initialized")
        return torch.cat([self.h, self.z], dim=-1).detach()

    @torch.no_grad()
    def imagine(self, future_actions):
        """Return world positions/status/reward/continuation without changing the filter.

        Future actions have shape [horizon, 7]; output shapes include batch size 1.
        For gradients with respect to actions, call model.imagine directly instead.
        """
        if self.h is None:
            raise RuntimeError("call reset(first_obs) before imagination")
        actions = torch.as_tensor(future_actions, dtype=torch.float32, device=self.device)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or not torch.isfinite(actions).all():
            raise ValueError("expected finite future actions [horizon, 7]")
        if (actions.abs() > 1.000001).any():
            raise ValueError("future actions exceed normalized bounds")
        return self.model.imagine(self.h, self.z, actions[None], self.stats)
