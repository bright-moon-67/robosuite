"""Minimal deployment-facing RSSM posterior filter. No actor is included."""
import torch
from worldmodel_stage import device_stats, DefenseRSSM, OBS_DIM, ACTION_DIM


class LatentFilter:
    def __init__(self, model: DefenseRSSM, stats: dict, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.stats = device_stats(stats, self.device)
        self.h, self.z = None, None

    def _obs_tensor(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if x.numel() != OBS_DIM:
            raise ValueError(f"expected {OBS_DIM}-D observation")
        return x.reshape(1, OBS_DIM)

    @torch.no_grad()
    def reset(self, first_obs):
        raw = self._obs_tensor(first_obs)
        norm = (raw-self.stats["mean"])/self.stats["std"]
        self.h, self.z = self.model.initialize(norm)
        return self.features()

    @torch.no_grad()
    def update(self, previous_action, current_obs):
        if self.h is None:
            raise RuntimeError("call reset(first_obs) before update")
        a = torch.as_tensor(previous_action, dtype=torch.float32,
                            device=self.device)
        if a.numel() != ACTION_DIM:
            raise ValueError(f"expected {ACTION_DIM}-D previous action")
        raw = self._obs_tensor(current_obs)
        norm = (raw-self.stats["mean"])/self.stats["std"]
        self.h, self.z, _, _ = self.model.transition(
            self.h, self.z, a.reshape(1,ACTION_DIM), norm)
        return self.features()

    def features(self):
        """[1, deter+latent] -> Actor latent input; detach for DiffDreamer."""
        if self.h is None:
            raise RuntimeError("filter was not initialized")
        return torch.cat([self.h,self.z], dim=-1).detach()
