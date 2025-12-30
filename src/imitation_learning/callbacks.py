from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np
import torch
import pytorch_lightning as pl
from godot_rl.wrappers.sbg_single_obs_wrapper import SBGSingleObsEnv

from tqdm import tqdm

class EvalCallback(pl.Callback):
    def __init__(
        self,
        env_path: Optional[str],
        obs_key: str,
        n_episodes: int,
        seed: int,
        speedup: int,
        viz: bool,
        max_steps: Optional[int],
    ):
        super().__init__()
        self.env_path = env_path
        self.obs_key = obs_key
        self.n_episodes = n_episodes
        self.seed = seed
        self.speedup = speedup
        self.viz = viz
        self.max_steps = max_steps
        self.env = None

    def on_fit_start(self, trainer, pl_module):
        self.env = SBGSingleObsEnv(
            env_path=self.env_path,
            show_window=self.viz,
            seed=self.seed,
            n_parallel=1,
            speedup=self.speedup,
            obs_key=self.obs_key,
        )

    def on_fit_end(self, trainer, pl_module):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None

    def _extract_obs(self, raw_obs):
        if isinstance(raw_obs, dict):
            if self.obs_key in raw_obs:
                return raw_obs[self.obs_key]
            if "obs" in raw_obs:
                return raw_obs["obs"]
        return raw_obs

    def _reset_env(self):
        out = self.env.reset()
        if isinstance(out, tuple):
            return out[0]
        return out

    def _step_env(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = terminated or truncated
            return obs, reward, done, info
        obs, reward, done, info = out
        return obs, reward, done, info

    def _build_window(
        self,
        obs_history: Deque[np.ndarray],
        seq_len: int,
        obs_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        obs_arr = np.zeros((16, seq_len, obs_dim), dtype=np.float32)
        mask = np.zeros((16, seq_len), dtype=np.bool_)
        for i, obs in enumerate(obs_history):
            obs_arr[:, i] = obs
            mask[:, i] = True
                
        return (
            torch.from_numpy(obs_arr),
            torch.from_numpy(mask),
            len(obs_history),
        )

    def on_train_epoch_end(self, trainer, pl_module):
        if self.env is None:
            return

        if trainer.current_epoch % 10 != 0:
            return

        seq_len = int(pl_module.cfg.train.seq_len)
        obs_dim = int(pl_module.net.obs_dim)
        action_mode = pl_module.net.action_mode

        device = pl_module.device
        pl_module.eval()
        episode_rewards = []
        with torch.no_grad():

            pbar = tqdm(total=self.n_episodes)
            while len(episode_rewards) < self.n_episodes:
                raw_obs = self._reset_env()
                obs = self._extract_obs(raw_obs)
                obs = np.asarray(obs, dtype=np.float32)
                if obs.shape[-1] != obs_dim:
                    raise ValueError(f"Expected obs shape ({obs_dim},), got {obs.shape}.")

                n_envs = int(obs.shape[0]) if obs.ndim > 1 else 1
                obs_history: Deque[np.ndarray] = deque(maxlen=seq_len)
                obs_history.append(obs)
                done = np.array(False).repeat(n_envs)
                total_reward = np.array(0.0).repeat(n_envs)
                steps = 0

                while not done.all():
                    obs_t, mask_t, valid_len = self._build_window(
                        obs_history,
                        seq_len,
                        obs_dim,
                    )
                    obs_t = obs_t.to(device)
                    mask_t = mask_t.to(device)

                    pred = pl_module.net(obs_t, mask_t)
                    action = pred[:, valid_len - 1].cpu().numpy()

                    raw_obs, reward, done, _ = self._step_env(action)
                    obs = self._extract_obs(raw_obs)
                    obs = np.asarray(obs, dtype=np.float32)
                    obs_history.append(obs)

                    total_reward += reward

                    if done.any():
                        done_indices = np.where(done)[0]
                        for idx in done_indices:
                            episode_rewards.append(total_reward[idx])
                            total_reward[idx] = 0.0
                            pbar.update(1)
                        if len(episode_rewards) >= self.n_episodes:
                            break

                    steps += 1
                    if self.max_steps is not None and steps >= self.max_steps:
                        break
        pl_module.train()

        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else float("nan")
        pl_module.log("eval/mean_reward", mean_reward, prog_bar=True, on_epoch=True)
