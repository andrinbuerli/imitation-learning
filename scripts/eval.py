from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

from godot_rl.wrappers.sbg_single_obs_wrapper import SBGSingleObsEnv
from tqdm import tqdm
from train_torch import BCTransformerLit
try:
    import onnxruntime as ort
except ImportError:
    ort = None


def _extract_obs(raw_obs, obs_key: str):
    if isinstance(raw_obs, dict):
        if obs_key in raw_obs:
            return raw_obs[obs_key]
        if "obs" in raw_obs:
            return raw_obs["obs"]
    return raw_obs


def _reset_env(env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def _step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
        return obs, reward, done, info
    obs, reward, done, info = out
    return obs, reward, done, info


def _build_window(
    obs_history: Deque[np.ndarray],
    seq_len: int,
    obs_dim: int,
    n_envs: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    obs_arr = np.zeros((n_envs, seq_len, obs_dim), dtype=np.float32)
    mask = np.zeros((n_envs, seq_len), dtype=np.bool_)
    for i, obs in enumerate(obs_history):
        obs_arr[:, i] = obs
        mask[:, i] = True
    return (
        torch.from_numpy(obs_arr),
        torch.from_numpy(mask),
        len(obs_history),
    )

class OnnxPolicy:
    def __init__(self, onnx_path: str):
        if ort is None:
            raise RuntimeError("onnxruntime is required to run ONNX models.")
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        if not self.input_names:
            raise RuntimeError("ONNX model has no inputs.")

    def predict(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 1:
            obs = obs[None, :]
        inputs = {self.input_names[0]: obs.astype(np.float32), "state_outs": None}
        outputs = self.session.run(self.output_names, inputs)
        action = outputs[0]
        return np.asarray(action)[0]


@hydra.main(config_path="../configs", config_name="evaluate_torch", version_base="1.3")
def main(cfg: DictConfig) -> None:
    using_onnx = bool(cfg.model.onnx_path)
    if using_onnx == bool(cfg.model.checkpoint_path):
        raise ValueError("Provide exactly one of --checkpoint_path or --onnx_path.")

    model = None
    policy = None
    seq_len = None
    obs_dim = None
    action_mode = None

    if cfg.model.checkpoint_path:
        checkpoint_path = to_absolute_path(cfg.model.checkpoint_path)
        model = BCTransformerLit.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        model.eval()
        model.to(cfg.eval.device)
        seq_len = int(model.hparams.cfg.train.seq_len)
        obs_dim = int(model.hparams.cfg.data.obs_dim)

        print(f"Model: {model.net}, {sum(p.numel() for p in model.net.parameters() if p.requires_grad)} parameters")
    else:
        onnx_path = to_absolute_path(cfg.model.onnx_path)
        policy = OnnxPolicy(onnx_path)

    env = SBGSingleObsEnv(
        env_path=to_absolute_path(cfg.env.env_path) if cfg.env.env_path else None,
        show_window=cfg.env.viz,
        seed=cfg.env.seed,
        n_parallel=1,
        speedup=cfg.env.speedup,
        obs_key=cfg.env.obs_key,
    )

    episode_rewards = []
    n_envs = 16
    pbar = tqdm(total=cfg.eval.n_episodes, desc="Evaluating")
    while len(episode_rewards) < cfg.eval.n_episodes:
        raw_obs = _reset_env(env)
        obs = _extract_obs(raw_obs, cfg.env.obs_key)
        obs = np.asarray(obs, dtype=np.float32)

        if model is not None:
            if obs.shape[-1] != obs_dim:
                raise ValueError(f"Expected obs shape ({obs_dim},), got {obs.shape}.")
            obs_history: Deque[np.ndarray] = deque(maxlen=seq_len)
            obs_history.append(obs)
        done = np.array(False).repeat(16)
        total_reward = np.array(0.0).repeat(16)
        steps = 0

        while not done.all():
            if model is not None:
                obs_t, mask_t, valid_len = _build_window(obs_history, seq_len, obs_dim, n_envs)
                obs_t = obs_t.to(cfg.eval.device)
                mask_t = mask_t.to(cfg.eval.device)

                with torch.no_grad():
                    pred = model.net(obs_t, mask_t)
                    action = pred[:, valid_len - 1].cpu().numpy()
            else:
                action = np.array([policy.predict(x[None]) for x in obs])
            raw_obs, reward, done, _ = _step_env(env, action)
            obs = _extract_obs(raw_obs, cfg.env.obs_key)
            obs = np.asarray(obs, dtype=np.float32)
            if model is not None:
                obs_history.append(obs)

            total_reward += reward
            
            if done.any():
                done_indices = np.where(done)[0]
                for idx in done_indices:
                    episode_rewards.append(total_reward[idx])
                    total_reward[idx] = 0.0
                    pbar.update(1)
                if len(episode_rewards) >= cfg.eval.n_episodes:
                    break
                
                curr_mean = float(np.mean(episode_rewards))
                pbar.set_description(f"Evaluating (mean reward: {curr_mean:.3f})")

            steps += 1
            if cfg.eval.max_steps is not None and steps >= cfg.eval.max_steps:
                break

    env.close()
    mean_reward = float(np.mean(episode_rewards)) if episode_rewards else float("nan")
    print(f"Mean reward over {len(episode_rewards)} episodes: {mean_reward:.3f}+-{np.std(episode_rewards):.3f}")


if __name__ == "__main__":
    main()
