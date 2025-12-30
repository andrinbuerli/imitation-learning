import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset




Trajectory = Dict[str, Any]  # expects keys: "obs", "acts"


def _standardize_trajectories(
    trajectories: Sequence[Union[Trajectory, Tuple[Sequence, Sequence]]]
) -> List[Trajectory]:
    """Accepts either dict trajectories {"obs": ..., "acts": ...}
    or tuple trajectories (obs_list, act_list) and returns dict form.
    """
    out: List[Trajectory] = []
    for tr in trajectories:
        if isinstance(tr, dict):
            obs = tr.get("obs", None)
            acts = tr.get("acts", None)
            if obs is None or acts is None:
                raise ValueError("Trajectory dict must have keys 'obs' and 'acts'")
            out.append({"obs": obs, "acts": acts})
        else:
            obs, acts = tr
            out.append({"obs": obs, "acts": acts})
    return out

class WindowedTrajectoryDataset(Dataset):
    """
    Creates training samples as fixed-length windows from trajectories.

    Each sample:
      - obs_seq: [L, obs_dim]
      - act_seq: [L, ...] (aligned with obs_seq)
      - attn_mask: [L] (True where valid, False where padded)
    """

    def __init__(
        self,
        trajectories: Sequence[Union[Trajectory, Tuple[Sequence, Sequence]]],
        seq_len: int,
        obs_dim: int,
        action_mode: str = "explicit",  # "explicit" or "explicit"
        action_dim: Optional[int] = None,  # for continuous
        num_actions: Optional[int] = None,  # for probabilistic
        stride: int = 1,
        drop_last_short: bool = False,
        seed: int = 0,
    ):
        super().__init__()
        assert action_mode in ("explicit", "implicit")
        self.action_mode = action_mode
        self.seq_len = int(seq_len)
        self.obs_dim = int(obs_dim)
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.stride = int(stride)
        self.drop_last_short = drop_last_short

        self.trajectories = _standardize_trajectories(trajectories)

        random.seed(seed)

        # Precompute an index mapping: dataset_idx -> (traj_idx, start_t)
        self._index: List[Tuple[int, int]] = []
        for ti, tr in enumerate(self.trajectories):
            obs = tr["obs"]
            acts = tr["acts"]
            if len(obs) != len(acts):
                raise ValueError(f"Trajectory {ti}: obs and acts must have same length")

            T = len(obs)
            if T == 0:
                continue

            if self.drop_last_short and T < self.seq_len:
                continue

            # windows start at 0..T-1 with stride
            for start in range(0, T, self.stride):
                # allow windows that go beyond T (we'll pad)
                self._index.append((ti, start))

        if len(self._index) == 0:
            raise ValueError("No windows created. Check seq_len/drop_last_short/trajectory lengths.")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx, start = self._index[idx]
        tr = self.trajectories[traj_idx]

        obs_list = tr["obs"]
        act_list = tr["acts"]

        # slice [start : start+seq_len]
        end = start + self.seq_len
        obs_slice = obs_list[start:end]
        act_slice = act_list[start:end]

        # Convert to tensors and pad
        L = len(obs_slice)
        assert 1 <= L <= self.seq_len

        # obs: [L, obs_dim]
        obs = torch.as_tensor(np.asarray(obs_slice, dtype=np.float32))
        if obs.ndim != 2 or obs.shape[1] != self.obs_dim:
            raise ValueError(
                f"Obs must be [T, obs_dim={self.obs_dim}], got {tuple(obs.shape)}"
            )

        if self.action_dim is None:
            raise ValueError("action_dim must be set for continuous mode")
        act = torch.as_tensor(np.asarray(act_slice, dtype=np.float32))
        if act.ndim == 1:
            # allow scalar action, interpret as [L, 1]
            act = act.unsqueeze(-1)
        if act.ndim != 2 or act.shape[1] != self.action_dim:
            raise ValueError(
                f"Acts must be [T, action_dim={self.action_dim}], got {tuple(act.shape)}"
            )

        # pad to [seq_len, ...]
        pad_len = self.seq_len - L
        attn_mask = torch.zeros(self.seq_len, dtype=torch.bool)
        attn_mask[:L] = True

        if pad_len > 0:
            obs_pad = torch.zeros(pad_len, self.obs_dim, dtype=torch.float32)
            obs = torch.cat([obs, obs_pad], dim=0)

            act_pad = torch.zeros(pad_len, self.action_dim, dtype=torch.float32)
            act = torch.cat([act, act_pad], dim=0)

        return {"obs": obs, "acts": act, "mask": attn_mask}