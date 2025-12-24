
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from imitation_learning.dataset import Trajectory, WindowedTrajectoryDataset



def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    obs = torch.stack([b["obs"] for b in batch], dim=0)   # [B, L, obs_dim]
    acts = torch.stack([b["acts"] for b in batch], dim=0) # [B, L, ...] or [B, L]
    mask = torch.stack([b["mask"] for b in batch], dim=0) # [B, L]
    return {"obs": obs, "acts": acts, "mask": mask}



class ILDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_trajectories: Sequence[Union[Trajectory, Tuple[Sequence, Sequence]]],
        val_trajectories: Optional[Sequence[Union[Trajectory, Tuple[Sequence, Sequence]]]],
        seq_len: int,
        obs_dim: int,
        action_mode: str,
        action_dim: Optional[int],
        num_actions: Optional[int],
        batch_size: int = 64,
        num_workers: int = 4,
        stride: int = 1,
        drop_last_short: bool = False,
        seed: int = 0,
    ):
        super().__init__()
        self._train_trajs = train_trajectories
        self._val_trajs = val_trajectories
        
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.action_mode = action_mode
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.stride = stride
        self.drop_last_short = drop_last_short
        self.seed = seed

    def setup(self, stage: Optional[str] = None):
        self.train_ds = WindowedTrajectoryDataset(
            self._train_trajs,
            seq_len=self.seq_len,
            obs_dim=self.obs_dim,
            action_mode=self.action_mode,
            action_dim=self.action_dim,
            num_actions=self.num_actions,
            stride=self.stride,
            drop_last_short=self.drop_last_short,
            seed=self.seed,
        )
        if self._val_trajs is None:
            self.val_ds = None
        else:
            self.val_ds = WindowedTrajectoryDataset(
                self._val_trajs,
                seq_len=self.seq_len,
                obs_dim=self.obs_dim,
                action_mode=self.action_mode,
                action_dim=self.action_dim,
                num_actions=self.num_actions,
                stride=self.stride,
                drop_last_short=self.drop_last_short,
                seed=self.seed + 1,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
            pin_memory=True,
        )