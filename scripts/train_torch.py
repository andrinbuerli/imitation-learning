import math
import random
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from imitation_learning.bc_model import MLPBC, TransformerBC
from imitation_learning.callbacks import EvalCallback
from imitation_learning.data_module import ILDataModule
from imitation_learning.dataset import Trajectory
from imitation_learning.trainer import BCTransformerLit
try:
    from pytorch_lightning.loggers import WandbLogger
except ImportError:
    WandbLogger = None
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import json
from pathlib import Path
    
def export_onnx_from_checkpoint(checkpoint_path: str, export_path: str) -> None:
    model = BCTransformerLit.load_from_checkpoint(checkpoint_path)
    model.eval()
    net = model.net

    seq_len = int(model.hparams.seq_len)
    obs_dim = int(model.hparams.obs_dim)

    obs = torch.zeros(1, seq_len, obs_dim, dtype=torch.float32)
    mask = torch.ones(1, seq_len, dtype=torch.bool)

    export_path = os.fspath(export_path)
    if not export_path.endswith(".onnx"):
        export_path = f"{export_path}.onnx"
    export_dir = os.path.dirname(export_path)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

    torch.onnx.export(
        net,
        (obs, mask),
        export_path,
        input_names=["obs", "mask"],
        output_names=["actions"],
        dynamic_axes={
            "obs": {0: "batch", 1: "seq"},
            "mask": {0: "batch", 1: "seq"},
            "actions": {0: "batch", 1: "seq"},
        },
        opset_version=17,
    )
    
def split_trajectories(
    trajectories: Sequence[Union[Trajectory, Tuple[Sequence, Sequence]]],
    val_frac: float = 0.1,
    seed: int = 0,
):
    trajs = list(trajectories)
    rng = random.Random(seed)
    rng.shuffle(trajs)
    n_val = max(1, int(len(trajs) * val_frac)) if len(trajs) > 1 else 0
    val = trajs[:n_val] if n_val > 0 else None
    train = trajs[n_val:] if n_val > 0 else trajs
    return train, val


@hydra.main(config_path="../configs", config_name="train_torch", version_base="1.3")
def main(cfg: DictConfig):
    data_file = Path(__file__).parent.parent / cfg.data.file  
    trajectories = json.load(data_file.open())
    
    cfg.data.obs_dim = len(trajectories[0][0][0])
    cfg.data.action_dim = len(trajectories[0][1][0])
    
    trajectories = [{"obs": x[0], "acts": x[1]} for x in trajectories]
    
    # cut last obs because there is not action for it
    for traj in trajectories:
        traj["obs"] = traj["obs"][:-1]
    
    train_trajs, val_trajs = split_trajectories(trajectories, val_frac=cfg.train.val_frac, seed=cfg.train.seed)

    dm = ILDataModule(
        train_trajectories=train_trajs,
        val_trajectories=val_trajs,
        seq_len=cfg.train.seq_len,
        obs_dim=cfg.data.obs_dim,
        action_mode=cfg.train.action_mode,
        action_dim=cfg.data.action_dim,
        num_actions=cfg.train.num_actions,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        stride=cfg.train.stride,
        drop_last_short=cfg.train.drop_last_short,
        seed=cfg.train.seed,
    )

    model = BCTransformerLit(
        obs_dim=cfg.data.obs_dim,
        cfg=cfg,
    )

    wandb_logger = None
    wandb_logger = WandbLogger(
        project=cfg.wandb.project if "project" in cfg.wandb else None,
        entity=cfg.wandb.entity if "entity" in cfg.wandb else None,
        log_model=False,
    )
    wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)

    checkpoint_dir = cfg.train.checkpoint_dir or "checkpoints"
    # checkpoint dir should be in hydra folder
    checkpoint_dir = os.path.join(HydraConfig.get().runtime.output_dir, checkpoint_dir)
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="last",
        save_last=True,
        save_top_k=1,
        every_n_epochs=1,
    )
    
    eval_callback =EvalCallback(
        env_path=cfg.eval.env_path,
        n_episodes=cfg.eval.episodes,
        obs_key=cfg.eval.obs_key,
        speedup=cfg.eval.speedup,
        seed=cfg.train.seed,
        viz=cfg.eval.viz,
        max_steps=cfg.eval.max_steps,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=20,
        gradient_clip_val=cfg.train.grad_clip,
        logger=wandb_logger if wandb_logger is not None else True,
        callbacks=[checkpoint_cb, eval_callback],
    )

    trainer.fit(model, datamodule=dm)
    if cfg.train.export_onnx_path:
        ckpt_path = checkpoint_cb.last_model_path
        if not ckpt_path:
            raise RuntimeError("No checkpoint available to export.")
        export_onnx_from_checkpoint(ckpt_path, cfg.train.export_onnx_path)
    return model


if __name__ == "__main__":
    main()