from typing import Optional

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from imitation_learning.bc_model import MLPBC, TransformerBC

class BCTransformerLit(pl.LightningModule):
    def __init__(
        self,
        obs_dim: int,
        cfg,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        grad_clip: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        if cfg.model.name == "transformer":
            net = TransformerBC(
                obs_dim=cfg.data.obs_dim,
                d_model=cfg.model.d_model,
                nhead=cfg.model.nhead,
                num_layers=cfg.model.num_layers,
                dim_feedforward=cfg.model.dim_feedforward,
                dropout=cfg.model.dropout,
                max_len=cfg.train.seq_len,
                action_mode=cfg.train.action_mode,
                action_dim=cfg.data.action_dim,
                num_actions=cfg.train.num_actions,
            )
        elif cfg.model.name == "mlp":
            net = MLPBC(
                obs_dim=cfg.data.obs_dim,
                hidden_dim=cfg.model.hidden_dim,
                n_layers=cfg.model.n_layers,
                action_mode=cfg.train.action_mode,
                action_dim=cfg.data.action_dim,
                num_actions=cfg.train.num_actions,  
            )

        self.net = net

    def training_step(self, batch, batch_idx):
        loss, logs = self._shared_step(batch, stage="train")
        self.log_dict(logs, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self._shared_step(batch, stage="val")
        self.log_dict(logs, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def _shared_step(self, batch, stage: str):
        obs = batch["obs"]
        acts = batch["acts"]
        mask = batch["mask"]

        pred = self.net(obs, mask)

        if self.net.action_mode == "continuous":
            # mask: [B,L] -> [B,L,1]
            m = mask.unsqueeze(-1).float()
            mse = (F.mse_loss(pred, acts, reduction="none") * m).sum() / (m.sum().clamp_min(1.0))
            loss = mse
            logs = {f"{stage}/loss": loss, f"{stage}/mse": mse}
        else:
            log_prob = self.net.log_prob(acts, pred)
            loss = - (log_prob * mask).sum() / mask.sum().clamp_min(1.0)
            prob_acts = (log_prob.exp() * mask).sum() / mask.sum().clamp_min(1.0)
            
            
            logs = {f"{stage}/loss": loss, f"{stage}/prob_acts": prob_acts}
        return loss, logs

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # simple cosine schedule (optional but nice)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
        return {"optimizer": opt, "lr_scheduler": sched}

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        if self.hparams.grad_clip and self.hparams.grad_clip > 0:
            self.clip_gradients(optimizer, gradient_clip_val=self.hparams.grad_clip, gradient_clip_algorithm="norm")

