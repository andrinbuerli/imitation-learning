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
        self.cfg = cfg
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
                token_dropout=cfg.model.token_dropout,
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

        if self.net.action_mode == "explicit":
            pred = self.net(obs, mask)
            log_prob = self.net.log_prob(acts, pred)
            loss = - (log_prob * mask).sum() / mask.sum().clamp_min(1.0)
            prob_acts = (log_prob.exp() * mask).sum() / mask.sum().clamp_min(1.0)
            
            logs = {f"{stage}/loss": loss, f"{stage}/prob_acts": prob_acts}
        elif self.net.action_mode == "implicit":
            # energy-based model loss
            batch_size, seq_len = acts.shape[:2]
            
            eps = torch.randn_like(acts) * self.cfg.train.ebm_noise_std
            acts_tilde = acts + eps
            
            # concat obs and acts_tilde
            inpt = torch.cat([obs, acts_tilde], dim=-1)
            scores = self.net(inpt, mask).squeeze(-1)  # [B, L]
            
            grad_scores = torch.autograd.grad(
                scores.sum(), acts_tilde, create_graph=True
            )[0]  # [B, L, action_dim]
            
            
            denoising_score_match = ((grad_scores + eps / (self.cfg.train.ebm_noise_std ** 2)) ** 2).sum(dim=-1)  # [B, L]
            
            loss = (denoising_score_match * mask).sum() / mask.sum().clamp_min(1.0)
            
            logs = {f"{stage}/loss": loss, f"{stage}/ebm_std": self.cfg.train.ebm_noise_std}
            
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

