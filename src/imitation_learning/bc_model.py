from typing import Optional

import torch
import torch.nn as nn
from stable_baselines3.common.distributions import DiagGaussianDistribution

class TransformerBC(nn.Module):
    """
    Encoder-only Transformer that maps obs sequences -> action predictions per timestep.
    """
    def __init__(
        self,
        obs_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
        action_mode: str = "continuous",
        action_dim: Optional[int] = None,
        num_actions: Optional[int] = None,
    ):
        super().__init__()
        assert action_mode in ("continuous", "discrete")
        self.obs_dim = obs_dim
        self.action_mode = action_mode
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.d_model = d_model
        self.max_len = max_len

        self.obs_proj = nn.Linear(obs_dim, d_model)

        # learned positional embedding (simple + effective for BC)
        self.pos_emb = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
        self.causal_mask = None

        if action_mode == "continuous":
            if action_dim is None:
                raise ValueError("action_dim must be set for continuous mode")
            self.head = nn.Linear(d_model, action_dim)
        else:
            if num_actions is None:
                raise ValueError("num_actions must be set for discrete mode")
            self.head = nn.Linear(d_model, num_actions)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        obs: [B, L, obs_dim]
        mask: [B, L] True where valid
        returns:
          continuous -> [B, L, action_dim]
          discrete   -> [B, L, num_actions] logits
        """
        B, L, _ = obs.shape
        if L > self.max_len:
            raise ValueError(f"Sequence length {L} exceeds max_len {self.max_len}")

        x = self.obs_proj(obs)  # [B, L, d_model]
        if self.causal_mask is None or self.causal_mask.size(0) != L:
            self.causal_mask = torch.triu(
                torch.ones(L, L, device=obs.device) * float("-inf"),
                diagonal=1,
            )
        pos = torch.arange(L, device=obs.device).unsqueeze(0).expand(B, L)
        x = x + self.pos_emb(pos)

        # Transformer uses src_key_padding_mask where True means "PAD to ignore".
        # our mask is True for valid => pad_mask = ~mask
        pad_mask = ~mask
        x = self.encoder(x, src_key_padding_mask=pad_mask, mask=self.causal_mask, is_causal=True)
        out = self.head(x)
        return out


class MLPBC(nn.Module):
    """
    Simple MLP baseline for behavior cloning.
    Can operate on single-step obs [B, obs_dim] or sequences [B, L, obs_dim].
    """
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        action_mode: str = "continuous",
        action_dim: Optional[int] = None,
        num_actions: Optional[int] = None,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        assert action_mode in ("continuous", "probabilistic")
        self.action_mode = action_mode
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.obs_dim = obs_dim

        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        # tanh final layer
        #layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)
        
        self.distribution = DiagGaussianDistribution(action_dim)
        self.mean_layer, self.log_std = self.distribution.proba_distribution_net(hidden_dim)
        
    def log_prob(self, actions: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
        """
        actions: [B, (L,), action_dim]
        means:   [B, (L,), action_dim]
        returns:
          log_prob: [B, (L,)]
        """
        orig_shape = actions.shape
        dist = self.distribution.proba_distribution(means.reshape(-1, self.action_dim),self.log_std)
        log_prob = dist.log_prob(actions.reshape(-1, self.action_dim))
        
        return log_prob.reshape(orig_shape[:-1])

    def forward(self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        obs:
          [B, obs_dim] or [B, L, obs_dim]
        mask: (unused, for API compatibility with TransformerBC)
        returns:
          continuous -> [B, (L,), action_dim]
          discrete   -> [B, (L,), num_actions] logits
        """
        if obs.dim() == 3:
            B, L, D = obs.shape
            assert D == self.obs_dim
            x = obs.reshape(B * L, D)
            x = self.net(x)
            x = x.reshape(B, L, -1)
        elif obs.dim() == 2:
            B, D = obs.shape
            assert D == self.obs_dim
            x = self.net(obs)  # [B, hidden_dim]
        else:
            raise ValueError(f"obs must be 2D or 3D, got shape {obs.shape}")

        mean = self.mean_layer(x)

        return mean
