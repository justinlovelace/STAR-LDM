import os

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf

from star_ldm.models.modules.diffusion import SinusoidalPosEmb
from star_ldm.models.modules.blocks import FeedForward
from star_ldm.data.CONSTANTS import DATA_STATS_PATH


class ConditionableMLP(nn.Module):
    """MLP with FiLM time conditioning on each layer."""

    def __init__(self, input_dim, mlp_dim, hidden_dim, n_layers, time_cond_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, mlp_dim)
        self.mlp = nn.ModuleList([
            FeedForward(
                dim=mlp_dim,
                hidden_dim=hidden_dim,
                mult=None,
                time_cond_dim=time_cond_dim,
                dropout=0.1,
                drop_seq_dim=True,
                init_zeros=True,
            )
            for _ in range(n_layers)
        ])
        self.output = nn.Linear(mlp_dim, 1)

    def forward(self, x, time_emb):
        x = self.input_proj(x)
        for ff in self.mlp:
            x = x + ff(x, time_emb=time_emb)
        return self.output(x)


class NoiseConditionedMLP(nn.Module):
    """
    Noise-conditioned binary classifier operating on sentence embeddings.

    Used for classifier-guided diffusion: given a noisy sentence embedding and
    noise level alpha^2, predicts logits for a binary attribute (e.g. sentiment,
    toxicity). During generation the gradient of the classifier's log-probability
    w.r.t. the noisy embedding steers the diffusion sampling.

    Args:
        mlp_dim: Hidden dimension of the MLP layers.
        mlp_hidden_dim: Inner dimension of the FeedForward blocks.
        mlp_depth: Number of FeedForward layers.
        sentence_emb_dim: Dimension of the input sentence embeddings.
        global_norm: If True, use global (scalar) mean/std for normalization.
    """

    def __init__(
        self,
        mlp_dim=768,
        mlp_hidden_dim=1536,
        mlp_depth=4,
        sentence_emb_dim=768,
        global_norm=True,
    ):
        super().__init__()

        time_emb_dim = sentence_emb_dim // 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(sentence_emb_dim),
            nn.Linear(sentence_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.noise_conditioned_mlp = ConditionableMLP(
            input_dim=sentence_emb_dim,
            mlp_dim=mlp_dim,
            hidden_dim=mlp_hidden_dim,
            n_layers=mlp_depth,
            time_cond_dim=time_emb_dim,
        )

        self.scale_by_std = True
        dataset_name = 'fineweb_100b'
        if global_norm:
            self.register_buffer('data_mean', torch.load(
                os.path.join(DATA_STATS_PATH[dataset_name], 'global_mean.pt'), weights_only=True))
            self.register_buffer('data_std', torch.load(
                os.path.join(DATA_STATS_PATH[dataset_name], 'global_std.pt'), weights_only=True))
        else:
            self.register_buffer('data_mean', torch.load(
                os.path.join(DATA_STATS_PATH[dataset_name], 'mean.pt'), weights_only=True))
            self.register_buffer('data_std', torch.load(
                os.path.join(DATA_STATS_PATH[dataset_name], 'std.pt'), weights_only=True))

        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def normalize_sentence_emb(self, sentence_emb):
        return (sentence_emb - self.data_mean) / self.data_std

    def get_logits(self, noisy_sentence_emb, alpha2):
        """
        Compute classification logits from noisy sentence embeddings.

        Args:
            noisy_sentence_emb: ``(B, D)`` noisy embeddings (already normalized).
            alpha2: ``(B, 1)`` noise level.

        Returns:
            Logits of shape ``(B, 1)``.
        """
        alpha2_time = rearrange(alpha2, 'b () -> b')
        time_emb = self.time_mlp(alpha2_time * 1000)
        return self.noise_conditioned_mlp(noisy_sentence_emb, time_emb)

    def get_loss(self, noisy_sentence_emb, alpha2, labels):
        """
        Compute per-sample BCE loss (used for classifier guidance gradients).

        Args:
            noisy_sentence_emb: ``(B, D)`` noisy embeddings.
            alpha2: ``(B, 1)`` noise level.
            labels: ``(B, 1)`` binary targets.

        Returns:
            Per-sample loss of shape ``(B,)``.
        """
        logits = self.get_logits(noisy_sentence_emb, alpha2)
        return self.loss_fn(logits, labels).squeeze(dim=1)


def load_classifier(checkpoint_path, device='cuda'):
    """
    Load a pretrained :class:`NoiseConditionedMLP` classifier from a checkpoint
    directory or ``.pt`` file.

    The checkpoint is expected to follow the EMA convention used during training:
    the ``ema`` key contains the EMA model weights.

    Args:
        checkpoint_path: Path to checkpoint directory (containing ``best_model.pt``
            or ``model.pt``) or a direct ``.pt`` file.
        device: Device to load the model onto.

    Returns:
        A :class:`NoiseConditionedMLP` in eval mode on the specified device.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    if os.path.isdir(checkpoint_path):
        config_path = os.path.join(checkpoint_path, 'config.yaml')
        # Prefer best_model.pt, fall back to model.pt
        pt_path = os.path.join(checkpoint_path, 'best_model.pt')
        if not os.path.exists(pt_path):
            pt_path = os.path.join(checkpoint_path, 'model.pt')
    else:
        pt_path = checkpoint_path
        config_path = os.path.join(os.path.dirname(checkpoint_path), 'config.yaml')

    # Build model from config if available, otherwise use defaults
    if os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
        model = NoiseConditionedMLP(
            mlp_dim=cfg.mlp.mlp_arch.dim,
            mlp_hidden_dim=cfg.mlp.mlp_arch.hidden_dim,
            mlp_depth=cfg.mlp.mlp_arch.depth,
            sentence_emb_dim=cfg.get('embedding_dim', 768),
            global_norm=cfg.get('global_norm', True),
        )
    else:
        model = NoiseConditionedMLP()

    checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)

    # Load from EMA weights if available
    if 'ema' in checkpoint:
        from ema_pytorch import EMA
        ema = EMA(model, beta=0.999, update_every=10, power=3 / 4, update_after_step=1000)
        ema.load_state_dict(checkpoint['ema'], strict=False)
        model = ema.ema_model
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model = model.to(device)
    model.eval()
    return model
