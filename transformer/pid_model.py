"""ParticleTransformer — same architecture as hepd2_transformer_pid/model.py,
restructured so the internal representations needed for OT calibration
(à la arXiv:2507.08867) are directly accessible:

    z_enc  : CLS embedding after the encoder      (B, d_model)  — calibration target
    z_head : hidden activation inside the head    (B, d_model)  — downstream check
    logits : classifier output                    (B, num_classes)

Input shape (B, 21, 8), output raw logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ParticleTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        n_features: int = 8,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model),
        )

        # head kept as separate modules so intermediates can be extracted
        self.head_norm = nn.LayerNorm(d_model)
        self.head_fc1 = nn.Linear(d_model, d_model)
        self.head_act = nn.GELU()
        self.head_drop = nn.Dropout(dropout)
        self.head_fc2 = nn.Linear(d_model, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 21, n_features) -> CLS embedding z_enc (B, d_model)."""
        B = x.size(0)
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        return x[:, 0]

    def head_from_latent(self, z_enc: torch.Tensor) -> torch.Tensor:
        """Push a (possibly calibrated) latent through the frozen head."""
        h = self.head_act(self.head_fc1(self.head_norm(z_enc)))
        return self.head_fc2(self.head_drop(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_from_latent(self.forward_features(x))

    @torch.no_grad()
    def latents(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """All representations relevant to the calibration study."""
        z_enc = self.forward_features(x)
        z_head = self.head_act(self.head_fc1(self.head_norm(z_enc)))
        logits = self.head_fc2(z_head)
        return {"z_enc": z_enc, "z_head": z_head, "logits": logits}

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
