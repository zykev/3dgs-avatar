# Code modified from
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py ;
# https://github.com/facebookresearch/deit/blob/main/models.py
# and https://github.com/facebookresearch/vissl/blob/main/vissl/models/trunks/vision_transformer.py


from functools import partial
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_

from models.deform_attn import DeformableAttention

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        attn_target: Callable = DeformableAttention,
        mlp_ratio: int = 4,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        ffn_dropout_rate: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        assert not isinstance(
            attn_target, nn.Module
        ), "attn_target should be a Callable. Otherwise attn_target is shared across blocks!"
        self.attn = attn_target()
        if drop_path > 0.0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
        self.norm_1 = norm_layer(dim)
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=ffn_dropout_rate,
        )
        self.norm_2 = norm_layer(dim)

        # TODO: resize image_features to different size in each block

    def forward(self, gaussians, gaussian_features: torch.Tensor, cameras):
        # bs, num_gaussians, 3, gaussian_feature_dims]
        gaussian_features = gaussian_features + self.drop_path(self.attn(gaussians, self.norm_1(gaussian_features), cameras))
        gaussian_features = gaussian_features + self.drop_path(self.mlp(self.norm_2(gaussian_features)))

        return gaussian_features



class DeformableTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        attn_target: Callable = DeformableAttention,
        block: Callable = Block,
        pre_transformer_layer: Optional[Callable] = None,
        post_transformer_layer: Optional[Callable] = None,
        drop_path_rate: float = 0.0,
        drop_path_type: str = "progressive",
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
        mlp_ratio: int = 4,
        ffn_dropout_rate: float = 0.0,
    ):
        """
        Simple Transformer with the following features

        """
        super().__init__()
        self.pre_transformer_layer = pre_transformer_layer
        if drop_path_type == "progressive":
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        elif drop_path_type == "uniform":
            dpr = [drop_path_rate for i in range(num_blocks)]
        else:
            raise ValueError(f"Unknown drop_path_type: {drop_path_type}")

        self.blocks = nn.Sequential(
            *[
                block(
                    dim=embed_dim,
                    attn_target=attn_target,
                    mlp_ratio=mlp_ratio,
                    ffn_dropout_rate=ffn_dropout_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(num_blocks)
            ]
        )
        self.post_transformer_layer = post_transformer_layer
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        gaussian,
        gaussian_features: torch.Tensor,
        cameras
    ):
        """
        Inputs
        - gaussian_features: [bs, num_gaussians, 3, feature_dims]

        """
        if self.pre_transformer_layer:
            gaussian_features = self.pre_transformer_layer(gaussian_features)

        for blk_id, blk in enumerate(self.blocks):
            gaussian_features = blk(gaussian, gaussian_features, cameras)
        if self.post_transformer_layer:
            gaussian_features = self.post_transformer_layer(gaussian_features)
        return gaussian_features