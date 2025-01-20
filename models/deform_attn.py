from typing import List, Optional, Tuple, NamedTuple
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from einops import rearrange, repeat

from utils.other_utils import get_rotation_matrix, safe_sigmoid



class OffsetNet(nn.Module):
    def __init__(
        self,
        embed_dims=128+3,
        num_learnable_offsets=6,
        fix_scale=None,
    ):
        super(OffsetNet, self).__init__()
        self.embed_dims = embed_dims
        self.num_learnable_offsets = num_learnable_offsets
        self.fix_scale = np.array(fix_scale)
        self.num_offsets = len(self.fix_scale) + num_learnable_offsets
        if num_learnable_offsets > 0:
            self.learnable_fc = nn.Linear(self.embed_dims, num_learnable_offsets * 3)


    def init_weight(self):
        if self.num_learnable_offsets > 0:
            nn.init.xavier_uniform_(self.learnable_fc.weight)
            if self.learnable_fc.bias is not None:
                nn.init.constant_(self.learnable_fc.bias, 0.0)

    def update_pc_range(self, pts3d):
   
        self.pc_range = [
            torch.min(pts3d[:, 0]), torch.min(pts3d[:, 1]), torch.min(pts3d[:, 2]),
            torch.max(pts3d[:, 0]), torch.max(pts3d[:, 0]), torch.max(pts3d[:, 0])
        ]

    def forward(
        self,
        gaussians_means, # [bs, num_gaussians, 3]
        gaussians_scales,
        gaussians_rotations,
        gaussians_features, # [bs, num_heads, num_gaussians, feature_dims]
    ):
        # pts3d = pts3d.reshape(-1, 3)
        # self.update_pc_range(pts3d)
        bs, num_heads, num_gaussians = gaussians_features.shape[:3]

        # generate learnable offsets for deformable attention
        fix_scale = gaussians_means.new_tensor(self.fix_scale)
        scale = fix_scale[None, None].tile([bs, num_heads, num_gaussians, 1, 1])

        if self.num_learnable_offsets > 0:
            offset_input = torch.cat([gaussians_means.unsqueeze(1).repeat(1, num_heads, 1, 1), gaussians_features], dim=-1)
            learnable_scale = (
                safe_sigmoid(self.learnable_fc(offset_input)
                .reshape(bs, num_heads, num_gaussians, self.num_learnable_offsets, 3))
                - 0.5  # [-0.5, 0.5]
            )
            scale = torch.cat([scale, learnable_scale], dim=-2)
        

        offset_points = scale * gaussians_scales 
        rotation_mat = get_rotation_matrix(gaussians_rotations).transpose(-1, -2)
        
        offset_points = torch.matmul(rotation_mat[:, :, None], offset_points[..., None]).squeeze(-1)

        offset_points = offset_points + gaussians_means.unsqueeze(2) # [bs, num_heads, num_gaussians, num_offsets, 3]

        return offset_points


class DeformableAttention(nn.Module):
    def __init__(
        self,
        attn_dims: int = 512,
        num_heads: int = 8,
        num_offsets: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        gaussian_feature_dims: int = 32, # feature_dims per plane
        image_feature_dims: int = 256,
        offsetnet_params: dict = None,
        residual_mode="add",
        sample_gaussians: int = 6870,
    ):
        super(DeformableAttention, self).__init__()
        assert attn_dims % num_heads == 0, "attn_dims must be divisible by num_heads"

        self.head_dims = attn_dims // num_heads
        self.attn_dims = attn_dims
        self.num_heads = num_heads
        self.num_offsets = num_offsets
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.sample_gaussians = sample_gaussians
        offsetnet_params["embed_dims"] = attn_dims + 3
        offsetnet_params["num_learnable_offsets"] = num_offsets
        self.offsetnet = OffsetNet(**offsetnet_params)

        self.scale = self.head_dims**-0.5
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.to_q = nn.Linear(gaussian_feature_dims, attn_dims, bias = False)
        self.to_k = nn.Conv2d(image_feature_dims, attn_dims, bias = False)
        self.to_v = nn.Conv2d(image_feature_dims, attn_dims, bias = False)
        self.output_proj = nn.Linear(attn_dims, gaussian_feature_dims)


    def forward(
        self,
        gaussians: NamedTuple,
        gaussians_features: torch.Tensor, # [bs, num_gaussians, 3, gaussian_feature_dims]
        cameras: NamedTuple,
    ):
        
        # generate offsets
        num_gaussians = gaussians.means.shape[0]
        sample_idxs = random.sample(range(num_gaussians), self.sample_gaussians) if self.sample_gaussians < num_gaussians else None

        gaussians_means = gaussians.means[sample_idxs] if sample_idxs else gaussians.means
        gaussians_scales = gaussians.scales[sample_idxs] if sample_idxs else gaussians.scales
        gaussians_rotations = gaussians.rotations[sample_idxs] if sample_idxs else gaussians.rotations
        # gaussians_features = gaussians.latent_features[sample_idxs] if sample_idxs else gaussians.latent_features

        # gaussian_features [bs, num_gaussians, 3, feature_dims] -> [bs, heads, num_gaussians, 3, attn_dims//heads]
        query_features = self.to_q(gaussians_features).reshape(-1, num_gaussians, 3, self.num_heads, self.head_dims).permute(0, 3, 1, 2, 4)

        # TODO: transform the gaussians to pose space
        transforms_mat = gaussians.fwd_transform()
        homo_coord = torch.ones(num_gaussians, 1, dtype=torch.float32, device=gaussians_means.device)
        x_hat_homo = torch.cat([gaussians_means, homo_coord], dim=-1).view(num_gaussians, 4, 1)
        gaussians_means = torch.matmul(transforms_mat, x_hat_homo)[:, :3, 0]

        offset_points = self.offsetnet(
            gaussians_means,
            gaussians_scales,
            gaussians_rotations,
            query_features,
        ) # [bs, num_heads, num_gaussians, num_offsets, 3]

        image_features = cameras.semantic_features
        points_2d = self.project_points(offset_points, cameras.projection_matrix, image_features.shape[-2:])     
        # [bs, heads, num_gaussians, num_offsets, 2]

        # get image features at projected offset points
        key_features = self.to_k(image_features)
        value_features = self.to_v(image_features)

        _, c, h, w = key_features.shape
        # 归一化 points2d 到 [-1, 1] 范围
        points2d_normalized = points_2d.clone()
        points2d_normalized[..., 0] = 2.0 * points_2d[..., 0] / (w - 1) - 1.0
        points2d_normalized[..., 1] = 2.0 * points_2d[..., 1] / (h - 1) - 1.0
        # 将 points2d 调整为 [bs * heads * num_gaussians, num_offsets, 1, 2]
        grid = points2d_normalized.reshape(-1, self.num_offsets, 1, 2)

        # 对 feature_map 进行采样
        key_features = F.grid_sample(
            key_features.repeat(self.num_heads * num_gaussians, 1, 1, 1),
            grid,
            mode='bilinear',
            padding_mode = 'zeros',
            align_corners=False
        )  # [bs * heads * num_gaussians, c, num_offsets, 1]
        key_features = key_features.squeeze(-1).permute(0, 2, 3, 1).reshape(-1, self.num_heads, num_gaussians, self.num_offsets, self.attn_dims) # [bs, heads, num_gaussians, num_offsets, c]

        value_features = F.grid_sample(
            value_features.repeat(self.num_heads * num_gaussians, 1, 1, 1),
            grid,
            mode='bilinear',
            padding_mode = 'zeros',
            align_corners=False
        )  # [bs * heads * num_gaussians, c, num_offsets, 1]
        value_features = value_features.squeeze(-1).permute(0, 2, 3, 1).reshape(-1, self.num_heads, num_gaussians, self.num_offsets, self.attn_dims)

        # attention
        bs = gaussians_features.shape[0]
        query_features = query_features.reshape(bs, self.num_heads, num_gaussians, 3, self.attn_dims).unsqueeze(-2)  # [bs, heads, num_gaussians, 3, 1, feat_dims]
        key_features = key_features.unsqueeze(-3).repeat(1, 1, 1, 3, 1, 1) # [bs, heads, num_gaussians, 3, num_offsets, feat_dims]
        value_features = value_features.unsqueeze(-3).repeat(1, 1, 1, 3, 1, 1) # [bs, heads, num_gaussians, 3, num_offsets, feat_dims]

        attn_scores = torch.matmul(query_features, key_features.transpose(-1, -2)) * self.scale  # [bs, heads, num_gaussians, 3, 1, num_offsets]
        # numerical stability
        attn_scores = attn_scores - attn_scores.amax(dim = -1, keepdim = True).detach()
        # attention
        attn_scores = attn_scores.softmax(dim = -1)
        attn_scores = self.attn_drop(attn_scores)
        output = torch.matmul(attn_scores.unsqueeze(-2), value_features).squeeze(-2) # [bs, heads, num_gaussians, 3, feat_dims]
        output = rearrange(output, 'b h g i d -> b g i (h d)') # [bs, num_gaussians, 3, gaussian_feature_dims]
        output = self.proj_drop(self.output_proj(output))

        output = output + gaussians_features

        return output


    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        "project 3d key_points to 2d image space"
        bs, num_heads, num_gaussians, num_offsets = key_points.shape[:4]

        pts_extend = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)

        points_2d = torch.matmul(projection_mat[:, None, None], pts_extend.unsqueeze(-1)).squeeze(-1)  # (bs, num_heads, num_gaussians, num_offsets, 3)

        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
            points_2d = torch.clamp(points_2d, min=0.0, max=0.9999)
            points_2d = points_2d.permute(0, 2, 3, 1, 4).reshape(bs, num_heads, num_gaussians, num_offsets, 2)

        return points_2d
