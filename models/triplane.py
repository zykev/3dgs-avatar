# from https://github.com/apple/ml-hugs/blob/main/hugs/models/modules/triplane.py

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.deform_transformer import DeformableTransformer

EPS = 1e-3

class CrossAttention(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, num_heads: int):
        """
        初始化 Multi-Head Cross Attention 模块。
        
        Args:
            features: 输入通道数。
            hidden_dim: 每个头的特征维度。
            num_heads: 注意力头的数量。
        """
        super(CrossAttention, self).__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        # 多头 Query, Key, Value
        self.query = nn.Conv2d(n_features, hidden_dim, kernel_size=1, bias=False)
        self.key = nn.Conv2d(n_features, hidden_dim, kernel_size=1, bias=False)
        self.value = nn.Conv2d(n_features, hidden_dim, kernel_size=1, bias=False)

        # 输出投影
        self.proj = nn.Conv2d(hidden_dim, n_features, kernel_size=1)

    def forward(self, x1, x2):
        """
        计算两个平面之间的 Multi-Head Cross Attention。
        
        Args:
            x1: 第一个平面，形状为 [1, features, H, W]
            x2: 第二个平面，形状为 [1, features, H, W]

        Returns:
            融合后的特征，形状为 [1, features, H, W]
        """
        B, C, H, W = x1.shape

        # 获取 Query, Key, Value
        query = self.query(x1).view(B, self.num_heads, self.head_dim, -1)  # [B, num_heads, head_dim, H*W]
        key = self.key(x2).view(B, self.num_heads, self.head_dim, -1)      # [B, num_heads, head_dim, H*W]
        value = self.value(x2).view(B, self.num_heads, self.head_dim, -1) # [B, num_heads, head_dim, H*W]

        # 计算注意力权重
        attention = torch.einsum("bhcn,bhcm->bhnm", query, key) * self.scale # [B, num_heads, H*W, H*W]
        attention = F.softmax(attention, dim=-1)

        # 加权求和
        fused = torch.einsum("bhnm,bhcm->bhcn", attention, value)  # [B, num_heads, hidden_dim, H*W]
        fused = fused.view(B, C, H, W)  # 恢复形状为 [B, features, H, W]

        # 输出投影
        fused = self.proj(fused)  # [1, features, H, W]
        return fused



act_fn_dict = {
    'softplus': torch.nn.Softplus(),
    'relu': torch.nn.ReLU(),
    'gelu': torch.nn.GELU(),
    'tanh': torch.nn.Tanh(),
}

class Normalize(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=self.dim, p=2)

class Clamp(nn.Module):
    def __init__(self, min_val, max_val) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, self.min_val, self.max_val)

class Softplus(nn.Module):
    def __init__(self, scale_factor: int) -> None:
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return self.scale_factor * torch.nn.functional.softplus(x)
    

class AppearanceDecoder(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=64, act='gelu'):
        super().__init__()
        self.hidden_dim = hidden_dim
            
        self.net = torch.nn.Sequential(
            nn.Conv2d(n_features, self.hidden_dim, kernel_size=3, stride=1, padding=1),
            act_fn_dict[act],
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1),
            act_fn_dict[act],
        )
        self.opacity = nn.Sequential(nn.Conv2d(self.hidden_dim, 1, kernel_size=1), nn.Sigmoid())
        self.shs = nn.Conv2d(self.hidden_dim, 16*3, kernel_size=1)
        
    def forward(self, x):

        x = self.net(x)
        shs = self.shs(x)
        opacity = self.opacity(x)
        output = torch.cat([opacity, shs], dim=1)
        return output 
    
    

class GeometryDecoder(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=128, act='gelu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.net = torch.nn.Sequential(
            nn.Conv2d(n_features, self.hidden_dim, kernel_size=3, stride=1, padding=1),
            act_fn_dict[act],
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1),
            act_fn_dict[act],
        )
        self.xyz = nn.Sequential(nn.Conv2d(self.hidden_dim, 3, kernel_size=1), Clamp(-1, 1))
        self.rotations = nn.Sequential(nn.Conv2d(self.hidden_dim, 4, kernel_size=1), Normalize(dim=-1))
        self.scales = nn.Sequential(nn.Conv2d(self.hidden_dim, 3, kernel_size=1), Softplus(scale_factor=10))

        
    def forward(self, x):
        x = self.net(x)
        xyz = self.xyz(x)
        rotations = self.rotations(x)
        scales = self.scales(x)

        output = torch.cat([xyz, rotations, scales], dim=1)
        return output
    

class PlanePredictor(nn.Module):
    def __init__(self, n_features: int, apperance_hidden_dim: int, geometry_hidden_dim: int):
        """
        初始化 PlanePredictor。

        Args:
            input_features: 输入通道数。
            output_features: 输出通道数。
        """
        super(PlanePredictor, self).__init__()
        self.cross_attention = CrossAttention(n_features)
        self.appearance_decoder = AppearanceDecoder(n_features, apperance_hidden_dim)
        self.geometry_decoder = GeometryDecoder(n_features, geometry_hidden_dim)

    def forward(self, plane_xy, plane_xz, plane_yz):
        """
        前向传播，合并平面并通过 CNN 预测。

        Args:
            plane_xy: [1, features, resX, resY]
            plane_xz: [1, features, resX, resZ]
            plane_yz: [1, features, resY, resZ]

        Returns:
            输出张量，形状为 [1, output_features, resX, resY]。
        """
        # 两两计算 Cross Attention
        fused_xy_xz = self.cross_attention(plane_xy, plane_xz)  # [1, features, H, W]
        fused_xz_yz = self.cross_attention(plane_xz, plane_yz)  # [1, features, H, W]
        fused_yz_xy = self.cross_attention(plane_yz, plane_xy)  # [1, features, H, W]

        fused_xy = torch.cat([plane_xy, fused_xy_xz, fused_yz_xy], dim=1)
        fused_xz = torch.cat([plane_xz, fused_xy_xz, fused_xz_yz], dim=1)
        fused_yz = torch.cat([plane_yz, fused_yz_xy, fused_xz_yz], dim=1)


        # 对每个融合特征分别进行解码
        plane_properties = []
        for fused in [fused_xy, fused_xz, fused_yz]:
            geometry_out = self.geometry_decoder(fused)
            appearance_out = self.appearance_decoder(fused)
            plane_properties.append(torch.cat(geometry_out, appearance_out, dim=1))
            
        plane_properties = torch.stack(plane_properties, dim=1)

        return plane_properties # [3, features_per_plane, H, W] (xyz, rot, scale, opacity, shs)
    
class GaussianTriPlane(nn.Module):
    def __init__(self, smpl_metadata, features=32, resX=256, resY=256, resZ=256):
        super().__init__()

        self.aabb = smpl_metadata['aabb']

        self.plane_xy = nn.Parameter(torch.randn(1, features, resX, resY))
        self.plane_xz = nn.Parameter(torch.randn(1, features, resX, resZ))
        self.plane_yz = nn.Parameter(torch.randn(1, features, resY, resZ))
        self.plane_predictor = PlanePredictor(features * 3, features) # triplane feature -> gaussian properties

        self.feature_optimizer = DeformableTransformer(embed_dim=32, num_blocks=4)


    def query_triplane_feature(self, xyz):

        B, N, _ = xyz.shape
        xyz = xyz.reshape(B, N, 1, 3)  # (B, N, 1, 3)
        # align_corners=True ==> the extrema (-1 and 1) considered as the center of the corner pixels
        # F.grid_sample: [1, C, H, W], [1, N, 1, 2] -> [1, C, N, 1]
        feat_xy = F.grid_sample(self.plane_xy, xyz[..., [0, 1]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        feat_xz = F.grid_sample(self.plane_xz, xyz[..., [0, 2]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        feat_yz = F.grid_sample(self.plane_yz, xyz[..., [1, 2]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        feat = torch.stack([feat_xy, feat_xz, feat_yz], dim=2)  # (B, N, 3, features)
        return feat

    def query_gaussian_properties(self, gaussian_means, triplane_properties):
        """
        查询高斯点的属性值。

        Args:
            gaussian_coords: Tensor, 形状为 [bs, n_gaussians, 3]，表示高斯的三维坐标 (x, y, z)。
            outputs: 字典，来自 PlanePredictor 的 forward 输出，包含每个平面的几何和外观属性。

        Returns:
            一个字典，包含每个高斯点的属性值，每个属性是一个 Tensor, 形状为 [bs, n_gaussians, 属性维度]。
        """
        bs, n_gaussians, _ = gaussian_means.shape

        gaussian_properties = []

        # 遍历每个平面
        for i, plane in enumerate(triplane_properties):
            # 将 3D 坐标投影到对应平面的 2D 坐标
            if i == 0: # "plane_xy":
                coords_2d = gaussian_means[..., :2]  # 取 (x, y)
            elif i == 1: # "plane_xz":
                coords_2d = gaussian_means[..., [0, 2]]  # 取 (x, z)
            elif i == 2: #"plane_yz":
                coords_2d = gaussian_means[..., 1:]  # 取 (y, z)

            # 将 2D 坐标归一化到 [-1, 1]
            h, w = plane["xyz_offsets"].shape[-2:]  # 获取平面分辨率
            coords_2d_normalized = coords_2d.clone()
            coords_2d_normalized[..., 0] = coords_2d[..., 0] / (w - 1) * 2 - 1  # x 归一化
            coords_2d_normalized[..., 1] = coords_2d[..., 1] / (h - 1) * 2 - 1  # y 归一化


            # 添加 batch 维度和通道维度
            plane = plane.unsqueeze(0).expand(bs, -1, -1, -1)  # [bs, C, H, W]
            coords_2d_normalized = coords_2d_normalized.unsqueeze(2)  # [bs, n_gaussians, 1, 2]

            # 使用 grid_sample 查询
            plane_properties = F.grid_sample(plane, coords_2d_normalized, mode="bilinear", align_corners=True)  # [1, C, bs, n_gaussians]

            # 转置为 [bs, n_gaussians, C]
            plane_properties = plane_properties.squeeze(0).permute(1, 2, 0)
            gaussian_properties.append(plane_properties)

        # 对三个平面的特征取平均
        gaussian_properties = torch.stack(gaussian_properties, dim=0)  # (3, bs, num_points, c)
        gaussian_properties = torch.mean(gaussian_properties, dim=0)  # (bs, num_points, c)

        gaussian_properties[:, :, :3] = gaussian_means + gaussian_properties[:, :, :3]

        return gaussian_properties

    def update_triplane(
            self,
            gaussian_features: torch.Tensor,  # [bs, num_gaussians, 3, feat_dim]
            gaussian_xyz: torch.Tensor,     # [bs, num_gaussians, 3]
    ):
        """
        更新 GaussianTriPlane 的特征。

        Args:
            gaussian_features: 高斯特征，形状为 [bs, num_gaussians, 3, feat_dim]。
            gaussian_xyz: 高斯均值，形状为 [bs, num_gaussians, 3]。
        """

        def update_plane_with_gaussians(plane, gaussian_indices, gaussian_features):
            """
            更新给定的 plane tensor 中的特征。

            Args:
                plane: [feat_dim, h, w], 表示三维平面特征。
                gaussian_indices: [bs, num_gaussians, 2], 每个高斯的二维坐标 (x, y)。
                gaussian_features: [bs, num_gaussians, feat_dim], 每个高斯的子平面特征。

            Returns:
                更新后的 plane tensor, 维度为 [1, feat_dim, h, w]。
            """
            _, feat_dim, h, w = plane.shape
            bs, num_gaussians, _ = gaussian_indices.shape

            # 获取高斯点的整数部分和小数部分
            px, py = gaussian_indices[..., 0], gaussian_indices[..., 1]  # [bs, num_gaussians]
            px0, py0 = px.floor().long(), py.floor().long()  # 左上角整数坐标
            px1, py1 = px0 + 1, py0 + 1  # 右下角整数坐标

            # 保证整数坐标不超出 plane 边界
            px0.clamp_(0, w - 1)
            px1.clamp_(0, w - 1)
            py0.clamp_(0, h - 1)
            py1.clamp_(0, h - 1)

            # 计算每个高斯点的插值权重
            wx0, wx1 = px1 - px, px - px0  # 水平方向权重
            wy0, wy1 = py1 - py, py - py0  # 垂直方向权重

            # 计算每个位置上的权重
            weights = torch.stack([
                wx0 * wy0,  # 左上
                wx0 * wy1,  # 左下
                wx1 * wy0,  # 右上
                wx1 * wy1,  # 右下
            ], dim=-1)  # [bs, num_gaussians, 4]

            # 计算更新位置的索引
            indices_list = torch.stack([py0, py1, py0, py1], dim=-1), torch.stack([px0, px0, px1, px1], dim=-1)
            linear_indices = indices_list[0] * w + indices_list[1]  # 将二维索引转为一维
            linear_indices = linear_indices.unsqueeze(-1)  # [bs, num_gaussians, 4, 1]

            # 广播 features 和权重
            weighted_features = (gaussian_features.unsqueeze(2) * weights.unsqueeze(-1))  # [bs, num_gaussians, 4, feat_dim]

            # 展平 indices 和 features
            linear_indices = linear_indices.reshape(bs * num_gaussians * 4, 1)
            weighted_features = weighted_features.reshape(bs * num_gaussians * 4, feat_dim)

            # 累加到 plane 中
            plane_flat = plane.view(feat_dim, -1)
            scatter_indices = linear_indices.expand(-1, feat_dim).T  # 广播 indices
            plane_flat.scatter_add_(1, scatter_indices, weighted_features.T)

            return plane_flat.view(1, feat_dim, h, w)

        resX, resY, resZ = self.plane_xy.shape[2], self.plane_xy.shape[3], self.plane_xz.shape[3]

        # 归一化 3D 坐标到三平面的范围
        gaussian_xyz = (gaussian_xyz + 1) / 2 # change range (-1, 1) to (0, 1)
        gaussian_xyz[..., 0] = torch.clamp(gaussian_xyz[..., 0], 0.0, 0.999) * (resX - 1)
        gaussian_xyz[..., 1] = torch.clamp(gaussian_xyz[..., 1], 0.0, 0.999) * (resY - 1)
        gaussian_xyz[..., 2] = torch.clamp(gaussian_xyz[..., 2], 0.0, 0.999) * (resZ - 1)

        # 提取每个平面的高斯特征
        features_xy = gaussian_features[:, :, 0, :]  # [bs, num_gaussians, feat_dim]
        features_xz = gaussian_features[:, :, 1, :]
        features_yz = gaussian_features[:, :, 2, :]

        # 分别提取坐标
        x = gaussian_xyz[..., 0]  # [bs, num_gaussians]
        y = gaussian_xyz[..., 1]
        z = gaussian_xyz[..., 2]

        # 更新 plane_xy
        indices_xy = torch.stack([x, y], dim=-1)  # [bs, num_gaussians, 2]
        self.plane_xy = update_plane_with_gaussians(self.plane_xy, indices_xy, features_xy)

        # 更新 plane_xz
        indices_xz = torch.stack([x, z], dim=-1)  # [bs, num_gaussians, 2]
        self.plane_xz = update_plane_with_gaussians(self.plane_xz, indices_xz, features_xz)

        # 更新 plane_yz
        indices_yz = torch.stack([y, z], dim=-1)  # [bs, num_gaussians, 2]
        self.plane_yz = update_plane_with_gaussians(self.plane_yz, indices_yz, features_yz)



    def forward(self, iteration, camera, gaussians):

        image_feature = camera.image_feature

        gaussian_xyz = gaussians.get_xyz
        gaussian_xyz = self.aabb.normalize(gaussian_xyz, sym=True)
        gaussian_feature = gaussians.get_latent_feature

        if iteration == 0:
            gaussian_feature = self.query_triplane_feature(gaussian_xyz)
        
        # update gaussian feature by transformer with deformable attention
        gaussian_feature = self.feature_optimizer(gaussians, gaussian_feature, image_feature)

        # update triplane with optimized gaussian feature
        self.update_triplane(gaussian_feature, gaussian_xyz)

        # predict plane gaussian properties and query to update gaussian properties
        triplane_properties = self.plane_predictor(self.plane_xy, self.plane_xz, self.plane_yz)
        gaussian_properties = self.query_gaussian_properties(gaussian_xyz, triplane_properties)

        updated_gaussians = gaussians.clone()
        updated_gaussians._xyz = gaussian_properties[..., :3]
        updated_gaussians._rot = gaussian_properties[..., 3:7]
        updated_gaussians._scale = gaussian_properties[..., 7:10]
        updated_gaussians._opacity = gaussian_properties[..., 10:11]
        updated_gaussians._shs = gaussian_properties[..., 11:]
        updated_gaussians._latent_feature = gaussian_feature

        return updated_gaussians

def get_updater(cfg, metadata):
    return GaussianTriPlane(cfg, metadata)