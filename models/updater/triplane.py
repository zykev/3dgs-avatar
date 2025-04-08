# from https://github.com/apple/ml-hugs/blob/main/hugs/models/modules/triplane.py

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# from models.deform_transformer import DeformableTransformer
from models.updater.gaussian_pred import PropertyPredictor
from utils.general_utils import quaternion_multiply


def print_grad(name):
    def hook(grad):
        print(f"Gradient of {name}: {grad.mean().item() if grad is not None else 'None'}")
    return hook

EPS = 1e-4

act_fn_dict = {
    'softplus': torch.nn.Softplus(),
    'relu': torch.nn.ReLU(),
    'gelu': torch.nn.GELU(),
    'tanh': torch.nn.Tanh(),
    'silu': torch.nn.SiLU(),
}

class Normalize(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=self.dim, p=2, eps=EPS)

class Clamp(nn.Module):
    def __init__(self, min_val, max_val) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, self.min_val, self.max_val)

# class Softplus(nn.Module):
#     def __init__(self, scale_factor: int) -> None:
#         super().__init__()
#         self.scale_factor = scale_factor

#     def forward(self, x):
#         return self.scale_factor * torch.nn.functional.softplus(x)

xyz_activation = nn.Tanh()
rot_activation = Normalize(dim=-1)
scale_activation = nn.Sigmoid()
opacity_activation = nn.Sigmoid()
rgb_activation = Clamp(min_val=-0.5, max_val=0.5)

# class CrossAttention(nn.Module):
#     def __init__(self, n_features: int, hidden_dim: int, num_heads: int):
#         """
#         初始化 Multi-Head Cross Attention 模块。
        
#         Args:
#             features: 输入通道数。
#             hidden_dim: 每个头的特征维度。
#             num_heads: 注意力头的数量。
#         """
#         super(CrossAttention, self).__init__()
#         assert hidden_dim % num_heads == 0
#         self.num_heads = num_heads
#         self.hidden_dim = hidden_dim
#         self.head_dim = hidden_dim // num_heads
#         self.scale = self.head_dim**-0.5

#         # 多头 Query, Key, Value
#         self.query = nn.Conv2d(n_features, hidden_dim, kernel_size=1, bias=False)
#         self.key = nn.Conv2d(n_features, hidden_dim, kernel_size=1, bias=False)
#         self.value = nn.Conv2d(n_features, hidden_dim, kernel_size=1, bias=False)

#         # 输出投影
#         self.proj = nn.Conv2d(hidden_dim, n_features, kernel_size=1)

#     def forward(self, x1, x2):
#         """
#         计算两个平面之间的 Multi-Head Cross Attention。
        
#         Args:
#             x1: 第一个平面，形状为 [1, features, H, W]
#             x2: 第二个平面，形状为 [1, features, H, W]

#         Returns:
#             融合后的特征，形状为 [1, features, H, W]
#         """
#         _,c, h, w = x1.shape

#         # 获取 Query, Key, Value
#         query = self.query(x1).view(1, self.num_heads, self.head_dim, -1)  # [B, num_heads, head_dim, H*W]
#         key = self.key(x2).view(1, self.num_heads, self.head_dim, -1)      # [B, num_heads, head_dim, H*W]
#         value = self.value(x2).view(1, self.num_heads, self.head_dim, -1) # [B, num_heads, head_dim, H*W]

#         # 计算注意力权重
#         attention = torch.einsum("bhcn,bhcm->bhnm", query, key) * self.scale # [B, num_heads, H*W, H*W]
#         attention = F.softmax(attention, dim=-1)

#         # 加权求和
#         fused = torch.einsum("bhnm,bhcm->bhcn", attention, value)  # [B, num_heads, hidden_dim, H*W]
#         fused = fused.reshape(1, -1, h, w)  # 恢复形状为 [B, features, H, W]

#         # 输出投影
#         fused = self.proj(fused)  # [1, features, H, W]
#         return fused


class CrossAttention(nn.Module):
    def __init__(self, n_features, reduction=4):
        """
        Cross-Attention module with downsampling and upsampling to reduce computation.
        
        Args:
        - in_channels (int): Number of input channels (assuming grayscale features).
        - reduction (int): Factor by which the feature map size is reduced.
        """
        super(CrossAttention, self).__init__()
        hidden_dim = n_features // 2
        self.scale = hidden_dim**-0.5

        # Downsample feature maps to reduce computation
        self.downsample = nn.Conv2d(n_features, n_features, kernel_size=3, stride=reduction, padding=1)

        # Convolutions for query, key, and value
        self.query_conv = nn.Conv2d(n_features, hidden_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(n_features, hidden_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(n_features, n_features, kernel_size=1)

        # Upsample to restore original size
        self.upsample = nn.ConvTranspose2d(n_features, n_features, kernel_size=3, stride=reduction, output_padding=1)
        self.fusion_conv = nn.Conv2d(n_features*3, n_features, kernel_size=1)

    def forward(self, x1, x2, x3):
        """
        Args:
        - x1: Tensor of shape (B, C, 64, 64)
        - x2: Tensor of shape (B, C, 64, 64)
        
        Returns:
        - output: Tensor of shape (B, C, 64, 64)
        """
        # Downsample to reduce computation
        x1_down = self.downsample(x1)  # (B, C, H//r, W//r)
        x2_down = self.downsample(x2)  # (B, C, H//r, W//r)
        x3_down = self.downsample(x3)
        

        # Compute query, key, and value
        query = self.query_conv(x1_down) 
        key_a = self.key_conv(x2_down)     
        value_a = self.value_conv(x2_down)
        key_b = self.key_conv(x3_down)
        value_b = self.value_conv(x3_down)

        # Compute attention map
        _, _, h, w = query.shape  
        attn_map_a = torch.einsum('bci,bcj->bij', query.flatten(2), key_a.flatten(2)) * self.scale  # (B, h*w, h*w)
        attn_map_a = F.softmax(attn_map_a, dim=-1)
        attn_feature_a = torch.einsum('bij,bcj->bci', attn_map_a, value_a.flatten(2)).view(1, -1, h, w)  # (B, C, h, w)
        attn_feature_a = self.upsample(attn_feature_a)  # (B, C, H, W)

        attn_map_b = torch.einsum('bci,bcj->bij', query.flatten(2), key_b.flatten(2)) * self.scale  # (B, h*w, h*w)
        attn_map_b = F.softmax(attn_map_b, dim=-1)
        attn_feature_b = torch.einsum('bij,bcj->bci', attn_map_b, value_b.flatten(2)).view(1, -1, h, w)  # (B, C, h, w)
        attn_feature_b = self.upsample(attn_feature_b)  # (B, C, H, W)


        # Concatenate with original features and refine
        out = torch.cat([attn_feature_a, attn_feature_b, x1], dim=1)  # (B, 3C, 64, 64)
        out = self.fusion_conv(out)  # (B, C, 64, 64)


        return out

    

class AppearanceDecoder(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=64, act='silu'):
        super().__init__()
        self.hidden_dim = hidden_dim
            
        self.net = torch.nn.Sequential(
            nn.Conv2d(n_features, self.hidden_dim, kernel_size=3, stride=1, padding=1),
            act_fn_dict[act],
            # nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1),
            # act_fn_dict[act],
        )
        self.opacity = nn.Sequential(nn.Conv2d(self.hidden_dim, 1, kernel_size=1))
        self.color_shs = nn.Conv2d(self.hidden_dim, 16*3, kernel_size=1)
        self.color_rgb = nn.Sequential(nn.Conv2d(self.hidden_dim, 3, kernel_size=1))
        
    def forward(self, x, fix_opacity=False, use_sh=False):

        x = self.net(x)
        if use_sh:
            color = self.color_shs(x)   
        else:
            color = self.color_rgb(x)
        if not fix_opacity:
            opacity = self.opacity(x)
        else:
            opacity = torch.ones_like(color[:, :1])
        output = torch.cat([opacity, color], dim=1)
        return output 
    
    

class GeometryDecoder(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=128, act='silu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.net = torch.nn.Sequential(
            nn.Conv2d(n_features, self.hidden_dim, kernel_size=3, stride=1, padding=1),
            act_fn_dict[act],
            # nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1),
            # act_fn_dict[act],
        )
        self.xyz = nn.Sequential(nn.Conv2d(self.hidden_dim, 3, kernel_size=1))
        self.rotations = nn.Sequential(nn.Conv2d(self.hidden_dim, 4, kernel_size=1))
        self.scales = nn.Sequential(nn.Conv2d(self.hidden_dim, 3, kernel_size=1))

        
    def forward(self, x):
        x = self.net(x)
        xyz = self.xyz(x)
        rotations = self.rotations(x)
        scales = self.scales(x)

        output = torch.cat([xyz, rotations, scales], dim=1)
        return output
    

class PlanePredictor(nn.Module):
    def __init__(self, n_features: int, attn_reduction: int, apperance_hidden_dim: int, geometry_hidden_dim: int,
                 fix_opacity=False):
        """
        初始化 PlanePredictor。

        Args:
            input_features: 输入通道数。
            output_features: 输出通道数。
        """
        super(PlanePredictor, self).__init__()
        self.cross_attention = CrossAttention(n_features, attn_reduction)
        self.appearance_decoder = AppearanceDecoder(n_features, apperance_hidden_dim)
        self.geometry_decoder = GeometryDecoder(n_features, geometry_hidden_dim)


    def forward(self, plane_xy, plane_xz, plane_yz, fix_opacity=False, use_sh=False):    
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
        fused_xy = self.cross_attention(plane_xy, plane_xz, plane_yz)  # [1, features, H, W]
        fused_xz = self.cross_attention(plane_xz, plane_xy, plane_yz)  # [1, features, H, W]
        fused_yz = self.cross_attention(plane_yz, plane_xy, plane_xz)  # [1, features, H, W]


        # 对每个融合特征分别进行解码
        plane_properties = []
        for fused in [fused_xy, fused_xz, fused_yz]:
            geometry_out = self.geometry_decoder(fused)
            appearance_out = self.appearance_decoder(fused, fix_opacity, use_sh)
            plane_properties.append(torch.cat([geometry_out, appearance_out], dim=1))

        plane_properties = torch.cat(plane_properties, dim=0)  # [3, features, H, W]    
        # plane_properties = torch.stack(plane_properties, dim=1)

        return plane_properties # [3, features_per_plane, H, W] (xyz, rot, scale, opacity, shs)
    
class GaussianTriPlane(nn.Module):
    def __init__(self, smpl_metadata, feature_dim=32, res=256, trainable=True, device='cuda'):
        super().__init__()

        self.aabb = smpl_metadata['aabb']

        self.feature_dim = feature_dim
        self.res = res
        self.trainable = trainable
        self.device = device

        self.init_weights()

    def create_plane(self):
        """创建一个随机初始化的平面"""
        plane = nn.Parameter(torch.randn(1, self.feature_dim, self.res, self.res, device=self.device))
        return plane

    def init_weights(self):

        self.plane_xy = self.create_plane()
        self.plane_xz = self.create_plane()
        self.plane_yz = self.create_plane()

        self.layer_norm = nn.LayerNorm([self.feature_dim, self.res, self.res])  # Layer Normalization


    def query_triplane(self, xyz):
        N, _ = xyz.shape  # Now assuming xyz has shape (N, 3) for a single example
        xyz = xyz.reshape(N, 1, 3).unsqueeze(0)  # (N, 1, 3)

        # Perform grid sampling for each plane
        feat_xy = F.grid_sample(self.plane_xy, xyz[..., [0, 1]], align_corners=True)[0, ..., 0].transpose(0, 1)  # (N, features)
        feat_xz = F.grid_sample(self.plane_xz, xyz[..., [0, 2]], align_corners=True)[0, ..., 0].transpose(0, 1)
        feat_yz = F.grid_sample(self.plane_yz, xyz[..., [1, 2]], align_corners=True)[0, ..., 0].transpose(0, 1)

        # Stack the features from the three planes
        feat = torch.stack([feat_xy, feat_xz, feat_yz], dim=1).view(N, -1)  # (N, 3, features) --> (N, 3*features)

        return feat

    def gaussian_blur(self, plane, kernel_size=5, sigma=1.0):
        """
        对 triplane 进行高斯模糊，以扩散梯度影响范围。
        """
        channels = plane.shape[1]
        kernel = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32, device=plane.device)
        kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
        kernel = kernel / kernel.sum()

        # 生成二维高斯核
        kernel_2d = kernel[:, None] * kernel[None, :]
        kernel_2d = kernel_2d.expand(channels, 1, kernel_size, kernel_size)  # 形状 (C, 1, k, k)

        # 使用 conv2d 进行模糊
        plane = F.conv2d(plane, kernel_2d, padding=kernel_size // 2, groups=channels)

        return plane

    def update_triplane(self, gaussian_features: torch.Tensor, gaussian_xyz: torch.Tensor):
        """
        更新 GaussianTriPlane 的特征。

        Args:
            gaussian_features: 高斯特征，形状为 [num_gaussians, 3, feat_dim]。
            gaussian_xyz: 高斯均值，形状为 [num_gaussians, 3]。
        """
        def update_plane_with_gaussians(plane, gaussian_indices, gaussian_features):
            """
            更新给定的 plane tensor 中的特征。

            Args:
                plane: [feat_dim, h, w], 表示triplane子平面特征。
                gaussian_indices: [num_gaussians, 2], 每个高斯点在triplane 子平面(h, w)中的二维坐标 (x, y)。
                gaussian_features: [num_gaussians, feat_dim], 每个高斯点的特征。

            Returns:
                更新后的 plane tensor, 维度为 [1, feat_dim, h, w]。
            """
            _, feat_dim, h, w = plane.shape
            num_gaussians, _ = gaussian_indices.shape

            # 获取高斯点的整数部分和小数部分
            px, py = gaussian_indices[..., 0], gaussian_indices[..., 1]  # [num_gaussians]
            px0, py0 = px.floor().long(), py.floor().long()  # 左上角整数坐标
            px1, py1 = px0 + 1, py0 + 1  # 右下角整数坐标

            # 保证整数坐标不超出 plane 边界
            px0.clamp_(0, w - 1)
            px1.clamp_(0, w - 1)
            py0.clamp_(0, h - 1)
            py1.clamp_(0, h - 1)

            # 计算每个高斯点的插值权重
            wx0, wx1 = (px1 - px).clamp(0, 1), (px - px0).clamp(0, 1)  # 水平方向权重
            wy0, wy1 = (py1 - py).clamp(0, 1), (py - py0).clamp(0, 1)  # 垂直方向权重

            # 计算每个位置上的权重
            weights = torch.stack([
                wx0 * wy0,  # 左上
                wx0 * wy1,  # 左下
                wx1 * wy0,  # 右上
                wx1 * wy1,  # 右下
            ], dim=-1)  # [num_gaussians, 4]

            # 计算更新位置的索引
            indices_list = torch.stack([py0, py1, py0, py1], dim=-1), torch.stack([px0, px0, px1, px1], dim=-1)
            linear_indices = indices_list[0] * w + indices_list[1]  # 将二维索引转为一维
            linear_indices = linear_indices.unsqueeze(-1)  # [num_gaussians, 4, 1]

            # 广播 features 和权重
            weighted_features = (gaussian_features.unsqueeze(1) * weights.unsqueeze(-1))  # [num_gaussians, 4, feat_dim]

            # 展平 indices 和 features
            linear_indices = linear_indices.reshape(num_gaussians * 4, 1)
            weighted_features = weighted_features.reshape(num_gaussians * 4, feat_dim)

            # 在新plane中添加高斯特征
            new_plane = torch.zeros_like(plane.view(feat_dim, -1))
            new_plane = plane.view(feat_dim, -1) + (new_plane - plane.view(feat_dim, -1)).detach()
            # new_plane = plane.view(feat_dim, -1).clone()

            scatter_indices = linear_indices.expand(-1, feat_dim).T  # 广播 indices
            new_plane = new_plane.scatter_add(1, scatter_indices, weighted_features.T)

            counts = torch.zeros_like(new_plane).scatter_add(1, scatter_indices, torch.ones_like(weighted_features.T))
            new_plane = new_plane / (counts + 1e-6)
            new_plane = new_plane.view(1, feat_dim, h, w)
            

            new_plane = self.layer_norm(new_plane)

            new_plane = self.gaussian_blur(new_plane, kernel_size=5, sigma=1.0)

            new_plane = new_plane + plane
            
            # plane_flat = torch.nn.functional.normalize(plane_flat, dim=0, p=2, eps=EPS)

            return new_plane


        # 3D 坐标 (-1, 1) -> (0, 1) -> (0, res-1)
        gaussian_xyz = (gaussian_xyz + 1) / 2 # change range (-1, 1) to (0, 1)
        x = torch.clamp(gaussian_xyz[..., 0], 0.0, 0.999) * (self.res - 1)
        y = torch.clamp(gaussian_xyz[..., 1], 0.0, 0.999) * (self.res - 1)
        z = torch.clamp(gaussian_xyz[..., 2], 0.0, 0.999) * (self.res - 1)

        # 提取每个平面的高斯特征
        # gaussian_features = torch.nn.functional.normalize(gaussian_features, dim=-1, p=2, eps=EPS)
        # gaussian_features.register_hook(print_grad('gsfeature'))
        
        gaussian_features = gaussian_features.view(gaussian_xyz.size(0), 3, -1)
        features_xy = gaussian_features[:, 0, :]  # [num_gaussians, feat_dim]
        features_xz = gaussian_features[:, 1, :]
        features_yz = gaussian_features[:, 2, :]

        # self.plane_xy.register_hook(print_grad('plane_xy'))

        # 更新 plane_xy
        indices_xy = torch.stack([x, y], dim=-1)  # [num_gaussians, 2]
        new_plane_xy = update_plane_with_gaussians(self.plane_xy, indices_xy, features_xy)

        # 更新 plane_xz
        indices_xz = torch.stack([x, z], dim=-1)  # [num_gaussians, 2]
        new_plane_xz = update_plane_with_gaussians(self.plane_xz, indices_xz, features_xz)

        # 更新 plane_yz
        indices_yz = torch.stack([y, z], dim=-1)  # [num_gaussians, 2]
        new_plane_yz = update_plane_with_gaussians(self.plane_yz, indices_yz, features_yz)

        # new_plane_xy.register_hook(print_grad('new_plane_xy'))
        

        return new_plane_xy, new_plane_xz, new_plane_yz



class GaussianUpdater(nn.Module):
    def __init__(self, cfg, metadata):
        super().__init__()


        self.aabb = metadata['aabb']

        self.triplane_body = GaussianTriPlane(metadata, **cfg.triplane_body)

        # self.plane_predictor = PlanePredictor(**cfg.plane_predictor) # triplane feature -> gaussian properties

        self.property_predictor = PropertyPredictor(**cfg.property_predictor) # gaussian feature -> gaussian properties

        # self.feature_optimizer = DeformableTransformer(metadata=metadata,
        #                                                gaussian_feature_dim=cfg.feature_optimizer.gaussian_feature_dim,
        #                                                num_points_per_batch=cfg.feature_optimizer.num_points_per_batch,
        #                                                num_blocks=cfg.feature_optimizer.num_blocks,
        #                                                drop_rate=cfg.feature_optimizer.drop_rate,
        #                                                self_attn_args=cfg.feature_optimizer.self_attn_args,  # 传递整个字典
        #                                                deform_attn_args=cfg.feature_optimizer.deform_attn_args,  # 传递整个字典
        #                                                ffd_args=cfg.feature_optimizer.ffd_args  # 传递整个字典
        #                                                )
        # self.feature_optimizer = DeformableTransformer(metadata=metadata, **cfg.feature_optimizer)

        # self.fix_opacity = cfg.plane_predictor.get('fix_opacity', False)

        self.offset_max = 0.2
        self.scale_max = 0.02

    def query_gaussian_properties(self, gaussian_means, triplane_properties):
        """
        查询高斯点的属性值。

        Args:
            gaussian_coords: Tensor, 形状为 [n_gaussians, 3]，表示高斯的三维坐标 (x, y, z)。
            outputs: 字典，来自 PlanePredictor 的 forward 输出，包含每个平面的几何和外观属性。

        Returns:
            一个字典，包含每个高斯点的属性值，每个属性是一个 Tensor, 形状为 [n_gaussians, 属性维度]。
        """

        gaussian_properties = []

        # 遍历每个平面
        for i, plane in enumerate(triplane_properties):
            # 将 3D 坐标投影到对应平面的 2D 坐标
            if i == 0: # "plane_xy"
                coords_2d = gaussian_means[..., :2]  # 取 (x, y)
            elif i == 1: # "plane_xz"
                coords_2d = gaussian_means[..., [0, 2]]  # 取 (x, z)
            elif i == 2: #"plane_yz"
                coords_2d = gaussian_means[..., 1:]  # 取 (y, z)


            # 使用 grid_sample 查询
            plane_properties = F.grid_sample(plane.unsqueeze(0), coords_2d.view(1, -1, 1, 2), mode="bilinear", align_corners=True)[0, ..., 0].transpose(0, 1)  # [n_gaussians, C]
            gaussian_properties.append(plane_properties)

        # 对三个平面的特征取平均
        gaussian_properties = torch.stack(gaussian_properties, dim=0)  # (3, n_gaussians, c)
        gaussian_properties = torch.mean(gaussian_properties, dim=0)  # (n_gaussians, c)

        return gaussian_properties

    
    def update_gaussians(self, gaussians, gaussian_properties, gaussian_feature):
        """
        更新 gaussians 对象中的属性，并确保梯度不会丢失
        :param gaussians: 需要更新的 Gaussians 对象
        :param gaussian_properties: Tensor，包含新的高斯属性，形状为 [sum(mask), total_properties_dim]
        :param mask: bool 类型的 Tensor，表示需要更新的位置
        :return: 更新后的 gaussians 对象
        """
        # 复制原始 tensor 确保梯度传播
        # update_gaussians = gaussians.clone()

        # 高斯坐标的平移
        new_means = xyz_activation(gaussian_properties[:, :3]) * self.offset_max
        new_means = gaussians.get_xyz + new_means

        # 旋转
        new_rots = rot_activation(gaussian_properties[:, 3:7])
        new_rots = quaternion_multiply(gaussians.get_rotation, new_rots)

        # scaling
        new_scales = scale_activation(gaussian_properties[:, 7:10])
        new_scales = gaussians.get_scaling + new_scales

        # opacity
        new_opacity = opacity_activation(gaussian_properties[:, 10:11])

        if gaussians.use_sh:
            new_shs = gaussian_properties[:, 11:]
        
        else:
            new_rgb = rgb_activation(gaussian_properties[:, 11:]) + 0.5


        # new_means.register_hook(print_grad("xyz"))
        # new_rots.register_hook(print_grad("rots"))
        # new_scales.register_hook(print_grad("scaling"))
        # new_opacity.register_hook(print_grad("opacity"))
        # gaussian_feature.register_hook(print_grad("gaussian_feature"))


        d = {
            # 'xyz': new_means, 
            # 'rotation': new_rots, 
            'scaling': new_scales, 
            'opacity': new_opacity,
            'latent_feature': gaussian_feature.unsqueeze(-1)}
        
        if gaussians.use_sh:
            d.update({'features_dc': new_shs[..., :3].unsqueeze(1),
                'features_rest': new_shs[..., 3:].view(-1, 15, 3)})
        else:
            d.update({'features_dc': new_rgb[..., :1].unsqueeze(-1),
                'features_rest': new_rgb[..., 1:].unsqueeze(-1)})

        # gaussians._latent_feature = gaussian_feature.unsqueeze(-1) 
        # gaussians._xyz = new_means
        # gaussians._rotation = new_rots
        # gaussians._scaling = new_scales
        # gaussians._opacity = new_opacity
        # if gaussians.use_sh:
        #     gaussians._features_dc = new_shs[..., :3].unsqueeze(1)
        #     gaussians._features_rest = new_shs[..., 3:].view(-1, 15, 3)
        # else: 
        #     gaussians._features_dc = new_rgb[..., :1].unsqueeze(-1)
        #     gaussians._features_rest = new_rgb[..., 1:].unsqueeze(-1)
                   
        
            
        return new_means, new_rots, d

        
    def forward(self, gaussians_body, camera, iteration):

        gaussian_xyz_body = gaussians_body.get_xyz
        gaussian_xyz_body_norm = self.aabb.normalize(gaussian_xyz_body, sym=True)


        gaussian_feature_body = self.triplane_body.query_triplane(gaussian_xyz_body_norm)

        # gaussian_feature_body.register_hook(print_grad("gaussian_feature_body_0"))

        # update gaussian garment feature by transformer with deformable attention
        # gaussian_feature_body = self.feature_optimizer(gaussian_xyz_body, gaussian_feature_body, camera)

        # gaussian_feature_body.register_hook(print_grad("gaussian_feature_body"))

        # update gaussian properties by gaussian feature
        updated_gaussians = self.property_predictor(gaussians_body, gaussian_feature_body)
        
        # # update triplane with optimized gaussian feature
        # plane_xy, plane_xz, plane_yz = self.triplane_body.update_triplane(gaussian_feature_body, gaussian_xyz_body_norm)

        # # # predict plane gaussian properties and query to update gaussian properties
        # triplane_properties_body = self.plane_predictor(plane_xy, plane_xz, plane_yz, fix_opacity=self.fix_opacity, use_sh=gaussians_body.use_sh)
        # gaussian_properties_body = self.query_gaussian_properties(gaussian_xyz_body_norm, triplane_properties_body)

        # gaussians_xyz, gaussians_rot, gaussians_dict = self.update_gaussians(gaussians_body, gaussian_properties_body, gaussian_feature_body)

        return updated_gaussians

def get_updater(cfg, metadata):
    name = cfg.name
    model_dict = {
        "gaussian_mlp": GaussianUpdater,
    }
    return model_dict[name](cfg, metadata)
