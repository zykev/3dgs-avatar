#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general_utils import quaternion_multiply


def print_grad(name):
    def hook(grad):
        print(f"Gradient of {name}: {grad.mean().item() if grad is not None else 'None'}")
    return hook


class SineActivation(nn.Module):
    def __init__(self, omega_0=30) -> None:
        super().__init__()
        self.omega_0 = omega_0
    
    def forward(self, x):
        return torch.sin(self.omega_0 * x)

class Normalize(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=self.dim, p=2, eps=1e-6)

class Clamp(nn.Module):
    def __init__(self, min_val, max_val, trans=0.) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.trans = trans

    def forward(self, x):

        x = torch.clamp(x, self.min_val, self.max_val)
        x = x + self.trans

        return x

    
act_fn_dict = {
    'softplus': torch.nn.Softplus(),
    'relu': torch.nn.ReLU(),
    'sine': SineActivation(omega_0=30),
    'gelu': torch.nn.GELU(),
    'tanh': torch.nn.Tanh(),
    'silu': torch.nn.SiLU(),
}

xyz_activation = nn.Tanh()
rot_activation = Normalize(dim=-1)
scale_activation = nn.Softplus()
opacity_activation = nn.Sigmoid()
rgb_activation = Clamp(min_val=-0.5, max_val=0.5, trans=0.5)


class AppearanceDecoder(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=64, act='silu'):
        super().__init__()
        self.hidden_dim = hidden_dim
            
        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )
        self.opacity = nn.Sequential(nn.Linear(self.hidden_dim, 1), opacity_activation)
        self.rgb = nn.Sequential(nn.Linear(hidden_dim, 3), rgb_activation)
        self.shs = nn.Linear(hidden_dim, 16*3)
        
    def forward(self, x, use_sh=False):

        x = self.net(x)
        opacity = self.opacity(x)
        if use_sh:
            color = self.shs(x)
        else:
            color = self.rgb(x)
        
        return {'color': color, 'opacity': opacity}
    

# class DeformationDecoder(torch.nn.Module):
#     def __init__(self, n_features, hidden_dim=128, weight_norm=True, act='gelu', disable_posedirs=False):
#         super().__init__()
#         self.hidden_dim = hidden_dim

#         self.sine = SineActivation(omega_0=30)
#         self.disable_posedirs = disable_posedirs
        
#         self.net = torch.nn.Sequential(
#             nn.Linear(n_features, self.hidden_dim),
#             act_fn_dict[act],
#             nn.Linear(self.hidden_dim, self.hidden_dim),
#             act_fn_dict[act],
#         )
#         self.skinning_linear = nn.Linear(hidden_dim, hidden_dim)
#         self.skinning = nn.Linear(hidden_dim, 24)
        
#         if weight_norm:
#             self.skinning_linear = nn.utils.weight_norm(self.skinning_linear)
            
#         # initialize blendshapes to be zero, and skinning weights to be equal for every bone (after softmax activation)
#         if not disable_posedirs:
#             self.blendshapes = nn.Linear(hidden_dim, 3 * 207)
#             torch.nn.init.constant_(self.blendshapes.bias, 0.0)
#             torch.nn.init.constant_(self.blendshapes.weight, 0.0)
        
#     def forward(self, x):
#         x = self.net(x)
#         if not self.disable_posedirs:
#             posedirs = self.blendshapes(x)
#             posedirs = posedirs.reshape(207, -1)
            
#         lbs_weights = self.skinning(F.gelu(self.skinning_linear(x)))
#         lbs_weights = F.gelu(lbs_weights)
        
#         return {
#             'lbs_weights': lbs_weights,
#             'posedirs': posedirs if not self.disable_posedirs else None,
#         }
    

class GeometryDecoder(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=128, act='silu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )
        self.xyz = nn.Sequential(nn.Linear(self.hidden_dim, 3), xyz_activation)
        self.rotations = nn.Sequential(nn.Linear(self.hidden_dim, 4), rot_activation)
        self.scales = nn.Sequential(nn.Linear(self.hidden_dim, 3), scale_activation)

        self.init_weights()

    def init_weights(self):
        for m in self.xyz.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.01)
                # torch.nn.init.uniform_(m.weight, a=-1e-3, b=1e-3) 
                torch.nn.init.zeros_(m.bias)

        for m in self.scales.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.01)
                torch.nn.init.zeros_(m.bias)
            

    def forward(self, x):
        x = self.net(x)
        xyz = self.xyz(x)
        rotations = self.rotations(x)
        scales = self.scales(x)
                
        return {
            'xyz': xyz,
            'rotations': rotations,
            'scales': scales,
        }
    


class PropertyPredictor(nn.Module):
    def __init__(self, 
                 n_features: int, # gaussian_feature
                 apperance_hidden_dim: int, 
                 geometry_hidden_dim: int,
                 ):
        """
        初始化 PlanePredictor。

        Args:
            input_features: 输入通道数。
            output_features: 输出通道数。
        """
        super(PropertyPredictor, self).__init__()
        n_features = n_features * 3
        self.appearance_decoder = AppearanceDecoder(n_features, apperance_hidden_dim)
        self.geometry_decoder = GeometryDecoder(n_features, geometry_hidden_dim)

        self.offset_max = 0.2
        self.scale_max = 0.01


    def forward(self, gaussians, gaussian_feature):    
        """
        前向传播，合并平面并通过 CNN 预测。

        Args:
            gaussian_features: (n_pts, features) 的张量，表示高斯特征。

        Returns:
            输出张量，形状为 [1, output_features, resX, resY]。
        """
        


        # 对每个融合特征分别进行解码
        appearance = self.appearance_decoder(gaussian_feature, gaussians.use_sh)
        geometry = self.geometry_decoder(gaussian_feature)

        # 高斯坐标的平移
        xyz_offsets = geometry['xyz'] * self.offset_max
        gs_xyz = gaussians.get_xyz + xyz_offsets

        # 旋转
        gs_rots = quaternion_multiply(gaussians.get_rotation, geometry['rotations'])

        # scaling
        gs_scales = gaussians.get_scaling * geometry['scales']

        # opacity
        gs_opacity = appearance['opacity']

        gs_color = appearance['color']
        


        # gs_xyz.retain_grad()
        # gs_rots.retain_grad()
        # gs_scales.retain_grad()

        gs_xyz.register_hook(print_grad("xyz"))
        gs_rots.register_hook(print_grad("rots"))
        gs_scales.register_hook(print_grad("scaling"))
        gs_opacity.register_hook(print_grad("opacity"))
        gaussian_feature.register_hook(print_grad("gaussian_feature"))


        # d = {
        #     # 'xyz': gs_xyz, 
        #     # 'rotation': gs_rots, 
        #     'scaling': gs_scales, 
        #     'opacity': gs_opacity,
        #     'latent_feature': gaussian_feature.unsqueeze(-1)}
        
        # if gaussians.use_sh:
        #     d.update({'features_dc': gs_color[..., :3].unsqueeze(1),
        #         'features_rest': gs_color[..., 3:].view(-1, 15, 3)})
        # else:
        #     d.update({'features_dc': gs_color[..., :1].unsqueeze(-1),
        #         'features_rest': gs_color[..., 1:].unsqueeze(-1)})
            
        # 复制原始 tensor 确保梯度传播
        update_gaussians = gaussians.clone()

        update_gaussians._latent_feature = gaussian_feature.unsqueeze(-1) 
        update_gaussians._xyz = gs_xyz
        update_gaussians._rotation = gs_rots
        update_gaussians._scaling = gs_scales
        update_gaussians._opacity = gs_opacity
        if gaussians.use_sh:
            update_gaussians._features_dc = gs_color[..., :3].unsqueeze(1)
            update_gaussians._features_rest = gs_color[..., 3:].view(-1, 15, 3)
        else: 
            update_gaussians._features_dc = gs_color[..., :1].unsqueeze(-1)
            update_gaussians._features_rest = gs_color[..., 1:].unsqueeze(-1)



        return update_gaussians