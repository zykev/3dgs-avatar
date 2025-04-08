#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import torch.nn.functional as F
import os
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

# import trimesh
# import igl

class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2) # + 1e-6 * torch.eye(3, device=L.device)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.covariance_activation = build_covariance_from_scaling_rotation

        # self.fwd_transform = torch.empty(0)
        # self.rotation_precomp = torch.empty(0)



    def __init__(self, cfg):
        self.cfg = cfg

        # two modes: SH coefficient or feature
        # if name == "body":
        #     self.use_sh = cfg.use_sh_body
        # elif name == "garment":
        #     self.use_sh = cfg.use_sh_garment
        self.use_sh = cfg.use_sh
        self.active_sh_degree = 0
        if self.use_sh:
            self.max_sh_degree = cfg.sh_degree
            self.feature_dim = (self.max_sh_degree + 1) ** 2
        else:
            self.feature_dim = cfg.feature_dim

        self.latent_feature_dim = cfg.latent_feature_dim
        self.device = cfg.device

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._latent_feature = torch.empty(0)
        # self._label = torch.empty(0)
        self.setup_functions()

    def clone(self):
        cloned = GaussianModel(self.cfg)

        # properties = [
        #     # "fwd_transform",
        #               "rotation_precomp",
        #               ]
        # for property in properties:
        #     if hasattr(self, property):
        #         setattr(cloned, property, getattr(self, property).clone())

        parameters = ["_xyz",
                      "_features_dc",
                      "_features_rest",
                      "_scaling",
                      "_rotation",
                      "_opacity",
                      "_latent_feature",
                    #   "_label"
                      ]
        for parameter in parameters:
            setattr(cloned, parameter, getattr(self, parameter).clone())

        return cloned

    # def set_fwd_transform(self, T_fwd, mask):
    #     # if not hasattr(self, 'fwd_transform'):
    #     #     self.fwd_transform = torch.zeros((self._xyz.shape[0], 4, 4), device=T_fwd.device)
    #     if mask is not None:
    #         self.fwd_transform[mask] = T_fwd
    #     else:
    #         self.fwd_transform = T_fwd
    
    # def set_rotation_precomp(self, rotation_precomp, mask):
    #     # if not hasattr(self, 'rotation_precomp'):
    #     #     self.rotation_precomp = torch.zeros((self._xyz.shape[0], 3, 3), device=rotation_precomp.device)
    #     if mask is not None:
    #         self.rotation_precomp[mask] = rotation_precomp
    #     else:
    #         self.rotation_precomp = rotation_precomp

    def set_rotation_precomp(self, rotation_precomp):

            self.rotation_precomp = rotation_precomp

    def color_by_opacity(self):
        cloned = self.clone()
        cloned._features_dc = self.get_opacity.unsqueeze(-1).expand(-1,-1,3)
        cloned._features_rest = torch.zeros_like(cloned._features_rest)
        return cloned

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._latent_feature,
            # self._label,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._latent_feature,
        # self._label,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom

    @property
    def get_scaling(self):
        return self._scaling
    
    @property
    def get_rotation(self):
        return self._rotation
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self._opacity
    
    @property
    def get_latent_feature(self):
        return self._latent_feature # .squeeze(-1) 


    
    def get_covariance(self, scaling_modifier = 1):
        if hasattr(self, 'rotation_precomp'):
            return self.covariance_activation(self.get_scaling, scaling_modifier, self.rotation_precomp)
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    # def update_latent_feature(self, latent_feature, mask):
    #     """
    #     根据给定的 mask 更新 _latent_feature
    #     :param mask: bool 类型的 Tensor，表示需要更新的位置
    #     :param latent_feature: Tensor，表示用于更新的 latent feature，长度与 mask 中 True 的个数相同
    #     """
    #     if mask is not None:
    #         assert mask.shape[0] == self._latent_feature.shape[0], "Mask size mismatch with latent feature"
    #         assert latent_feature.shape[0] == mask.sum(), "Latent feature size mismatch with mask"
            
    #         new_latent_feature = self._latent_feature.clone().squeeze(-1)
    #         new_latent_feature[mask] = latent_feature
    #         self._latent_feature = new_latent_feature.unsqueeze(-1)
        
    #     else:
    #         self._latent_feature = latent_feature.unsqueeze(-1)
    
    # def update_latent_feature(self, latent_feature):
    #     self._latent_feature = latent_feature.unsqueeze(-1)

    # def update_xyz(self, xyz, mask):
    #     if mask is not None:
    #         assert mask.shape[0] == self._xyz.shape[0], "Mask size mismatch with xyz"
    #         assert xyz.shape[0] == mask.sum(), "xyz size mismatch with mask"
            
    #         new_xyz = self._xyz.clone()
    #         new_xyz[mask] = xyz
    #         self._xyz = new_xyz
    #     else:
    #         self._xyz = xyz


    def update_properties(self, dict):
        self._xyz = dict['xyz']
        self._features_dc = dict['features_dc']
        self._features_rest = dict['features_rest']
        self._scaling = dict['scaling']
        self._rotation = dict['rotation']
        self._opacity = dict['opacity']
        self._latent_feature = dict['latent_feature']

    def oneupSHdegree(self):
        if not self.use_sh:
            return
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def get_opacity_loss(self):
        # opacity classification loss
        opacity = self.get_opacity
        eps = 1e-6
        loss_opacity_cls = -(opacity * torch.log(opacity + eps) + (1 - opacity) * torch.log(1 - opacity + eps)).mean()
        return {'opacity': loss_opacity_cls}

    def init_points(self, points_dict, spatial_lr_scale=1.):
        self.spatial_lr_scale = spatial_lr_scale
        points = points_dict['init_positions']
        n_points = points.shape[0]

        if self.use_sh:
            features = torch.zeros((n_points, 3, (self.max_sh_degree + 1) ** 2), device=self.device)
        else:
            features = torch.zeros((n_points, 1, self.feature_dim), device=self.device)

        # labels_body = torch.zeros((points_body.shape[0], 1), device=self.device)  # 身体部分为 0
        # labels_garment = torch.ones((points_garment.shape[0], 1), device=self.device)  # 衣物部分为 1
        # labels = torch.cat((labels_body, labels_garment))  # 合并

        print("Number of points at initialisation: ", n_points)

        # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        # rots = torch.zeros((n_points, 4), device=self.device)
        # rots[:, 0] = 1

        self._xyz = points  # 3D点坐标
        self._features_dc = features[:, :, 0:1].transpose(1, 2).contiguous() # 基础特征
        self._features_rest = features[:, :, 1:].transpose(1, 2).contiguous()  # 附加特征
        self._scaling =  points_dict['init_scalings'] # 缩放参数
        self._rotation = points_dict['init_rotations']   # 旋转参数
        self._opacity = torch.ones((n_points, 1), device=features.device)  # 不透明度参数
        self._latent_feature = torch.zeros((n_points, self.latent_feature_dim*3, 1), device=features.device)  # 潜在特征
        self.max_radii2D = torch.zeros((n_points), device=features.device)  # 2D最大半径
        # self._label = labels  # 标签


        # self.rotation_precomp = torch.zeros((n_points, 3, 3), device=self.device)
        # self.fwd_transform = torch.zeros((n_points, 4, 4), device=self.device)


    def training_setup(self, densification_args):
        # self.percent_dense = densification_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # Add semantic features
        for i in range(self._latent_feature.shape[1]*self._latent_feature.shape[2]):  
            l.append('latent_{}'.format(i))
        # l.append('label')
        return l

    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.cpu().numpy()
        scale = self._scaling.cpu().numpy()
        rotation = self._rotation.cpu().numpy()
        latent_feature = self._latent_feature.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy() 
        # labels = self._label.cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, latent_feature), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # def reset_opacity(self):
    #     opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
    #     optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
    #     self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        # labels = np.asarray(plydata.elements[0]["label"])[..., np.newaxis]

        count = sum(1 for name in plydata.elements[0].data.dtype.names if name.startswith("latent_"))
        latent_feature = np.stack([np.asarray(plydata.elements[0][f"latent_{i}"]) for i in range(count)], axis=1) 
        latent_feature = np.expand_dims(latent_feature, axis=-1) 


        if self.use_sh:
            features_dc = np.zeros((xyz.shape[0], 3, 1))
            features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
            features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
            
            extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
            assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        else:
            features_dc = np.zeros((xyz.shape[0], 1, 1))
            features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            features_extra = np.zeros((xyz.shape[0], 2, 1))
            features_extra[:, 0, 0] = np.asarray(plydata.elements[0]["f_rest_0"])
            features_extra[:, 1, 0] = np.asarray(plydata.elements[0]["f_rest_1"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = torch.tensor(xyz, dtype=torch.float, device=self.device)
        self._features_dc = torch.tensor(features_dc, dtype=torch.float, device=self.device).transpose(1, 2).contiguous()
        self._features_rest = torch.tensor(features_extra, dtype=torch.float, device=self.device).transpose(1, 2).contiguous()
        self._opacity = torch.tensor(opacities, dtype=torch.float, device=self.device)
        self._scaling = torch.tensor(scales, dtype=torch.float, device=self.device)
        self._rotation = torch.tensor(rots, dtype=torch.float, device=self.device)
        self._latent_feature = torch.tensor(latent_feature, dtype=torch.float, device=self.device).transpose(1, 2).contiguous()
        # self._label = torch.tensor(labels, dtype=torch.float, device=self.device)

        self.active_sh_degree = self.max_sh_degree


    def prune_points(self, mask):
        """移除指定 mask 位置的点，保留未被掩码的点"""
        valid_points_mask = ~mask

        self._xyz = self._xyz[valid_points_mask]
        self._features_dc = self._features_dc[valid_points_mask]
        self._features_rest = self._features_rest[valid_points_mask]
        self._opacity = self._opacity[valid_points_mask]
        self._scaling = self._scaling[valid_points_mask]
        self._rotation = self._rotation[valid_points_mask]
        self._latent_feature = self._latent_feature[valid_points_mask]
        # self._label = self._label[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]


    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_latent_feature):
        """合并新生成的点，并重新初始化梯度相关变量"""
        self._xyz = torch.cat((self._xyz, new_xyz), dim=0)
        self._features_dc = torch.cat((self._features_dc, new_features_dc), dim=0)
        self._features_rest = torch.cat((self._features_rest, new_features_rest), dim=0)
        self._opacity = torch.cat((self._opacity, new_opacities), dim=0)
        self._scaling = torch.cat((self._scaling, new_scaling), dim=0)
        self._rotation = torch.cat((self._rotation, new_rotation), dim=0)
        self._latent_feature = torch.cat((self._latent_feature, new_latent_feature), dim=0)
        # self._label = torch.cat((self._label, new_label), dim=0)

        num_points = self._xyz.shape[0]
        self.xyz_gradient_accum = torch.zeros((num_points, 1), device=self.device)
        self.denom = torch.zeros((num_points, 1), device=self.device)
        self.max_radii2D = torch.zeros((num_points,), device=self.device)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N)
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_latent_feature = self._latent_feature[selected_pts_mask].repeat(N,1,1)
        # new_label = self._label[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_latent_feature)


    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_latent_feature = self._latent_feature[selected_pts_mask]
        # new_label = self._label[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_latent_feature)

    def densify_and_prune(self, opt, cameras_extent, max_screen_size):
        extent = cameras_extent

        max_grad = opt.densify_grad_threshold
        min_opacity = opt.opacity_threshold

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1