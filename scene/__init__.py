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

import os
import torch
from models import GaussianConverter
from scene.gaussian_model_new import GaussianModel
from dataset import load_dataset

from utils.general_utils import rotation_matrix_to_quaternion
from simple_knn._C import distCUDA2


class Scene:

    gaussians : GaussianModel

    def __init__(self, cfg, gaussians : GaussianModel, save_dir : str):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.cfg = cfg

        self.save_dir = save_dir
        self.gaussians = gaussians

        self.train_dataset = load_dataset(cfg.dataset, split='train')
        self.metadata = self.train_dataset.metadata
        if cfg.mode == 'train':
            self.test_dataset = load_dataset(cfg.dataset, split='val')
        elif cfg.mode == 'test':
            self.test_dataset = load_dataset(cfg.dataset, split='test')
        elif cfg.mode == 'predict':
            self.test_dataset = load_dataset(cfg.dataset, split='predict')
        else:
            raise ValueError

        self.cameras_extent = self.metadata['cameras_extent']

        gaussians_body_init = self.init_gaussians()
        self.gaussians.init_points(points_dict = gaussians_body_init, spatial_lr_scale=self.cameras_extent)

        # self.gaussians.create_from_pcd(self.test_dataset.readPointCloud(), spatial_lr_scale=self.cameras_extent)

        self.converter = GaussianConverter(cfg, self.metadata).cuda()

    def train(self):
        self.converter.train()

    def eval(self):
        self.converter.eval()

    def optimize(self, iteration):
        # gaussians_delay = self.cfg.model.gaussian.get('delay', 0)
        # if iteration >= gaussians_delay:
        #     self.gaussians.optimizer.step()
        # self.gaussians.optimizer.zero_grad(set_to_none=True)
        self.converter.optimize()
    
    def init_gaussians(self):

        sample_ratio=self.cfg.get('sample_ratio', 0.5)

        verts = self.metadata['cano_verts']
        faces = self.metadata['faces']
        
        # faces = torch.from_numpy(faces).to(verts.device)  # (F, 3)

        v0 = verts[faces[:, 0]]  # 第一个顶点
        v1 = verts[faces[:, 1]]  # 第二个顶点
        v2 = verts[faces[:, 2]]  # 第三个顶点
        
        # 计算每个面的法向量
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # 叉积得到法向量
        face_normals = face_normals / face_normals.norm(dim=1, keepdim=True)  # 归一化法向量

        # **添加随机掩码减少采样点**
        num_faces = len(faces)
        mask = torch.rand(num_faces, device=verts.device) < sample_ratio  # 生成 bool 掩码
        selected_faces = faces[mask]  # 仅保留选中的面
        v0, v1, v2 = v0[mask], v1[mask], v2[mask]  # 过滤对应顶点
        face_normals = face_normals[mask]  # 过滤法向量


        # 生成随机重心坐标 (r1, r2)
        num_samples_per_face = 1
        r1 = torch.rand(len(selected_faces), num_samples_per_face, dtype=torch.float32, device=verts.device)  # 随机数 1
        r2 = torch.rand(len(selected_faces), num_samples_per_face, dtype=torch.float32, device=verts.device)  # 随机数 2

        # 计算重心坐标 u, v, w
        u = 1 - torch.sqrt(r1)  # 重心坐标 u
        v = torch.sqrt(r1) * (1 - r2)  # 重心坐标 v
        w = torch.sqrt(r1) * r2  # 重心坐标 w

        # 将重心坐标转换为 3D 点
        points = (u[:, :, None] * v0[:, None, :] + 
                v[:, :, None] * v1[:, None, :] + 
                w[:, :, None] * v2[:, None, :])  # (M, num_samples_per_face, 3)
        
        points = points.squeeze()  # (M, 3)

        # 沿着法向量方向增加 1 cm 的偏移
        offset_distance = 0.01  # 1 cm 的偏移量
        offset_points = points + face_normals * offset_distance  # (M, 3)

        # 生成初始旋转矩阵
        # Compute tangent and bitangent vectors
        tangent = (v1 - v0)
        tangent = tangent / tangent.norm(dim=1, keepdim=True)  # Normalize

        bitangent = torch.cross(face_normals, tangent, dim=1)  # Compute bitangent
        bitangent = bitangent / bitangent.norm(dim=1, keepdim=True)  # Normalize

        # Construct rotation matrix as (tangent, bitangent, normal)
        init_rotations = torch.stack([tangent, bitangent, face_normals], dim=-1)  # (M, 3, 3)

        init_rotations = rotation_matrix_to_quaternion(init_rotations)  # (M, 4)

        # 生成初始scaling
        dist2 = torch.clamp_min(distCUDA2(points), 0.0000001)
        init_scalings = torch.sqrt(dist2)[...,None].repeat(1, 3)
        # init_scalings[:, -1] = 0.1  # Fix last dimension


        points_dict_body = {
            'init_positions': points,
            'init_rotations': init_rotations,  
            'init_scalings': init_scalings
        }

        return points_dict_body
    

    def convert_gaussians(self, viewpoint_camera, iteration, compute_loss=True):
        return self.converter(self.gaussians, viewpoint_camera, iteration, compute_loss)

    def get_skinning_loss(self):
        loss_reg = self.converter.deformer.rigid.regularization()
        loss_skinning = loss_reg.get('loss_skinning', torch.tensor(0.).cuda())
        return loss_skinning

    def save(self, iteration):
        point_cloud_path = os.path.join(self.save_dir, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def save_checkpoint(self, iteration):
        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        torch.save((self.gaussians.capture(),
                    self.converter.state_dict(),
                    self.converter.optimizer.state_dict(),
                    self.converter.scheduler.state_dict(),
                    iteration), self.save_dir + "/ckpt" + str(iteration) + ".pth")

    def load_checkpoint(self, path):
        (gaussian_params, converter_sd, converter_opt_sd, converter_scd_sd, first_iter) = torch.load(path)
        self.gaussians.restore(gaussian_params, self.cfg.opt)
        self.converter.load_state_dict(converter_sd)
        # self.converter.optimizer.load_state_dict(converter_opt_sd)
        # self.converter.scheduler.load_state_dict(converter_scd_sd)