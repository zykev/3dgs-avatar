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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera:
    def __init__(self, camera=None, **kwargs):
        if camera is not None:
            self.data = camera.data.copy()
            return

        self.data = kwargs
        # self.data['trans'] = np.array([0.0, 0.0, 0.0])
        # self.data['scale'] = 1.0

        self.data['original_image'] = self.image.clamp(0.0, 1.0).to(self.data_device)
        self.data['image_width'] = self.original_image.shape[2]
        self.data['image_height'] = self.original_image.shape[1]
        self.data['original_mask'] = self.mask.float().to(self.data_device)
        if kwargs.get('semantic_feature') is not None:
            self.data['image_feature'] = kwargs.get('semantic_feature').to(self.data_device)
        if kwargs.get('seg_label') is not None:
            self.data['seg_label'] = kwargs.get('seg_label').to(self.data_device)
        if kwargs.get('depth') is not None:
            self.data['depth'] = kwargs.get('depth').to(self.data_device)
        if kwargs.get('normal') is not None:
            self.data['normal'] = kwargs.get('normal').to(self.data_device)

        # 获取相机参数
        # self.data['K'] = torch.tensor(kwargs.get('K')).to(self.data_device)
        # self.data['R'] = torch.tensor(kwargs.get('R')).to(self.data_device)
        # self.data['T'] = torch.tensor(kwargs.get('T')).to(self.data_device)


        self.data['zfar'] = 100.0
        self.data['znear'] = 0.01

        self.data['world_view_transform'] = torch.tensor(getWorld2View2(self.R, self.T)).transpose(0, 1).cuda()
        self.data['projection_matrix'] = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1).cuda()
        self.data['full_proj_transform'] = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.data['camera_center'] = self.world_view_transform.inverse()[3, :3]

        # self.data['world_view_transform'], self.data['camera_center'] = getWorld2View2(self.data['R'], self.data['T']) # 第三个参数 self.data['trans'].squeeze(0)
        # self.data['projection_matrix'] = getProjectionMatrix(znear=self.data['znear'], zfar=self.data['zfar'], fovX=self.data['FoVx'], fovY=self.data['FoVy']).transpose(0, 1).to(self.data_device)
        # self.data['full_proj_transform'] = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)




        self.data['rots'] = self.rots.to(self.data_device)
        # self.data['Jtrs'] = self.Jtrs.to(self.data_device)
        self.data['bone_transforms'] = self.bone_transforms.to(self.data_device)
        self.data['trans'] = self.trans.to(self.data_device)


    def __getattr__(self, item):
        return self.data[item]

    def update(self, **kwargs):
        self.data.update(kwargs)

    def copy(self):
        new_cam = Camera(camera=self)
        return new_cam

    def merge(self, cam):
        self.data['frame_id'] = cam.frame_id
        self.data['rots'] = cam.rots.detach()
        self.data['Jtrs'] = cam.Jtrs.detach()
        self.data['bone_transforms'] = cam.bone_transforms.detach()
