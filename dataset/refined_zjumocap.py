import os
import sys
import glob
import cv2
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from utils.dataset_utils import get_02v_bone_transforms, fetchPly, storePly, AABB
from scene.cameras import Camera


import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
import trimesh

from smpl_loader.body_model import SMPL

class RefinedZJUMoCapDataset(Dataset):
    def __init__(self, cfg, split='train'):
        super().__init__()
        self.cfg = cfg
        self.split = split

        self.root_dir = cfg.root_dir
        self.subject = cfg.subject # subject name e.g. my_377
        self.train_frame_interval = cfg.train_frame_interval # 5
        self.train_frame_num = cfg.train_frame_num # 100
        self.train_cams = cfg.train_views
        self.val_frame_interval = cfg.val_frame_interval # 30
        self.val_frame_num = cfg.val_frame_num # 17
        self.val_cams = cfg.val_views
        self.white_bg = cfg.white_background
        self.H, self.W = 1024, 1024 # hardcoded original size
        self.h, self.w = cfg.img_hw

        ann_file = os.path.join(self.root_dir, self.subject, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cameras = annots['cams']
        num_cams = len(self.cameras['K'])

        self.body_model = SMPL(model_path='data/body_models/smpl/neutral/model.pkl', gender='neutral').cuda()
        self.get_smpl_metadata()


        if split == 'train':
            cam_views = self.train_cams
            frame_interval = self.train_frame_interval
            frame_num = self.train_frame_num
        elif split == 'val':
            cam_views = self.val_cams
            frame_interval = self.train_frame_interval
            frame_num = self.train_frame_num
        elif split == 'test':
            cam_views = self.cfg.test_views[self.cfg.test_mode]
            frame_interval = self.cfg.test_frame_interval[self.cfg.test_mode]
            frame_num = self.cfg.test_frame_num[self.cfg.test_mode]
        # elif split == 'predict':
        #     cam_names = self.cfg.predict_views
        #     frames = self.cfg.predict_frames
        else:
            raise ValueError

        if len(cam_views) == 0:
            cam_views = list(range(num_cams))
        # else:
        #     cam_names = [int(cams) - 1 for cams in cam_names]


        self.subject_dir = os.path.join(self.root_dir, self.subject)


        ims = np.array([
            np.array(ims_data['ims'])[cam_views]
            for ims_data in annots['ims'][:frame_num * frame_interval][::frame_interval]
        ]) # sample 1 frame every 5 frames and collect 100 frames for training

        cam_idxs = np.array([
            np.arange(len(ims_data['ims']))[cam_views]
            for ims_data in annots['ims'][:frame_num * frame_interval][::frame_interval]
        ])

        # if 'CoreView_313' in path or 'CoreView_315' in path:
        #     for i in range(ims.shape[0]):
        #         ims[i] = [x.split('/')[0] + '/' + x.split('/')[1].split('_')[4] + '.jpg' for x in ims[i]]

        # TODO: update selected frames to metadata


        self.data = []

        for frame_index in range(frame_num):
            for view_index in range(len(cam_views)):
            

                # collect image and mask paths
                image_path = os.path.join(self.subject_dir, ims[frame_index][view_index].replace('\\', '/'))
                mask_path = image_path.replace('images', 'mask').replace('jpg', 'png')

                # feature map
                semantic_feature_path = image_path.replace('images', 'feature_maps').replace(".jpg", "_fmap.pt")

                # collect smpl data for each sample frame
                if view_index == 0:
                    smpl_data, cano_smpl_data = self.get_smpl_data(image_path)

                # collect camera parameters K, D, R, T
                cam_idx = cam_idxs[frame_index][view_index] # or use view_index ?
                K = np.array(annots['cams']['K'][cam_idx]) # float32?
                D = np.array(annots['cams']['D'][cam_idx])
                R = np.array(annots['cams']['R'][cam_idx])
                T = np.array(annots['cams']['T'][cam_idx]).reshape(3, 1) / 1000.

                cam_params = {'K': K, 'D': D, 'R': R, 'T': T}

                self.data.append({
                        'cam_view': view_index,
                        'frame_idx': frame_index,
                        'image_path': image_path,
                        'mask_path': mask_path,
                        'feature_path': semantic_feature_path,
                        'cam_idx': cam_idx,
                        'cam_params': cam_params,
                        'smpl_data': smpl_data,
                        'cano_smpl_data': cano_smpl_data,
                    })



        self.preload = cfg.get('preload', True)
        if self.preload:
            self.cameras = [self.getitem(idx) for idx in range(len(self))]


    def get_smpl_metadata(self):

        self.smpl_metadata = {
            'v_template': self.body_model.v_template,
            'faces': self.body_model.faces_tensor,
            'posedirs': self.body_model.posedirs,
            'J_regressor': self.body_model.J_regressor,
            'skinning_weights': self.body_model.lbs_weights,
            'ktree_parents': self.body_model.parents,
            'cameras_extent': 3.469298553466797, # hardcoded, used to scale the threshold for scaling/image-space gradient
            # 'frame_dict': ims,
        }

    def get_cano_smpl_verts(self, minimal_shape):
        '''
            Compute star-posed SMPL body vertices.
            To get a consistent canonical space,
            we do not add pose blend shape
        '''

        # Break symmetry if given in float16:
        if minimal_shape.dtype == np.float16:
            minimal_shape = minimal_shape.astype(np.float32)
            minimal_shape += 1e-4 * np.random.randn(*minimal_shape.shape)
        else:
            minimal_shape = minimal_shape.astype(np.float32)

        # Minimally clothed shape
        J_regressor = self.body_model.J_regressor
        Jtr = np.dot(J_regressor, minimal_shape)

        skinning_weights = self.body_model.lbs_weights
        # Get bone transformations that transform a SMPL A-pose mesh
        # to a star-shaped A-pose (i.e. Vitruvian A-pose)
        transforms_mat_02v = get_02v_bone_transforms(Jtr)

        T = np.matmul(skinning_weights, transforms_mat_02v.reshape([-1, 16])).reshape([-1, 4, 4])
        vertices = np.matmul(T[:, :3, :3], minimal_shape[..., np.newaxis]).squeeze(-1) + T[:, :3, -1]

        coord_max = np.max(vertices, axis=0)
        coord_min = np.min(vertices, axis=0)
        padding_ratio = self.cfg.padding
        padding_ratio = np.array(padding_ratio, dtype=np.float)
        padding = (coord_max - coord_min) * padding_ratio
        coord_max += padding
        coord_min -= padding

        cano_mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=self.body_model.faces_tensor)

        return {
            'cano_verts': vertices.astype(np.float32),
            'Jtr': Jtr, # joint position under minimal shape in canonical space
            'transforms_mat_02v': transforms_mat_02v,
            'cano_mesh': cano_mesh,
            'coord_min': coord_min,
            'coord_max': coord_max,
            'aabb': AABB(coord_max, coord_min),
        } # canonical SMPL vertices without pose, but consider body shape

    def get_smpl_data(self, image_path):
        frame_idx = int(os.path.basename(image_path)[:-4])
        smpl_param_path = os.path.join(self.subject_dir, "smpl_params", '{}.npy'.format(frame_idx))
        smpl_params = np.load(smpl_param_path, allow_pickle=True).item()

        root_orient = Rotation.from_rotvec(np.array(smpl_params['Rh']).reshape([-1])).as_matrix() # (3, 3)
        trans = np.array(smpl_params['Th']).reshape([3, 1]) # (3, 1)

        betas = np.array(smpl_params['shapes'], dtype=np.float32) # (1, 10)
        poses = np.array(smpl_params['poses'], dtype=np.float32) # (1, 72)
        body_poses = poses[:, 3:].copy()

        body_poses_torch = torch.from_numpy(body_poses).cuda()
        betas_torch = torch.from_numpy(betas).cuda()

        new_root_orient = Rotation.from_matrix(root_orient).as_rotvec().reshape([1, 3]).astype(np.float32) # (1, 3)
        new_trans = trans.reshape([1, 3]).astype(np.float32)

        new_root_orient_torch = torch.from_numpy(new_root_orient).cuda()
        new_trans_torch = torch.from_numpy(new_trans).cuda()

        # Get shape vertices
        body = self.body_model(betas=betas_torch)
        minimal_shape = body.v.detach().cpu().numpy()[0]
        # fit star-posed SMPL body based on minimal shape
        cano_smpl_data = self.get_cano_smpl_verts(minimal_shape)

        # Get bone transforms
        body = self.body_model(global_orient=new_root_orient_torch, body_pose=body_poses_torch, betas=betas_torch, transl=new_trans_torch)

        body_wo_global_orient = self.body_model(body_pose=body_poses_torch, betas=betas_torch, transl=new_trans_torch, return_verts=True)[0].detach().cpu().numpy()



        vertices = body.vertices.detach().cpu().numpy()[0]
        vertices_wo_global_orient = body_wo_global_orient.vertices.detach().cpu().numpy()[0]
        new_trans = new_trans + (vertices_wo_global_orient - vertices).mean(0, keepdims=True)
        new_trans_torch = torch.from_numpy(new_trans).cuda()

        # body = self.body_model(global_orient=new_root_orient_torch, body_pose=body_poses_torch, betas=betas_torch, transl=new_trans_torch)

        # # Visualize SMPL mesh
        # import trimesh
        # smpl_out_dir = os.path.join(self.subject_dir, 'smpl_meshes')
        # pose_mesh = trimesh.Trimesh(vertices=vertices, faces=self.body_model.faces_tensor)
        # out_filename = os.path.join(smpl_out_dir, '{:06d}.ply'.format(frame_idx))
        # pose_mesh.export(out_filename)

        # no output for now
        # transforms_mat = body.transforms_mat.detach().cpu().numpy()

        
        smpl_data = {'frame': frame_idx,
                     'minimal_shape': minimal_shape, 
                     'betas': betas, 
                    #  'transforms_mat': transforms_mat[0], #
                     'trans': new_trans[0], 
                     'root_orient': new_root_orient[0],
                     'body_poses': body_poses[0],
                     }
        
        return smpl_data, cano_smpl_data  

                

    def __len__(self):
        return len(self.data)

    def getitem(self, idx, data_dict=None):
        if data_dict is None:
            data_dict = self.data[idx]

        cam_view = data_dict['cam_view']
        frame_idx = data_dict['frame_idx']
        img_file = data_dict['image_path']
        mask_file = data_dict['mask_path']
        feature_file = data_dict['feature_path']
        cam_params = data_dict['cam_params']
        smpl_data = data_dict['smpl_data']
        cano_smpl_data = data_dict['cano_smpl_data']

        K = cam_params['K']
        D = cam_params['D'].ravel()
        R = cam_params['R']
        T = cam_params['T']

        # note that in ZJUMoCap the camera center does not align perfectly
        # here we try to offset it by modifying the extrinsic...
        M = np.eye(3)
        M[0, 2] = (K[0, 2] - self.W / 2) / K[0, 0]
        M[1, 2] = (K[1, 2] - self.H / 2) / K[1, 1]
        K[0, 2] = self.W / 2
        K[1, 2] = self.H / 2
        R = M @ R
        T = M @ T

        R = np.transpose(R)
        T = T[:, 0]

        image = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        image = cv2.undistort(image, K, D, None)
        mask = cv2.undistort(mask, K, D, None)

        lanczos = self.cfg.get('lanczos', False)
        interpolation = cv2.INTER_LANCZOS4 if lanczos else cv2.INTER_LINEAR

        image = cv2.resize(image, (self.w, self.h), interpolation=interpolation)
        mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        mask = mask != 0
        image[~mask] = 255. if self.white_bg else 0.
        image = image / 255.

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        semantic_feature = torch.load(feature_file)

        # update camera parameters
        K[0, :] *= self.w / self.W
        K[1, :] *= self.h / self.H

        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovY = focal2fov(focal_length_y, self.h)
        FovX = focal2fov(focal_length_x, self.w)

        # Compute posed SMPL body
        minimal_shape = smpl_data['minimal_shape']

        trans = smpl_data['trans'].astype(np.float32)
        # transforms_mat = smpl_data['transforms_mat'].astype(np.float32)
        # Also get GT SMPL poses
        root_orient = smpl_data['root_orient'].astype(np.float32)
        body_poses = smpl_data['body_poses'].astype(np.float32)
        pose = np.concatenate([root_orient, body_poses], axis=-1)
        pose = Rotation.from_rotvec(pose.reshape([-1, 3]))
        
        # get relative rotation matrix
        pose_mat_full = pose.as_matrix()  # 24 x 3 x 3
        # pose_mat = pose_mat_full[1:, ...].copy()  # 23 x 3 x 3
        # pose_rot = np.concatenate([np.expand_dims(np.eye(3), axis=0), pose_mat], axis=0).reshape(
        #     [-1, 9])  # 24 x 9, root rotation is set to identity
        pose_rot = pose_mat_full.reshape([-1, 9])  # 24 x 9, including root rotation
        # # get absolute rotation 
        # # 构建层次结构矩阵
        # ktree_parents = self.smpl_metadata['ktree_parents']
        # num_joints = len(ktree_parents)
        # hierarchy_mat = torch.zeros(num_joints, num_joints, dtype=torch.float32)
        # for j in range(num_joints):
        #     parent = ktree_parents[j]
        #     if parent != -1:
        #         hierarchy_mat[j, parent] = 1

        # # 计算全局旋转矩阵
        # abs_rot_mats = torch.eye(3).unsqueeze(0).repeat(num_joints, 1, 1)  # (24, 3, 3)
        # for _ in range(num_joints):
        #     abs_rot_mats = torch.einsum('nij,njk->nik', hierarchy_mat @ abs_rot_mats, pose.as_matrix())
        # pose_rot = abs_rot_mats.reshape(-1, 9)


        # Minimally clothed shape
        Jtr = cano_smpl_data['Jtr']

        # canonical SMPL vertices without pose correction, to normalize joints
        center = np.mean(minimal_shape, axis=0)
        minimal_shape_centered = minimal_shape - center
        cano_max = minimal_shape_centered.max()
        cano_min = minimal_shape_centered.min()
        padding = (cano_max - cano_min) * 0.05

        # compute pose condition
        Jtr_norm = Jtr - center
        Jtr_norm = (Jtr_norm - cano_min + padding) / (cano_max - cano_min) / 1.1
        Jtr_norm -= 0.5
        Jtr_norm *= 2.

        # final bone transforms that transforms the canonical Vitruvian-pose mesh to the posed mesh
        # transforms_mat_02v = cano_smpl_data['transforms_mat_02v']
        # transforms_mat = transforms_mat @ np.linalg.inv(transforms_mat_02v)
        # transforms_mat = transforms_mat.astype(np.float32)
        # transforms_mat[:, :3, 3] += trans  # add global offset
        smpl_cano_verts = cano_smpl_data['cano_verts']
        smpl_cano_mesh = cano_smpl_data['cano_mesh']
        smpl_cano_verts_range = cano_smpl_data['aabb']

        return Camera(
            frame_id=frame_idx,
            cam_id=int(cam_view),
            K=K, R=R, T=T,
            FoVx=FovX,
            FoVy=FovY,
            image=image,
            mask=mask,
            semantic_feature=semantic_feature,
            gt_alpha_mask=None,
            image_name=f"c{int(cam_view):02d}_f{frame_idx if frame_idx >= 0 else -frame_idx - 1:06d}",
            data_device=self.cfg.data_device,
            # human params
            rots=torch.from_numpy(pose_rot).float().unsqueeze(0), # relative joints rotaiton
            Jtrs=torch.from_numpy(Jtr_norm).float().unsqueeze(0), # normalized joints position in canonical space considering shape
            trans=torch.from_numpy(trans).float().unsqueeze(0), # global translation
            smpl_cano_verts=smpl_cano_verts,
            smpl_cano_mesh=smpl_cano_mesh,
            smpl_cano_verts_range=smpl_cano_verts_range,
        )

    def __getitem__(self, idx):
        if self.preload:
            return self.cameras[idx]
        else:
            return self.getitem(idx)

    def readPointCloud(self,):
        if self.cfg.get('random_init', False):
            ply_path = os.path.join(self.root_dir, self.subject, 'random_pc.ply')

            aabb = self.metadata['aabb']
            coord_min = aabb.coord_min.unsqueeze(0).numpy()
            coord_max = aabb.coord_max.unsqueeze(0).numpy()
            n_points = 50_000

            xyz_norm = np.random.rand(n_points, 3)
            xyz = xyz_norm * coord_min + (1. - xyz_norm) * coord_max
            rgb = np.ones_like(xyz) * 255
            storePly(ply_path, xyz, rgb)

            pcd = fetchPly(ply_path)
        else:
            ply_path = os.path.join(self.root_dir, self.subject, 'cano_smpl.ply')
            try:
                pcd = fetchPly(ply_path)
            except:
                verts = self.metadata['smpl_cano_verts']
                faces = self.faces
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                n_points = 50_000

                xyz = mesh.sample(n_points)
                rgb = np.ones_like(xyz) * 255
                storePly(ply_path, xyz, rgb)

                pcd = fetchPly(ply_path)

        return pcd
