import os
import sys
import glob
import cv2
from utils.graphics_utils import focal2fov
import numpy as np
import json
from utils.dataset_utils import get_02v_bone_transforms, fetchPly, storePly, AABB
from scene.cameras import Camera


import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
import trimesh
from smpl_loader.body_model import SMPLLayer
class ZJUMoCapDataset(Dataset):
    def __init__(self, cfg, split='train'):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.device = cfg.device

        self.root_dir = cfg.root_dir

        self.subject = cfg.subject
        self.train_frames = cfg.train_frames # 5# 100
        self.train_cams = cfg.train_views

        self.val_frames = cfg.val_frames # 30
        self.val_cams = cfg.val_views

        self.white_bg = cfg.white_background
        self.H, self.W = 1024, 1024 # hardcoded original size
        self.h, self.w = cfg.img_hw


        annots = np.load(os.path.join(self.root_dir, self.subject, 'annots.npy'), allow_pickle=True).item()
        self.cameras = annots['cams']
        

        self.body_model = SMPLLayer(model_path='.datasets/body_models/smpl/neutral/model.pkl', gender='neutral')
        self.load_cano_smpl_flag = False


        if split == 'train':
            cam_views = self.train_cams
            frames = self.train_frames
        elif split == 'val':
            cam_views = self.val_cams
            frames = self.val_frames
        else:
            raise ValueError



        self.subject_dir = os.path.join(self.root_dir, self.subject)

        start_frame, end_frame, sampling_rate = frames


        frame_slice = slice(start_frame, end_frame, sampling_rate)
        frames = list(range(len(os.listdir(os.path.join(self.subject_dir, 'new_params')))))
        frames = frames[frame_slice]

        self.frame_dict = {
            frame: i for i, frame in enumerate(frames)
        }

        self.get_metadata()


        self.data = []
        
        for cam_idx, cam_name in enumerate(cam_views):
            cam_dir = os.path.join(self.subject_dir, cam_name)
            img_files = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))[frame_slice]
            mask_files = sorted(glob.glob(os.path.join(cam_dir, '*.png')))[frame_slice]

            for d_idx, f_idx in enumerate(frames):
                img_file = img_files[d_idx]
                mask_file = mask_files[d_idx]

                if cam_idx == 0:
                    smpl_data = self.get_smpl_data(img_file)
                
                # collect camera parameters K, D, R, T
                # cam_idx = cam_idxs[frame_index][view_index]
                K = np.array(annots['cams']['K'][cam_idx], dtype=np.float32)
                D = np.array(annots['cams']['D'][cam_idx], dtype=np.float32)
                R = np.array(annots['cams']['R'][cam_idx], dtype=np.float32)
                T = np.array(annots['cams']['T'][cam_idx], dtype=np.float32) / 1000.

                cam_params = {'K': K, 'D': D, 'R': R, 'T': T}

                self.data.append({
                    'cam_idx': cam_idx,
                    'cam_name': cam_name,
                    'data_idx': d_idx,
                    'frame_idx': f_idx,
                    'img_file': img_file,
                    'mask_file': mask_file,
                    'cam_params': cam_params,
                    'smpl_data': smpl_data,
                })



        self.preload = cfg.get('preload', True)
        if self.preload:
            self.cameras = [self.getitem(idx) for idx in range(len(self))]

    def get_metadata(self):

        self.metadata = {
            'v_template': self.body_model.v_template.to(self.device),
            'faces': self.body_model.faces_tensor.to(self.device),
            'shapedirs': self.body_model.shapedirs.to(self.device),
            'posedirs': self.body_model.posedirs.to(self.device),
            'J_regressor': self.body_model.J_regressor.to(self.device),
            'skinning_weights': self.body_model.lbs_weights.to(self.device),
            'kintree_parents': self.body_model.kintree_parents.to(self.device),
            'cameras_extent': 3.469298553466797, # hardcoded, used to scale the threshold for scaling/image-space gradient
            'frame_dict': self.frame_dict,
            'img_hw': (self.h, self.w),
        }

    
    def get_cano_smpl_verts(self, minimal_shape):
        '''
            Compute star-posed SMPL body vertices.
            To get a consistent canonical space,
            we do not add pose blend shape
        '''

        # Break symmetry if given in float16
        if minimal_shape.dtype == torch.float16:
            minimal_shape = minimal_shape.to(torch.float32)
            minimal_shape += 1e-4 * torch.randn_like(minimal_shape)
        else:
            minimal_shape = minimal_shape.to(torch.float32)

        # Minimally clothed shape (minimal body shape)
        J_regressor = self.body_model.J_regressor  # Assuming this is a torch tensor
        Jtr = torch.matmul(J_regressor, minimal_shape)

        # Get bone transformations that transform a SMPL A-pose mesh
        skinning_weights = self.body_model.lbs_weights  # Assuming this is a torch tensor
        
        transforms_mat_02v = get_02v_bone_transforms(Jtr)
        T = torch.matmul(skinning_weights, transforms_mat_02v.reshape([-1, 16])).reshape([-1, 4, 4])
        vertices = torch.matmul(T[:, :3, :3], minimal_shape[..., None]).squeeze(-1) + T[:, :3, -1]

        # Compute bounding box and padding
        # vertices = minimal_shape
        coord_max = torch.max(vertices, dim=0)[0]
        coord_min = torch.min(vertices, dim=0)[0]
        # padding_ratio = torch.tensor(self.cfg.padding, dtype=torch.float32)
        # padding = (coord_max - coord_min) * padding_ratio
        padding = self.cfg.padding
        coord_max += padding
        coord_min -= padding

        # Create the canonical mesh
        cano_mesh = trimesh.Trimesh(vertices=vertices.numpy(), 
                                    faces=self.body_model.faces_tensor.numpy())

        # Center minimal shape and normalize
        center = torch.mean(minimal_shape, dim=0)
        minimal_shape_centered = minimal_shape - center
        cano_max = torch.max(minimal_shape_centered, dim=0)[0]
        cano_min = torch.min(minimal_shape_centered, dim=0)[0]
        padding = (cano_max - cano_min) * 0.05

        # Compute pose condition (normalize joint position)
        Jtr_norm = Jtr - center
        Jtr_norm = (Jtr_norm - cano_min + padding) / (cano_max - cano_min) / 1.1
        Jtr_norm -= 0.5
        Jtr_norm *= 2.0

        self.metadata.update({
            'cano_verts': vertices.to(self.device),
            'Jtr': Jtr.to(self.device),  # joint position under minimal shape in canonical space
            'Jtr_norm': Jtr_norm.unsqueeze(0).to(self.device),  # normalized joint position under minimal shape in canonical space
            'cano_mesh': cano_mesh,  # You can handle meshes with torch3d later if needed
            'aabb': AABB(coord_max.to(self.device), coord_min.to(self.device)),  # Assuming AABB is a class that accepts torch tensors
            'transforms_mat_02v': transforms_mat_02v,
        })


    def get_smpl_data(self, image_path):
        frame_idx = int(os.path.basename(image_path)[:-4])
        smpl_param_path = os.path.join(self.subject_dir, "new_params", '{}.npy'.format(frame_idx))
        smpl_params = np.load(smpl_param_path, allow_pickle=True).item()

        betas = np.array(smpl_params['shapes'], dtype=np.float32) # (1, 10)
        betas = torch.from_numpy(betas)

        root_orient = np.array(smpl_params['Rh'], dtype=np.float32).reshape([-1, 3])
        root_orient = torch.from_numpy(root_orient)

        poses = np.array(smpl_params['poses'], dtype=np.float32) # (1, 72)
        body_poses = poses[:, 3:].copy()
        body_poses = torch.from_numpy(body_poses)

        # get relative rotation matrix
        poses_all = torch.cat([root_orient, body_poses], dim=-1)
        poses_all = Rotation.from_rotvec(poses_all.reshape(-1, 3).numpy()).as_matrix() # 24 x 3 x 3
        pose_mat = poses_all[1:, ...].copy()  # 23 x 3 x 3
        pose_mat = np.concatenate([np.expand_dims(np.eye(3), axis=0), pose_mat], axis=0).reshape(
            [-1, 9])  # 24 x 9, root rotation is set to identity
        poses_all = torch.from_numpy(pose_mat).to(torch.float32)  # 24 x 9, not including root rotation
        
        # poses_all = torch.from_numpy(poses_all.reshape([-1, 9])).to(torch.float32)  # 24 x 9, including root rotation

        trans = np.array(smpl_params['Th'], dtype=np.float32).reshape([1, 3])
        trans = torch.from_numpy(trans)

        # Get shape vertices
        if not self.load_cano_smpl_flag:
            body = self.body_model(betas=betas)
            minimal_shape = body.vertices.detach()[0]
            self.get_cano_smpl_verts(minimal_shape)
            self.load_cano_smpl_flag = True

        # Get corrected translation
        body = self.body_model(global_orient=root_orient, 
                               body_pose=body_poses, 
                               betas=betas, 
                               transl=trans)

        body_wo_global_orient = self.body_model(body_pose=body_poses, betas=betas, transl=trans)



        vertices = body.vertices[0]
        vertices_wo_global_orient = body_wo_global_orient.vertices[0]
        new_trans = trans + (vertices_wo_global_orient - vertices).mean(0, keepdims=True)

        # get transformation matrix
        body = self.body_model(global_orient=root_orient, body_pose=body_poses, betas=betas, transl=new_trans)
        transforms_mat = body.transforms_mat[0]

        # final bone transforms that transforms the canonical Vitruvian-pose mesh to the posed mesh
        transforms_mat_02v_inv = torch.linalg.inv(self.metadata['transforms_mat_02v'])
        transforms_mat = torch.matmul(transforms_mat, transforms_mat_02v_inv)
        transforms_mat[:, :3, 3] += new_trans  # add global offset

        # Compute bounding box and padding
        coord_max = torch.max(body.vertices[0], dim=0)[0]
        coord_min = torch.min(body.vertices[0], dim=0)[0]
        padding = self.cfg.padding
        coord_max += padding
        coord_min -= padding
        
        smpl_data = {'frame': frame_idx,
                    #  'minimal_shape': minimal_shape,
                     'betas': betas, 
                     'transforms_mat': transforms_mat,
                     'trans': new_trans.unsqueeze(0), 
                     'poses': poses_all.unsqueeze(0),
                     'coord_bound': (coord_min, coord_max)
                     }
        
        return smpl_data

    def __len__(self):
        return len(self.data)

    def getitem(self, idx, data_dict=None):
        if data_dict is None:
            data_dict = self.data[idx]
        cam_idx = data_dict['cam_idx']
        cam_name = data_dict['cam_name']
        data_idx = data_dict['data_idx']
        frame_idx = data_dict['frame_idx']
        img_file = data_dict['img_file']
        mask_file = data_dict['mask_file']
        cam_params = data_dict['cam_params']
        smpl_data = data_dict['smpl_data']

        K = cam_params['K'].copy()
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
        T = M @ T.reshape(3, 1)

        R = np.transpose(R)
        # T = T[:, 0]

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

        # update camera parameters
        K[0, :] *= self.w / self.W
        K[1, :] *= self.h / self.H

        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovY = focal2fov(focal_length_y, self.h)
        FovX = focal2fov(focal_length_x, self.w)

        # get smpl data
        betas = smpl_data['betas']
        trans = smpl_data['trans']
        rots = smpl_data['poses']
        transforms_mat = smpl_data['transforms_mat']


        # Also get GT SMPL poses


        return Camera(
            frame_id=frame_idx,
            cam_id=int(cam_name),
            K=K, R=R, T=T,
            FoVx=FovX,
            FoVy=FovY,
            image=image,
            mask=mask,
            gt_alpha_mask=None,
            image_name=f"c{int(cam_name):02d}_f{frame_idx if frame_idx >= 0 else -frame_idx - 1:06d}",
            data_device=self.cfg.device,
            # human params
            betas=betas,
            rots=rots,
            bone_transforms=transforms_mat,
            trans=trans,
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
                verts = self.metadata['cano_verts']
                faces = self.faces
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                n_points = 50_000

                xyz = mesh.sample(n_points)
                rgb = np.ones_like(xyz) * 255
                storePly(ply_path, xyz, rgb)

                pcd = fetchPly(ply_path)

        return pcd
