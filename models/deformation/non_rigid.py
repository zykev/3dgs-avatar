import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d.transforms as tf
import pytorch3d.ops as ops

from models.network_utils import (HierarchicalPoseEncoder,
                                  VanillaCondMLP)
from utils.general_utils import rotation_matrix_to_quaternion

from models.deformation.pesudo_lbs import pesudo_lbs

from kmeans.soft_kmeans import SoftKMeans



class GarmentPoseDecoder(nn.Module):
    # predict garment joint rotation and location wrt smpl root given absolute smpl body joint rotation and location
    def __init__(self, cfg, metadata, pose_feature_dim, latent_feature_dim, middle_dim=64, **kwargs):
        super().__init__()
        in_dim = pose_feature_dim + latent_feature_dim
        middle_dim = middle_dim

        self.skinning_weights = metadata['skinning_weights']

        self.body_pose_proj = nn.Sequential(nn.Linear(in_dim, middle_dim), torch.nn.GELU())

        # add latent code (frame embedding)
        self.latent_dim = cfg.get('latent_dim', 0)
        d_cond = middle_dim
        if self.latent_dim > 0:
            d_cond += self.latent_dim
            self.frame_dict = metadata['frame_dict']
            self.latent_layer = nn.Embedding(len(self.frame_dict), self.latent_dim)

        self.position_proj = VanillaCondMLP(d_in=3, d_cond=d_cond, d_out=middle_dim)
    
        self.rot_proj = nn.Sequential(nn.Linear(middle_dim, 4), F.normalize(dim=-1)) # rotation (quaternion representation) 
        self.trans_proj = nn.Linear(middle_dim, 3)

    def query_lbs_weights(self, xyz, smpl_verts):
        # find the skinning weights of nearest smpl vertex given xyz
        knn_ret = ops.knn_points(xyz.unsqueeze(0), smpl_verts.unsqueeze(0))
        p_idx = knn_ret.idx.squeeze()
        pts_W = self.skinning_weights[p_idx, :]

        return pts_W
    
    def forward(self, body_pose_feature, garment_joint_feature, garment_joint_xyz, camera):
        smpl_cano_verts = camera.smpl_cano_verts
        aabb = camera.smpl_cano_verts_range

        # query skinning weights of input garment vertices
        pts_W = self.query_lbs_weights(garment_joint_xyz, smpl_cano_verts)
        garment_body_feature = torch.matmul(pts_W, body_pose_feature)
        garment_feature = torch.cat([garment_body_feature, garment_joint_feature], dim=-1)
        garment_feature = self.body_pose_proj(garment_feature)

        # add time embedding to concatenate with garment feature
        if self.time_embed_dim > 0:
            frame_idx = camera.frame_id
            if frame_idx not in self.frame_dict:
                latent_idx = len(self.frame_dict) - 1
            else:
                latent_idx = self.frame_dict[frame_idx]
            latent_idx = torch.Tensor([latent_idx]).long().to(garment_feature.device)
            latent_code = self.latent(latent_idx)
            latent_code = latent_code.expand(garment_feature.shape[0], -1)
            garment_feature = torch.cat([garment_feature, latent_code], dim=1)

        # add xyz of garment joint to concatenate with garment feature
        garment_joint_xyz_norm = aabb.normalize(garment_joint_xyz, sym=True)
        garment_feature = self.position_proj(garment_joint_xyz_norm, cond=garment_feature)

        # output joint position and rotation
        garment_joint_rot = self.rot_proj(garment_feature)
        garment_joint_trans = self.trans_proj(garment_feature)


        return garment_joint_rot, garment_joint_trans




class NonRigidDeform(nn.Module):
    def __init__(self, cfg, metadata, pose_encoder):
        super().__init__(cfg)
        self.pose_encoder = pose_encoder
        d_cond = self.pose_encoder.n_output_dims # n_body_joints * feature_dim_per_joint
        self.num_body_joints = cfg.pose_encoder.get('num_joints', 24)

        self.garment_pose_decoder = GarmentPoseDecoder(**cfg.garment_pose_decoder)

        # add latent code
        self.latent_dim = cfg.get('latent_dim', 0)
        if self.latent_dim > 0:
            d_cond += self.latent_dim
            self.frame_dict = metadata['frame_dict']
            self.latent = nn.Embedding(len(self.frame_dict), self.latent_dim)

        d_in = 3
        d_out = 3 + 3 + 4
        self.feature_dim = cfg.get('feature_dim', 0)
        d_out += self.feature_dim

        # output dimension: position + scale + rotation
        self.mlp = VanillaCondMLP(d_in, d_cond, d_out, cfg.mlp)

        self.skinning_weights = torch.from_numpy(metadata['skinning_weights']).float().cuda()

        self.delay = cfg.get('delay', 0)

        # number of pesudo joints for garment
        self.joint_num = cfg.get('joint_num', 80)

    def garment_joint_prediction(self, gaussians):
        # TODO: partition gaussians into different groups and apply kmeans in each group to produce group-wise garment joints
        gaussian_feature = gaussians.get_latent_feature # (bs, #gaussians, feature_dim)
        xyz = gaussians.get_xyz

        soft_kmeans = SoftKMeans()
        cluster_result = soft_kmeans(gaussian_feature, k=self.joint_num)

        c_assign = cluster_result.soft_assignment # softmax probability (bs, num_init, #gaussians, #joints)
        bs, num_init, n, k = c_assign.size()
        d = xyz.size(-1)
        per_cluster = c_assign.sum(dim=-2)
        # get garment joint xyz and joint feature
        # (BS, num_init, N, K)
        # -> (BS, num_init, K, 1, N) @ (BS, num_init, K, N, D)
        # -> (BS, num_init, K, D)
        cluster_xyz_mean = (
            c_assign.permute(0, 1, 3, 2)[:, :, :, None, :]
            @ xyz[:, None, None, :, :].expand(bs, num_init, k, n, d)
        ).squeeze(-2)
        joint_xyz = torch.diag_embed(1.0 / (per_cluster + self.eps)) @ cluster_xyz_mean
        joint_feature = cluster_result.centers.squeeze(1) # (bs, #joints, feature_dim)

        return joint_xyz, joint_feature
    

    def forward(self, gaussians, iteration, camera):

        smpl_cano_verts = camera.smpl_cano_verts

        if iteration < self.delay:
            deformed_gaussians = gaussians.clone()
            if self.feature_dim > 0:
                setattr(deformed_gaussians, "non_rigid_feature", torch.zeros(gaussians.get_xyz.shape[0], self.feature_dim).cuda())
            return deformed_gaussians, {}

        # obtain body pose feature given absolute body joint positions and rotations
        rots = camera.rots
        Jtrs = camera.Jtrs

        # change rots from rotation matrix to quaternion
        bs, num_body_joints, _ = rots.size()
        rots = rotation_matrix_to_quaternion(rots.view(-1, 9).view(-1, 3, 3)).view(bs, num_body_joints, -1)

        pose_feat = self.pose_encoder(rots, Jtrs)
        pose_feat = pose_feat.view(self.num_body_joints, -1)

        # predict garment joint position and feature 
        garment_joint_xyz, garment_joint_feature = self.garment_joint_prediction(gaussians)

        # predict garment joint rotaion and translation wrt smpl root
        garment_joint_rot, garment_joint_trans = self.garment_pose_decoder(pose_feat, garment_joint_feature, garment_joint_xyz, smpl_cano_verts)

        # garment joint to vertices lbs weights
        xyz = gaussians.get_xyz
        deformed_gaussians._xyz, deformed_garments_joint_xyz = pesudo_lbs(garment_joint_xyz, 
                                                                          garment_joint_rot,
                                                                          garment_joint_trans,
                                                                          rots[:, 0, ...],
                                                                          Jtrs[:, 0, ...],
                                                                          xyz)


        return deformed_gaussians
    
        
    def regularization(self, camera):
        # TODO: compute lbs loss: regularization for pesudo_lbs to deform gaussians to similar positions as lbs when using garment joints
        # sample canonical gaussian points and apply rigid and non-rigid deformation, they should be close to each other at early iterations
        pass
        

def get_non_rigid_deform(cfg, metadata, pose_encoder):
    return NonRigidDeform(cfg, metadata, pose_encoder)