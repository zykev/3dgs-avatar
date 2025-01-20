import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch3d.ops as ops
import trimesh
import igl

from utils.general_utils import build_rotation
from models.network_utils import VanillaCondMLP
from deformation.lbs import lbs, get_transforms_02v


class LBSWeightOptimizer(nn.Module):

    #   n_reg_pts: 1024
    def __init(self, input_dims = 3, output_dims = 25, distill = True, distill_grid_res = (64//4, 64, 64), softmax_blend=20, **kwargs):
        super().__init__()

        cfg_lbsnet = {"n_neurons": 128, "n_hidden_layers": 4, "skip_in": [], "cond_in": [], "multires": 0}
        self.lbs_network = VanillaCondMLP(input_dims, 0, output_dims, cfg_lbsnet)

        self.distill = distill
        self.resolution = distill_grid_res
        d, h, w = distill_grid_res
        if self.distill:
            self.grid = create_voxel_grid(d, h, w).cuda()
            self.precompute_lbs_voxel_weights()

        self.softmax_blend = softmax_blend

    def softmax(self, logit):
        if logit.shape[-1] == 25:
            w = hierarchical_softmax(logit)
        elif logit.shape[-1] == 24:
            w = F.softmax(logit, dim=-1)
        else:
            raise ValueError
        return w
    
    def precompute_lbs_voxel_weights(self):
        # precompute lbs weights of grid points
        if not hasattr(self, "lbs_voxel_weights"):
            d, h, w = self.resolution

            lbs_voxel_weights = self.lbs_network(self.grid[0]).float()
            lbs_voxel_weights = self.softmax_blend * lbs_voxel_weights 
            lbs_voxel_weights = self.softmax(lbs_voxel_weights)

            self.lbs_voxel_weights = lbs_voxel_weights.permute(1, 0).reshape(1, 24, d, h, w)

    def forward(self, xyz):
        # obtain lbs weights of input points xyz
        if self.distill:
            bs, n_points, _ = xyz.shape
            pts_W = F.grid_sample(self.lbs_voxel_weights.expand(bs, -1, -1, -1, -1),
                                  xyz.view(bs, 1, 1, -1, 3),
                                  mode='bilinear',
                                  padding_mode='border',
                                  align_corners=True) 
            pts_W = pts_W.squeeze(-1).squeeze(-1).permute(0, 2, 1) # (bs, #gaussians, 24)
        
        else:
            pts_W = self.lbs_network(xyz)
            pts_W = self.softmax(pts_W)

        return pts_W

class RigidDeform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, gaussians, iteration, camera):
        raise NotImplementedError

    def regularization(self, camera):
        return NotImplementedError

class Identity(RigidDeform):
    """ Identity mapping for single frame reconstruction """
    def __init__(self, cfg, metadata):
        super().__init__(cfg)

    def forward(self, gaussians, iteration, camera):
        return gaussians

    def regularization(self, camera):
        return {}

class SMPLNN(RigidDeform):
    def __init__(self, cfg, smpl_metadata):
        super().__init__(cfg)
        self.skinning_weights = torch.from_numpy(smpl_metadata["skinning_weights"]).float().cuda()

    def query_weights(self, xyz, smpl_verts):
        # find the nearest vertex
        knn_ret = ops.knn_points(xyz.unsqueeze(0), smpl_verts.unsqueeze(0))
        p_idx = knn_ret.idx.squeeze()
        pts_W = self.skinning_weights[p_idx, :]

        return pts_W

    def forward(self, gaussians, iteration, camera):
        smpl_verts = torch.from_numpy(camera.smpl_verts).float().cuda() # TODO: smpl verts in c space
        transforms_mat = camera.transforms_mat

        xyz = gaussians.get_xyz
        n_pts = xyz.shape[0]
        pts_W = self.query_weights(xyz, smpl_verts)
        T_fwd = torch.matmul(pts_W, transforms_mat.view(-1, 16)).view(n_pts, 4, 4).float()

        deformed_gaussians = gaussians.clone()
        deformed_gaussians.set_fwd_transform(T_fwd.detach())

        homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=xyz.device)
        x_hat_homo = torch.cat([xyz, homo_coord], dim=-1).view(n_pts, 4, 1)
        x_bar = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]
        deformed_gaussians._xyz = x_bar

        rotation_hat = build_rotation(gaussians._rotation)
        rotation_bar = torch.matmul(T_fwd[:, :3, :3], rotation_hat)
        setattr(deformed_gaussians, 'rotation_precomp', rotation_bar)
        # deformed_gaussians._rotation = tf.matrix_to_quaternion(rotation_bar)
        # deformed_gaussians._rotation = rotation_matrix_to_quaternion(rotation_bar)

        return deformed_gaussians

    def regularization(self, camera):
        return {}

def create_voxel_grid(d, h, w, device='cpu'):
    x_range = (torch.linspace(-1,1,steps=w,device=device)).view(1, 1, 1, w).expand(1, d, h, w)  # [1, H, W, D]
    y_range = (torch.linspace(-1,1,steps=h,device=device)).view(1, 1, h, 1).expand(1, d, h, w)  # [1, H, W, D]
    z_range = (torch.linspace(-1,1,steps=d,device=device)).view(1, d, 1, 1).expand(1, d, h, w)  # [1, H, W, D]
    grid = torch.cat((x_range, y_range, z_range), dim=0).reshape(1, 3,-1).permute(0,2,1)

    return grid

''' Hierarchical softmax following the kinematic tree of the human body. Imporves convergence speed'''
def hierarchical_softmax(x):
    def softmax(x):
        return F.softmax(x, dim=-1)

    def sigmoid(x):
        return torch.sigmoid(x)

    n_point, n_dim = x.shape

    prob_all = torch.ones(n_point, 24, device=x.device)
    # softmax_x = F.softmax(x, dim=-1)
    sigmoid_x = sigmoid(x).float()

    prob_all[:, [1, 2, 3]] = sigmoid_x[:, [0]] * softmax(x[:, [1, 2, 3]])
    prob_all[:, [0]] = 1 - sigmoid_x[:, [0]]

    prob_all[:, [4, 5, 6]] = prob_all[:, [1, 2, 3]] * (sigmoid_x[:, [4, 5, 6]])
    prob_all[:, [1, 2, 3]] = prob_all[:, [1, 2, 3]] * (1 - sigmoid_x[:, [4, 5, 6]])

    prob_all[:, [7, 8, 9]] = prob_all[:, [4, 5, 6]] * (sigmoid_x[:, [7, 8, 9]])
    prob_all[:, [4, 5, 6]] = prob_all[:, [4, 5, 6]] * (1 - sigmoid_x[:, [7, 8, 9]])

    prob_all[:, [10, 11]] = prob_all[:, [7, 8]] * (sigmoid_x[:, [10, 11]])
    prob_all[:, [7, 8]] = prob_all[:, [7, 8]] * (1 - sigmoid_x[:, [10, 11]])

    prob_all[:, [12, 13, 14]] = prob_all[:, [9]] * sigmoid_x[:, [24]] * softmax(x[:, [12, 13, 14]])
    prob_all[:, [9]] = prob_all[:, [9]] * (1 - sigmoid_x[:, [24]])

    prob_all[:, [15]] = prob_all[:, [12]] * (sigmoid_x[:, [15]])
    prob_all[:, [12]] = prob_all[:, [12]] * (1 - sigmoid_x[:, [15]])

    prob_all[:, [16, 17]] = prob_all[:, [13, 14]] * (sigmoid_x[:, [16, 17]])
    prob_all[:, [13, 14]] = prob_all[:, [13, 14]] * (1 - sigmoid_x[:, [16, 17]])

    prob_all[:, [18, 19]] = prob_all[:, [16, 17]] * (sigmoid_x[:, [18, 19]])
    prob_all[:, [16, 17]] = prob_all[:, [16, 17]] * (1 - sigmoid_x[:, [18, 19]])

    prob_all[:, [20, 21]] = prob_all[:, [18, 19]] * (sigmoid_x[:, [20, 21]])
    prob_all[:, [18, 19]] = prob_all[:, [18, 19]] * (1 - sigmoid_x[:, [20, 21]])

    prob_all[:, [22, 23]] = prob_all[:, [20, 21]] * (sigmoid_x[:, [22, 23]])
    prob_all[:, [20, 21]] = prob_all[:, [20, 21]] * (1 - sigmoid_x[:, [22, 23]])

    # prob_all = prob_all.reshape(n_batch, n_point, prob_all.shape[-1])
    return prob_all

class SkinningField(RigidDeform):
    def __init__(self, cfg, smpl_metadata):
        super().__init__(cfg)
        self.skinning_weights = smpl_metadata["skinning_weights"]
        self.faces = smpl_metadata["faces"]

        self.lbs_weight_optimizer = LBSWeightOptimizer()

    def forward_smpl(self, betas, pose, trans, xyz, lbs_weights):
        xyz_posed, joint_posed, joint_transforms = lbs(betas=betas,
                                                       pose=pose,
                                                       v_template=xyz,
                                                       shapedirs=self.shapedirs,
                                                       posedirs=self.posedirs,
                                                       J_regressor=self.J_regressor,
                                                       parents=self.kntree_parents,
                                                       lbs_weights=lbs_weights,
                                                       pose2rot=False,
                                                       dtype=torch.float32)

        joint_transforms_02v = get_transforms_02v(joint_posed.squeeze(0))
        joint_transforms = torch.matmul(joint_transforms.squeeze(0), torch.inverse(joint_transforms_02v))
        joint_transforms[:, :3, 3] = joint_transforms[:, :3, 3] + trans

        xyz_posed = xyz_posed + trans[None]

        # returen input rotation, input joints location before transformation (ie c space), 
        # bone transforms c space to p space transformation, vertices in pose space, 
        # vertices in c space (include shape and pose influence)
        return xyz_posed, joint_transforms
    

    def sample_skinning_loss(self, cano_mesh, smpl_verts):
        points_skinning, face_idx = cano_mesh.sample(self.cfg.n_reg_pts, return_index=True)
        points_skinning = points_skinning.view(np.ndarray).astype(np.float32)
        bary_coords = igl.barycentric_coordinates_tri(
            points_skinning,
            smpl_verts[self.faces[face_idx, 0], :],
            smpl_verts[self.faces[face_idx, 1], :],
            smpl_verts[self.faces[face_idx, 2], :],
        )
        vert_ids = self.faces[face_idx, ...]
        pts_W = (self.skinning_weights[vert_ids] * bary_coords[..., None]).sum(axis=1)

        points_skinning = torch.from_numpy(points_skinning).cuda()
        pts_W = torch.from_numpy(pts_W).cuda()
        return points_skinning, pts_W


    def get_skinning_loss(self, cano_mesh, smpl_verts, aabb):

        pts_skinning, sampled_weights = self.sample_skinning_loss(cano_mesh, smpl_verts)
        pts_skinning = aabb.normalize(pts_skinning, sym=True)

        pred_weights = self.lbs_weight_optimizer(pts_skinning) # .reshape(24, -1).permute(1, 0)

        skinning_loss = torch.nn.functional.mse_loss(
            pred_weights, sampled_weights, reduction='none').sum(-1).mean()
        # breakpoint()

        return skinning_loss


    def forward(self, gaussians, iteration, camera):

        aabb = camera.aabb
        xyz = gaussians.get_xyz
        xyz_norm = aabb.normalize(xyz, sym=True)

        # get skinning weights of input xyz
        lbs_weights = self.lbs_weight_optimizer(xyz_norm)
        # get forward transformation matrix and transformed xyz by smpl lbs
        transforms_mat, deformed_xyz = self.forward_smpl(camera.betas, camera.pose, camera.trans, xyz, lbs_weights)

        deformed_gaussians = gaussians.clone()
        deformed_gaussians.set_fwd_transform(transforms_mat.detach())

        deformed_gaussians._xyz = deformed_xyz

        rotation_hat = build_rotation(gaussians._rotation)
        rotation_bar = torch.matmul(transforms_mat[:, :3, :3], rotation_hat)
        setattr(deformed_gaussians, 'rotation_precomp', rotation_bar)
        # deformed_gaussians._rotation = rotation_matrix_to_quaternion(rotation_bar)

        return deformed_gaussians

    def regularization(self, camera):
        smpl_verts = camera.smpl_cano_verts
        cano_mesh = camera.smpl_cano_mesh
        aabb = camera.smpl_cano_verts_range
        loss_skinning = self.get_skinning_loss(smpl_verts, cano_mesh, aabb)
        return loss_skinning 

def get_rigid_deform(cfg, metadata):
    name = cfg.name
    model_dict = {
        "identity": Identity,
        "smpl_nn": SMPLNN,
        "skinning_field": SkinningField,
    }
    return model_dict[name](cfg, metadata)