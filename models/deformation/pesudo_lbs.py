# -*- coding: utf-8 -*-


import torch
import torch.nn.functional as F

from utils.general_utils import quaternion_to_rotation_matrix


def pesudo_lbs(joints, joint_rot, joint_trans, root_rot, root_trans, v_template, dtype=torch.float32):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        joints: torch.tensor bs x J x 3
            The position of the joints
        rotation_mat : torch.tensor bs x # J * 4
            The rotation quaternion of joints
        translation : torch.tensor bs x J x 3
            The translation of the joints
        v_template torch.tensor BxVx3
            The template mesh that will be deformed

        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size, num_joints, _ = joints.size()
    device = joints.device

    joint_rot = quaternion_to_rotation_matrix(joint_rot)
    root_rot = quaternion_to_rotation_matrix(root_rot.unsqueeze(1))

    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(joint_rot, joint_trans, joints, root_rot, root_trans, dtype=dtype)

    # 5. Do skinning:
    lbs_weights = compute_lbs_weights(joints, v_template)
    # W is N x V x (J + 1)
    W = lbs_weights.repeat([batch_size, 1, 1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_template.shape[1], 1], dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_template, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed


def compute_lbs_weights(self, garment_joints, gaussian_points):
        """
        计算 LBS weights。

        参数:
            garment_joints: 形状为 (bs, 80, 3) 的张量，表示 garment joints 的位置。
            gaussian_points: 形状为 (bs, # gaussians, 3) 的张量，表示高斯点的位置。

        返回:
            lbs_weights: 形状为 (bs, # gaussians, 80) 的张量，表示 LBS weights。
        """

        # 扩展 garment_joints 和 gaussian_points 的维度以支持广播
        garment_joints = garment_joints.unsqueeze(1)  # (bs, 1, 80, 3)
        gaussian_points = gaussian_points.unsqueeze(2)  # (bs, # gaussians, 1, 3)

        # 计算每个高斯点到每个 garment joint 的距离
        distances = torch.norm(gaussian_points - garment_joints, dim=-1)  # (bs, # gaussians, 80)

        # 使用距离的倒数作为权重
        inv_distances = 1.0 / (distances + 1e-8)  # 避免除零错误

        # 归一化权重，确保每个高斯点的权重和为 1
        lbs_weights = inv_distances / inv_distances.sum(dim=-1, keepdim=True)  # (bs, # gaussians, 80)

        return lbs_weights

def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(joint_rot_mats, joint_transl, joints, root_rot_mats, root_transl, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    joint_rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices of joints
    joint_transl : torch.tensor BxNx3
        Tensor of translations of joints
    root_rot_mats : torch.tensor Bx3x3
        Tensor of rotation matrices of the root joint
    root_transl : torch.tensor Bx3
        Tensor of translations of the root joint
    joints : torch.tensor BxNx3
        Locations of joints
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    bs, num_joints, _ = joints.size()
    device = joints.device

    joints = torch.unsqueeze(joints, dim=-1)

    joint_transforms_mats = transform_mat(
        joint_rot_mats.view(-1, 3, 3),
        joint_transl.reshape(-1, 3, 1)).view(-1, joints.shape[1], 4, 4) # (bs, n, 4, 4)
    
    root_transforms_mats = transform_mat(
        root_rot_mats.view(-1, 3, 3),
        torch.unsqueeze(root_transl, dim=-1).reshape(-1, 3, 1)).unsqueeze(1) # (bs, 1, 4, 4)

    # 计算全局变换矩阵 (bs, n, 4, 4)
    global_transforms = torch.matmul(root_transforms_mats, joint_transforms_mats)            
    joints_homo = torch.cat([joints, torch.ones(bs, num_joints, 1, device=device)], dim=-1).unsqueeze(-1)

    transformed_coords_homo = torch.matmul(global_transforms, joints_homo)
    transformed_joints = transformed_coords_homo[..., :3, 0]

    return transformed_joints, global_transforms

def batch_rodrigues(aa_rots):
    '''
    convert batch of rotations in axis-angle representation to matrix representation
    :param aa_rots: Nx3
    :return: mat_rots: Nx3x3
    '''

    dtype = aa_rots.dtype
    device = aa_rots.device

    batch_size = aa_rots.shape[0]

    angle = torch.norm(aa_rots + 1e-8, dim=1, keepdim=True)
    rot_dir = aa_rots / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat