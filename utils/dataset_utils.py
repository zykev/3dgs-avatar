import math
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import os

from scipy.spatial.transform import Rotation
from scene.gaussian_model import BasicPointCloud
from plyfile import PlyData, PlyElement

# add ZJUMoCAP dataloader
# def get_02v_bone_transforms(Jtr,):
    # rot45p = Rotation.from_euler('z', 45, degrees=True).as_matrix()
    # rot45n = Rotation.from_euler('z', -45, degrees=True).as_matrix()

    # # Specify the bone transformations that transform a SMPL A-pose mesh
    # # to a star-shaped A-pose (i.e. Vitruvian A-pose)
    # bone_transforms_02v = np.tile(np.eye(4), (24, 1, 1))

    # # First chain: L-hip (1), L-knee (4), L-ankle (7), L-foot (10)
    # chain = [1, 4, 7, 10]
    # rot = rot45p.copy()
    # for i, j_idx in enumerate(chain):
    #     bone_transforms_02v[j_idx, :3, :3] = rot
    #     t = Jtr[j_idx].copy()
    #     if i > 0:
    #         parent = chain[i-1]
    #         t_p = Jtr[parent].copy()
    #         t = np.dot(rot, t - t_p)
    #         t += bone_transforms_02v[parent, :3, -1].copy()

    #     bone_transforms_02v[j_idx, :3, -1] = t

    # bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)
    # # Second chain: R-hip (2), R-knee (5), R-ankle (8), R-foot (11)
    # chain = [2, 5, 8, 11]
    # rot = rot45n.copy()
    # for i, j_idx in enumerate(chain):
    #     bone_transforms_02v[j_idx, :3, :3] = rot
    #     t = Jtr[j_idx].copy()
    #     if i > 0:
    #         parent = chain[i-1]
    #         t_p = Jtr[parent].copy()
    #         t = np.dot(rot, t - t_p)
    #         t += bone_transforms_02v[parent, :3, -1].copy()

    #     bone_transforms_02v[j_idx, :3, -1] = t

    # bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)

    # return bone_transforms_02v


def get_02v_bone_transforms(Jtr):
    """
    Compute bone transformations that transform a SMPL A-pose mesh
    to a star-shaped A-pose (Vitruvian A-pose) using PyTorch.
    
    Args:
        Jtr (torch.Tensor): Joint locations of shape (24, 3)
    
    Returns:
        torch.Tensor: Bone transformations of shape (24, 4, 4)
    """
    device = Jtr.device  # Ensure computations stay on the same device

    # Define rotation matrices
    rot45p = torch.tensor(Rotation.from_euler('z', 45, degrees=True).as_matrix(), dtype=torch.float32, device=device)
    rot45n = torch.tensor(Rotation.from_euler('z', -45, degrees=True).as_matrix(), dtype=torch.float32, device=device)

    # Initialize transformation matrices (24 x 4 x 4 identity matrices)
    bone_transforms_02v = torch.eye(4, dtype=torch.float32, device=device).repeat(24, 1, 1)

    # First chain: L-hip (1), L-knee (4), L-ankle (7), L-foot (10)
    chain = [1, 4, 7, 10]
    rot = rot45p.clone()
    for i, j_idx in enumerate(chain):
        bone_transforms_02v[j_idx, :3, :3] = rot
        t = Jtr[j_idx].clone()
        if i > 0:
            parent = chain[i - 1]
            t_p = Jtr[parent].clone()
            t = torch.matmul(rot, (t - t_p))
            t += bone_transforms_02v[parent, :3, -1]

        bone_transforms_02v[j_idx, :3, -1] = t

    bone_transforms_02v[chain, :3, -1] -= torch.matmul(Jtr[chain], rot.T)

    # Second chain: R-hip (2), R-knee (5), R-ankle (8), R-foot (11)
    chain = [2, 5, 8, 11]
    rot = rot45n.clone()
    for i, j_idx in enumerate(chain):
        bone_transforms_02v[j_idx, :3, :3] = rot
        t = Jtr[j_idx].clone()
        if i > 0:
            parent = chain[i - 1]
            t_p = Jtr[parent].clone()
            t = torch.matmul(rot, (t - t_p))
            t += bone_transforms_02v[parent, :3, -1]

        bone_transforms_02v[j_idx, :3, -1] = t

    bone_transforms_02v[chain, :3, -1] -= torch.matmul(Jtr[chain], rot.T)

    return bone_transforms_02v


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

class AABB(torch.nn.Module):
    def __init__(self, coord_max, coord_min):
        super().__init__()
        self.coord_max = coord_max
        self.coord_min = coord_min
        # self.register_buffer("coord_max", torch.from_numpy(coord_max).float())
        # self.register_buffer("coord_min", torch.from_numpy(coord_min).float())

    def normalize(self, x, sym=False):
        x = (x - self.coord_min) / (self.coord_max - self.coord_min)
        if sym:
            x = 2 * x - 1.
        return x

    def unnormalize(self, x, sym=False):
        if sym:
            x = 0.5 * (x + 1)
        x = x * (self.coord_max - self.coord_min) + self.coord_min
        return x

    def clip(self, x):
        return x.clip(min=self.coord_min, max=self.coord_max)

    def volume_scale(self):
        return self.coord_max - self.coord_min

    def scale(self):
        return math.sqrt((self.volume_scale() ** 2).sum() / 3.)


def preprocess_image(img_file, K, D, img_size, white_bg=False, pca_components=48):

    mask_file = img_file.replace('images', 'mask').replace('.jpg', '.png')
    feature_file = img_file.replace('images', 'feat').replace('.jpg', '.npy')
    seglabel_file = img_file.replace('images', 'seg').replace('.jpg', '.npy')
    depth_file = img_file.replace('images', 'depth').replace('.jpg', '.npy')
    normal_file = img_file.replace('images', 'normal').replace('.jpg', '.npy')

    # image & mask
    image = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    image = cv2.undistort(image, K, D, None)
    mask = cv2.undistort(mask, K, D, None)

    image = cv2.resize(image, img_size, interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)

    mask = mask != 0
    image[~mask] = 255. if white_bg else 0.
    image = image / 255.

    image = torch.from_numpy(image).permute(2, 0, 1).float()
    mask = torch.from_numpy(mask).unsqueeze(0).float()

    # segmentation label
    seg_label = None
    if os.path.exists(seglabel_file):
        seg_logits = torch.from_numpy(np.load(seglabel_file))  # (28, h, w)
        seg_logits = F.interpolate(seg_logits.unsqueeze(0), size=img_size, mode="bilinear").squeeze(0)
        seg_label = seg_logits.argmax(dim=0, keepdim=True)  # (1, new_h, new_w)
        seg_label = seg_label.long() 
    
    # depth
    depth = None
    if os.path.exists(depth_file):
        depth = torch.from_numpy(np.load(depth_file))  # (h, w)
        depth = F.interpolate(depth.unsqueeze(0), size=img_size, mode="bilinear").squeeze(0)

    # normal
    normal = None
    if os.path.exists(normal_file):
        normal = torch.from_numpy(np.load(normal_file)).permute(2, 0, 1)  # (3, h, w)
        normal = F.interpolate(normal.unsqueeze(0), size=img_size, mode="bilinear").squeeze(0)

    # semantic feature
    semantic_feature = torch.from_numpy(np.load(feature_file))
    semantic_feature = pca_feature(semantic_feature, mask, mask_resize_shape=(1024, 1024), downsample_size=16, pca_components=pca_components)
    semantic_feature = F.interpolate(semantic_feature.unsqueeze(0), size=img_size, mode="bilinear").squeeze(0)  # (C, new_h, new_w)

    return image, mask, semantic_feature, seg_label, depth, normal

def pca_feature(feature_map, mask, mask_resize_shape=(1024, 1024), downsample_size=16, pca_components=48):
    """
    使用 PyTorch 进行 PCA 并应用到前景区域。

    Args:
        feature_map (torch.Tensor): (C, H, W) 形状的特征图
        mask (torch.Tensor): (1, H, W) 形状的掩码
        mask_resize_shape (tuple): 插值后的尺寸
        downsample_size (int): 池化下采样尺寸
        pca_components (int): PCA 维度

    Returns:
        torch.Tensor: (C_pca, H, W) 形状的 PCA 特征图
    """

    # 处理 mask，生成 feature_mask
    if mask.shape[-1] != feature_map.shape[-1]:
        mask = mask.unsqueeze(0).float()  # (1, 1, H, W)
        if mask.shape[-2:] != mask_resize_shape:
            mask = F.interpolate(mask, size=mask_resize_shape, mode='nearest')
        feature_mask = F.avg_pool2d(mask, kernel_size=downsample_size, stride=downsample_size)
        feature_mask = (feature_mask > 0.5).squeeze(0).squeeze(0)  # (h, w)
    else:
        feature_mask = mask.squeeze(0).bool()
    
    # 处理 feature_map
    feature_map = feature_map.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
    h, w, c = feature_map.shape
    feature_map_flat = feature_map.reshape(-1, c)  # (H*W, C)

    # 仅对前景特征应用 PCA
    feature_mask_flat = feature_mask.flatten()
    fg_features = feature_map_flat[feature_mask_flat]  # 仅保留前景区域

    if pca_components > 0:
        U, S, V = torch.pca_lowrank(fg_features, q=pca_components, center=True)
        pca_features = torch.matmul(fg_features - fg_features.mean(dim=0), V)  # (num_fg, pca_components)

        # 归一化 (Min-Max Scaling)
        min_vals, max_vals = pca_features.min(dim=0)[0], pca_features.max(dim=0)[0]
        pca_features = (pca_features - min_vals) / (max_vals - min_vals + 1e-6)  # 避免除 0
        pca_features = pca_features.clamp(0.0, 1.0)

        # 生成 PCA 特征图
        pca_feature_map = torch.zeros((h * w, pca_components), device=feature_map.device)
        pca_feature_map[feature_mask_flat] = pca_features
        pca_feature_map = pca_feature_map.reshape(h, w, pca_components)
        pca_feature_map = pca_feature_map.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

    
    else:
        pca_feature_map = torch.zeros((h * w, c), device=feature_map.device)
        pca_feature_map[feature_mask_flat] = fg_features
        pca_feature_map = pca_feature_map.reshape(h, w, c)
        pca_feature_map = pca_feature_map.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)


    return pca_feature_map
