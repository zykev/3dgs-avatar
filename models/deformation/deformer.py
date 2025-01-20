import torch.nn as nn

from models.deformation.rigid import get_rigid_deform
from models.deformation.non_rigid import get_non_rigid_deform
from models.deformation.pose_correction import get_pose_correction
from models.network_utils import HierarchicalPoseEncoder

class Deformer(nn.Module):
    def __init__(self, cfg, metadata):
        super().__init__()
        self.cfg = cfg

        self.pose_encoder = HierarchicalPoseEncoder(cfg.pose_encoder)
        self.pose_correction = get_pose_correction(cfg.pose_correction, metadata, self.pose_encoder)
        self.rigid = get_rigid_deform(cfg.rigid, metadata)
        self.non_rigid = get_non_rigid_deform(cfg.non_rigid, metadata, self.pose_encoder)

    def forward(self, gaussians, camera, iteration):
        
        camera_updated = self.pose_correction(camera, iteration)
        deformed_gaussians = self.non_rigid(gaussians, iteration, camera_updated)
        deformed_gaussians = self.rigid(deformed_gaussians, iteration, camera_updated)

        loss_reg = {'loss_poserefine_reg': self.pose_correction.regularization(camera, camera_updated),
                        'loss_skinning_reg': self.rigid.regularization(),
                        'loss_lbs_reg': self.non_rigid.regularization(),
                        }

        return deformed_gaussians, loss_reg

def get_deformer(cfg, metadata):
    return Deformer(cfg, metadata)