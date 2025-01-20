
import torch.nn as nn


from models.network_utils import HierarchicalPoseEncoder,VanillaCondMLP 
from utils.general_utils import rotation_matrix_to_quaternion, quaternion_to_rotation_matrix



class NoPoseCorrection(nn.Module):
    def __init__(self, config, metadata=None, pose_encoder=None):
        super(NoPoseCorrection, self).__init__()

    def forward(self, camera, iteration):
        return camera, {}

    def regularization(self, out):
        return {}

    
class BodyPoseOptimizer(nn.Module):
    # optimize body pose (root orientation, root & body joint rotations)
    def __init__(self, cfg, smpl_metadata, pose_encoder, 
                 rot_input_dims=6, trans_input_dims=3, rot_output_dims=4, trans_output_dims=3): 
        # input_dims: per joint feature dim in pose encoder
        # output_dims: 4 for quanterion output
        super(BodyPoseOptimizer, self).__init__(cfg, smpl_metadata)
        self.cfg = cfg

        self.v_template = smpl_metadata['v_template']
        self.shapedirs = smpl_metadata['shapedirs']
        self.posedirs = smpl_metadata['posedirs']
        self.J_regressor = smpl_metadata['J_regressor']
        self.lbs_weights = smpl_metadata['lbs_weights']
        self.kntree_parents = smpl_metadata['kintree_parents']

        self.pose_encoder = pose_encoder

        cfg_trans_decoder = {"n_neurons": 16, "n_hidden_layers": 1, "skip_in": [], "cond_in": [], "multires": 0, "activation": "tanh"}
        self.trans_decoder = VanillaCondMLP(trans_input_dims, 0, trans_output_dims, cfg_trans_decoder) # decode root translation
        
        self.rot_decoder = nn.Sequential(nn.Linear(rot_input_dims, rot_output_dims)) # decode root & body joint rotations

    
    def forward(self, camera, iteration):
        if iteration < self.cfg.get('delay', 0):
            return camera, {}

        rots = camera.rots
        Jtrs = camera.Jtrs
        trans = camera.trans
        # get optimized rotation and translation
        # change rots from rotation matrix to quaternion
        bs, num_body_joints, _ = rots.size()
        rots = rotation_matrix_to_quaternion(rots.view(-1, 9).view(-1, 3, 3)).view(bs, num_body_joints, -1)
        pose_feat = self.pose_encoder(rots, Jtrs)
        pose_feat = pose_feat.view(bs*num_body_joints, -1)
        rots_refined = self.rot_decoder(pose_feat)
        rots_refined = quaternion_to_rotation_matrix(rots_refined).view(bs, num_body_joints, -1)

        trans_refined = self.trans_decoder(trans) + trans

        updated_camera = camera.copy()
        updated_camera.update(rots=rots_refined, trans=trans_refined)

        return updated_camera
    
    def regularization(self, camera, camera_update):
        rots_diff = camera.rots - camera_update.rots
        loss_poserefine_reg = (rots_diff ** 2).mean()
        return loss_poserefine_reg

def get_pose_correction(cfg, metadata, pose_encoder):
    name = cfg.name
    model_dict = {
        "none": NoPoseCorrection,
        "direct": BodyPoseOptimizer,
    }
    return model_dict[name](cfg, metadata, pose_encoder)



    # move to rigid body transformation
    # def export(self, frame):
    #     model_dict = {}

    #     idx = torch.Tensor([self.frame_dict[frame]]).long().to(self.betas.device)
    #     root_orient = self.root_orients(idx)
    #     pose_body = self.pose_bodys(idx)
    #     pose_hand = self.pose_hands(idx)
    #     trans = self.trans(idx)

    #     betas = self.betas

    #     rots, Jtrs, bone_transforms, posed_smpl_verts, v_posed, Jtr_posed = self.forward_smpl(betas, root_orient, pose_body,
    #                                                                             pose_hand, trans)
    #     model_dict.update({
    #         'minimal_shape': v_posed[0],
    #         'betas': betas,
    #         'Jtr_posed': Jtr_posed[0],
    #         'bone_transforms': bone_transforms,
    #         'trans': trans[0],
    #         'root_orient': root_orient[0],
    #         'pose_body': pose_body[0],
    #         'pose_hand': pose_hand[0],
    #     })
    #     for k, v in model_dict.items():
    #         model_dict.update({k: v.detach().cpu().numpy()})
    #     return model_dict