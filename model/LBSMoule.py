import torch.nn as nn
import torch
from .utils.lbs import lbs,normalizeSafe

from scipy.optimize import lsq_linear



class LBSLayer(nn.Module):
    def __init__(self,
                 blend_weights,
                 offset_matrix,
                #  vb_start_ind,
                 train_weights = False,
                 weights_optimizer = "mask_lsq",
                 device = 'cuda:0',
                 ):
        super().__init__()
        """
        weights optimizer: one choice of [para, mask, mask_lsq]
        """
        # initialize weights
        self.train_weights = train_weights
        if train_weights:
            assert weights_optimizer in ["para", "mask", "mask_lsq"], "weight optimizer must be one of [para, mask, mask_lsq]"
        
        self.weights_optimizer = weights_optimizer
        self.device = device
        self.offset_matrices = offset_matrix.to(device)
        blend_weights = blend_weights.to(device)
        self.setWeigths(blend_weights)
        
        self.lsq_cache_bs = []
        self.lsq_cache_gt = []




    def setWeigths(self, blend_weights):
        blend_weights = normalizeSafe(blend_weights)
        if self.train_weights :
            if self.weights_optimizer == "mask_lsq":
                self.param_mask = (blend_weights > 0).to(torch.int32)
                self.blend_weights = blend_weights
                self.para_holder = nn.Parameter(torch.zeros([1],device=self.device))
            elif self.weights_optimizer == 'mask':
                self.param_mask = (blend_weights > 0).to(torch.int32)
                self.blend_weights = nn.Parameter(blend_weights)
            else:
                self.blend_weights = nn.Parameter(blend_weights)
        else:
            self.blend_weights = blend_weights
        
    def getWeights(self):
        if self.train_weights:
            if self.weights_optimizer == "mask_lsq":
                return self.blend_weights
            elif self.weights_optimizer == 'mask':
                return normalizeSafe(self.blend_weights * self.param_mask)
            else:
                return normalizeSafe(self.blend_weights)
        else:
            return self.blend_weights


    def __call__(self, src, transforms:torch.Tensor):
        return lbs(src, transforms, self.getWeights(), self.offset_matrices)

    
    def lsqAddCache(self, transforms, g_t):
        self.lsq_cache_bs.append(transforms.view(-1,*transforms.shape[-3:]).detach())
        self.lsq_cache_gt.append(g_t.view(-1,*g_t.shape[-2:]))

    def lsqClearCache(self):
        self.lsq_cache_bs = []
        self.lsq_cache_gt = []
    

    def LsqUpdateWeights(self, bind_pose):
        import numpy as np
        """
        inspire by SSDR weights optimizer, see more information in https://github.com/dalton-omens/SSDR
        """
        if self.train_weights == False:
            return

        bone_transforms = torch.cat(self.lsq_cache_bs, dim = 0)
        ground_truth = torch.cat(self.lsq_cache_gt, dim = 0)


        weights = torch.zeros_like(self.blend_weights,device=self.device)

        num_bones = bone_transforms.shape[1]
        num_poses = bone_transforms.shape[0]
        mask_clone = self.param_mask.clone()

        for v_ind in range(bind_pose.shape[0]):
            sel_ind = torch.where(self.param_mask[v_ind] > 0)[0]
            if len(sel_ind) == 0:
                continue
            b_t = bone_transforms[:,sel_ind]

            R_o = (bind_pose[v_ind].unsqueeze(0) + self.offset_matrices[sel_ind, :3, 3]).unsqueeze(0).expand(num_poses, -1, -1)
            Rp = torch.matmul(b_t[...,:3,:3], R_o.unsqueeze(-1)).squeeze(-1) # num_pose, num_bone, 3
            Rp_T = Rp + b_t[...,:3,3]# poses, bone 3

            A = Rp_T.permute(0,2,1).reshape(3 * Rp_T.shape[0], len(sel_ind)).cpu().numpy()
            b = ground_truth[:,v_ind].reshape(-1).cpu().numpy()
            w = lsq_linear(A, b, bounds=(0, 1), method='bvls').x

            weights[v_ind,sel_ind] = torch.tensor(w,device=self.device)
        self.setWeigths(weights)
        self.param_mask = mask_clone
        self.lsqClearCache()


        

