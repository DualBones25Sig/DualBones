import torch
import numpy as np
import open3d as o3d

from .transformUtil import transformRestore

def lbs(vertices:torch.Tensor,
        bs_transforms:torch.Tensor,
        weights:torch.Tensor,
        offset_matrix:torch.Tensor):
    
    M = torch.matmul(bs_transforms, offset_matrix)
    M_W = torch.matmul(weights,M.view(*M.shape[:-2],16))
    M_W = M_W.view(*M_W.shape[:-1],4,4)
    src_stack = torch.cat([vertices, torch.ones([*vertices.shape[:-1], 1]).to(vertices.device).float()],dim=-1)
    return torch.matmul(M_W, src_stack.unsqueeze(-1)).squeeze(-1)[...,:3]

def normalizeSafe(w):
    r_sum = w.sum(dim=-1, keepdims=True)
    n_w = w/r_sum
    n_w[r_sum[:,0] == 0] = 0
    return n_w

