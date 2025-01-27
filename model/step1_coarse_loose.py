import numpy as np
import os

import torch.nn as nn
from torch_geometric.nn.models import GAT
import torch

import sys
sys.path.append('.')
# from modelUtil.lbs import copyWeight
from .Layers import *
from .utils.transformUtil import transformSimplify,transformRestore

from psbody.mesh import Mesh as Obj

from data.utils.clothMesh import ClothMesh

class CoarseLooseModel(nn.Module):
    def __init__(self, cloth:ClothMesh, device):
        super(CoarseLooseModel, self).__init__()

        self.model_name = ""
        self.device = device

        self.cloth = cloth
        self.cloth_v = cloth.vertices[cloth.loose_v_indices].to(self.device)

        self.loadClothBones()
        self.joint_indices = cloth.joint_indices
        self.buildModel()

    def printModelInfo(self):
        print("------------step 1 model---------------")
        print("---Model Layers:",self)
        return


    def loadClothBones(self):
        self.loose_v_indices = self.cloth.loose_v_indices.to(self.device)
        self.cloth_loose_propotion = self.cloth.loose_propotion.to(self.device)
        self.cloth_bone_weights = self.cloth.bone_weights.to(self.device)
        self.cloth_bone_offset_matrices = self.cloth.bone_offset_matrix.to(self.device)
        # print("cloth bones num:", self.cloth_bone_weights.shape[1])


    @property
    def cloth_bone_num(self):
        return self.cloth_bone_offset_matrices.shape[0]


    def buildModel(self,dropout = 0.5):
        j_f_num = 9 # joint feature num
        self.coarse_hidden_size = 512
        self.trans_module = GRU_Model(encode_dim=[self.joint_indices.shape[0] * j_f_num+3, 512, 512],
                                          decode_dim=[self.coarse_hidden_size, self.cloth_bone_num * j_f_num],
                                          dropout=dropout
                                          )
        
        j_w = self.cloth.joint_weights[self.cloth.loose_v_indices]
        j_w = j_w[:,self.joint_indices].to(self.device)

        self.pt_module = PostTranslation(j_w,self.cloth_bone_weights,train_weights=True)
        
        return

    def data_forward(self,body_transforms,hip_trans, in_hidden):
        body_transforms = body_transforms[:,:,self.joint_indices]
        input_feature = transformSimplify(body_transforms).view(*body_transforms.shape[:2],-1)
        input_feature = torch.cat([input_feature,hip_trans],dim=-1)

        if in_hidden is None:
            in_hidden = torch.zeros([1,body_transforms.shape[0],self.coarse_hidden_size],device=self.device)
        


        return body_transforms, input_feature, in_hidden

    def network_forward(self,body_transforms, tran_simplified, in_hidden):
        x, next_hidden = self.trans_module(tran_simplified,in_hidden)
        x = x.view(*tran_simplified.shape[:2],-1,9)

        body_translation = body_transforms.view(*body_transforms.shape[:2],-1,4,4)[...,:3,3]
        x = self.pt_module(body_translation, x)
        return  body_transforms, x, next_hidden




    def forward(self, body_transforms, hip_trans, in_hidden):
        body_transforms, input_feature, hidden = self.data_forward(body_transforms, hip_trans, in_hidden)
        return self.network_forward(body_transforms,input_feature, hidden)
    