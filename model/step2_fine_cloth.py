import torch.nn as nn
from torch_geometric.nn.models import GAT
import torch

import sys
sys.path.append('.')
# from modelUtil.lbs import copyWeight
from .Layers import *

from psbody.mesh import Mesh as Obj

from .step1_coarse_loose import CoarseLooseModel
from data.datasets import ModelDataset
from .utils.transformUtil import transformSimplify,transformRestore
from data.utils.clothMesh import ClothMesh

class FineClothModel(nn.Module):
    def __init__(self,cloth:ClothMesh, config_model):
        super(FineClothModel, self).__init__()

        self.model_name = ""
        self.config_model = config_model


        self.buildModel(cloth.mixed_weights.shape[1],cloth.vertices.shape)
        return

    def buildModel(self, bone_num, cloth_shape, hidden_size = 512, dropout = 0.5):
        j_f_num = 12 # joint feature num
        self.hidden_size = hidden_size
        vertices_encoder = GRU_Model(encode_dim=[bone_num*j_f_num, 512, 512],
                                    decode_dim=[self.hidden_size, 512],
                                    dropout=dropout
                                    )
        vertices_decoder = PSD(512, cloth_shape)
        self.vt_gru = vertices_encoder
        self.vt_psd = vertices_decoder
        return


    def data_forward(self, AB_Trans, in_hidden):
        if in_hidden is None:
            in_hidden = torch.zeros([1,AB_Trans.shape[0],self.hidden_size],device=AB_Trans.device)
        AB_Trans = AB_Trans.reshape(*AB_Trans.shape[:2],-1)
        return AB_Trans, in_hidden

    def networkForward(self, AB_Trans, fine_hidden):
        gru_out, next_fine_hidden = self.vt_gru(AB_Trans, fine_hidden)
        x = self.vt_psd(gru_out)
        return x, next_fine_hidden
    
    def forward(self, AB_Trans, in_hidden):
        AB_t, hidden = self.data_forward(AB_Trans,in_hidden)
        return  self.networkForward(AB_t, hidden)
