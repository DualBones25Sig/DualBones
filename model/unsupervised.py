# import torch.nn as nn
# from torch_geometric.nn.models import GAT
# import torch

# import sys
# sys.path.append('.')
# # from modelUtil.lbs import copyWeight
# from .Layers import *

# from psbody.mesh import Mesh as Obj


# def getDerivative(X, n, dt):
#     for i in range(n):
#         X = (X[1:] - X[:-1]) / dt
#     return X

# class GRU_Model(nn.Module):
#     def __init__(self, input_num, hidden_num, output_num):
#         super(GRU_Model, self).__init__()
#         self.hidden_num = hidden_num
#         self.cell = nn.GRUCell(input_num, hidden_num)
#         self.output_num = copy.deepcopy(output_num)
#         self.output_num.insert(0, hidden_num)

#         self.out_mlp = nn.Sequential(
#             nn.Linear(self.output_num[-2], 500),
#             nn.PReLU(500),
#             nn.Dropout(p=0.2),
#             nn.Linear(500, 500),
#             nn.PReLU(500),
#             nn.Dropout(p=0.2),
#             nn.Linear(500, 20),
#             # nn.Linear(500, input_num),
#             )
        
#         self.gru_matrix = Parameter(
#             torch.zeros((20, int(self.output_num[-1] / 3), 3)))
#         nn.init.xavier_normal_(self.gru_matrix)

#     def forward(self, x, out_size, hidden=None, device=torch.device("cpu")):

#         next_hidden = self.cell(x, hidden)

#         y = self.out_mlp(next_hidden)

#         out = torch.zeros((y.size(0), out_size, 3)).to(device)   
#         for i in range(y.size(0)): # 不能使用np.arange
#             y_i = y[i]
#             out_i = torch.einsum('a,abc->bc', y_i, self.gru_matrix)
#             out[i] = out_i

#         return out, next_hidden


# class SSModel(nn.Module):
#     def __init__(self,step_two_model, config, data):
#         super(SSModel, self).__init__()

#         self.model_name = ""
#         self.device = config['device']

#         self.data = data
#         self.cloth = data.cloth_v.to(self.device)
#         self.gru_hidden_size = 1000
#         self.gravity = torch.Tensor([0,-9.8,0],self.device).to(torch.float32)

#         self.joint_indices = self.filterJoint()
#         self.buildModel(step_two_model)


#     def filterJoint(self):
#         weight_sum = self.data.cloth_weights.sum(dim=0)
#         return torch.where(weight_sum>0)[0]

#     def buildModel(self,step_two_model, dropout = 0.2):
#         def fusionWeight(weights1,joint_weights, joint_weights_indices, joint_weights_index_weights):
#             weights = torch.zeros([weights1.shape[0], weights1.shape[1] + joint_weights.shape[1]],\
#                                   dtype=torch.float32, device=weights1.device)
#             weights[:,:weights1.shape[1]] = weights1
#             weights[joint_weights_indices, :weights1.shape[1]] = torch.einsum("ij,i->ij",weights1[joint_weights_indices], (1. -joint_weights_index_weights))
#             weights[joint_weights_indices, weights1.shape[1]:] = torch.einsum("ij,i->ij",joint_weights, joint_weights_index_weights)
#             return weights
        
#         def disableParameter(module):
#             for param in module.parameters():
#                 param.requires_grad_(False)

#         body_encoder1       = MLP(dimensions = [16,32], dropout= dropout)
#         flatten             = nn.Flatten(2,-1)
#         body_encoder2       = MLP(dimensions = [self.joint_num*16, 512, 512], dropout = dropout)
#         transform_encoder   = nn.Sequential(flatten, body_encoder2)
#         transform_decoder   = MLP(dimensions = [512, self.joint_num* 16],dropout= dropout)

#         vertices_encoder    = nn.Sequential(
#             nn.Flatten(2,-1),
#             MLP(dimensions = [self.joint_num*16, 512, 512], dropout = dropout)
#             )
#         vertices_decoder    = PSD(512, self.cloth.shape)

#         self.joint_trans_module = step_two_model.trans_module
#         disableParameter(self.joint_trans_module)


#         self.cloth_joint_parent_bind_matrices = step_two_model.lbs_module.b_j_bind_matrices
#         self.cloth_joint_parent_indices = step_two_model.lbs_module.parent_indices
    
#         weights = fusionWeight(
#             self.data.cloth_weights[:,self.joint_indices].to(self.device),
#             step_two_model.lbs_module.weights,
#             step_two_model.loss_indices,
#             step_two_model.loss_indices_weight
#         )
#         weights.requires_grad_(False)
#         bind_matrix = torch.cat([self.data.body_offset_matrix[self.joint_indices].to(self.device),
#                                 step_two_model.c_c_j_bind_matrix], dim = 0)

#         lbs = LBSLayer(weights,
#                        bind_matrix,
#                        train_weight = False
#                        )
        
#         self.trans_module = nn.Sequential(transform_encoder,transform_decoder)
#         self.vt_module = nn.Sequential(vertices_encoder,vertices_decoder)
#         self.lbs_module = lbs
#         return

        
#     @property
#     def joint_num(self):
#         return len(self.joint_indices) + self.data.c_j_weights.shape[1]

#     def preprocess_data(self, body_transforms):
#         # encode velocity and gravity
#         pass

#     def process_data(self, body_transforms):

#         # encode velocity and gravity
#         joints_pos = body_transforms[...,:3,3]
#         joints_vecolity = getDerivative(joints_pos, 1, dt=1.0/30)

#         body_transforms = body_transforms[:,1:]

#         posed_vecolity = torch.einsum("bsjno,bsjno->bsjno", body_transforms, joints_vecolity)

#         posed_gravity = torch.zeros_like(joints_pos)

#         prev_transform  = body_transforms[:, :-1]
#         cur_transform   = body_transforms[:, 1:]

#         vecolity = body_transforms[:,:, self.joint_indices]
#         unpose_vecolity = torch.einsum("bsjno,bsjno->bsjno", body_transforms, vecolity)

#         pass

    

#     def network_forward(self,  body_features):
#         x_lbs_cloth_bone  = x_lbs_cloth_bone + cloth_delta
#         return x_lbs_cloth_bone

#     def forward(self,prev_cloth_state, body_transforms1, hip_translations):

#         return self.network_forward(body_features, body_transform)