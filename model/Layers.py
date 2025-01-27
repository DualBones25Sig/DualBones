import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing

from .utils.lbs import lbs,normalizeSafe


class BatchSeqNormal(nn.Module):
    """
    自定义模块，用于在三维数据上应用 BatchNorm1d
    """
    def __init__(self, num_features, momentum=1e-3):
        super(BatchSeqNormal, self).__init__()
        self.bn = nn.BatchNorm1d(num_features, momentum=momentum)

    def forward(self, x):
        batch, seq, feature = x.size()
        x = x.view(batch * seq, feature)
        x = self.bn(x)
        x = x.view(batch, seq, feature)
        return x


def MLP(dimensions, dropout=0.5, batch_norm = False):
    layers = []
    for i in range(1, len(dimensions)):
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dimensions[i - 1], dimensions[i]))
        layers.append(nn.PReLU())
        layers.append(BatchSeqNormal(dimensions[i], momentum=1e-3) if batch_norm else nn.Identity())
    return nn.Sequential(*layers)


class GRU_Model(nn.Module):
    def __init__(self, encode_dim, decode_dim, dropout=0.5):
        super(GRU_Model, self).__init__()
        
        # self.flatten = nn.Flatten(-2,-1)
        self.hidden_size = decode_dim[0]
        self.mlp_encoder = MLP(encode_dim, dropout)
        self.gru = nn.GRU(input_size = encode_dim[-1],
                          hidden_size = self.hidden_size,
                          batch_first = True)
        self.mlp_decoder = MLP(decode_dim, dropout)

    def forward(self, input, hidden):
        # batch sequence joint
        # x = self.flatten(input)
        x = self.mlp_encoder(input)
        # return self.mlp_decoder(x), hidden
        out_put, next_hidden = self.gru(x, hidden)
        x = self.mlp_decoder(out_put) 

        return x, next_hidden




class MLPDec(nn.Module):
    def __init__(self, in_channels, out_shape, dropout=0.5):
        super(MLPDec, self).__init__()
        self.MLP = MLP([in_channels,out_shape[0]*out_shape[1]], dropout=dropout)
        self.out_shape = out_shape
        # self.psd = nn.Parameter(torch.empty([in_channels,out_shape[0] * 3], dtype=torch.float32))
        # nn.init.xavier_normal_(self.psd)

    def forward(self, x):
        out = self.MLP(x)
        return out.view(*out.shape[:-1], *self.out_shape)


class PSD(nn.Module):
    def __init__(self, in_channels, cloth_shape, dropout=0.5):
        super(PSD, self).__init__()
        
        self.drop_out = nn.Dropout(dropout)
        self.psd = nn.Parameter(torch.empty([in_channels,cloth_shape[0] * 3], dtype=torch.float32))
        
        nn.init.xavier_normal_(self.psd)

    def forward(self, x):
        x = self.drop_out(x)
        return torch.matmul(x,self.psd).view(*x.shape[:-1],-1,3)



class PostTranslation(nn.Module):
    def __init__(self,
                 joint_weights: torch.Tensor,
                 bone_weights: torch.Tensor,
                 train_weights = False,
                #  mask_weights = True,
                 **kwargs):
        super().__init__(**kwargs)

        self.train_weights = train_weights

        weights = bone_weights.T @ joint_weights

        self.setWeights(weights)

    def setWeights(self,weights):
        w = normalizeSafe(weights)
        self.param_mask = (w > 0).to(torch.int32)
        self.weights = nn.Parameter(w)

    def getWeights(self):
        if self.train_weights:
            return self.weights * self.param_mask
        else:
            return self.weights

    def forward(self, joint_translation, targetTransforms):
        targetTransforms[...,6:9] = targetTransforms[...,6:9] + torch.matmul(self.getWeights(), joint_translation)
        return targetTransforms