
import torch 
from model.utils.meshUtil import *
from scipy.spatial import cKDTree

class SupervisedLosses():
    def __init__(self, v, f, lap_w, col_w, device) -> None:
        def laplacian_operator(vert, face):
            num_vert = len(vert)
            laplacian_matrix = torch.zeros((num_vert, num_vert), dtype=torch.float32)
            edges = set()
            for f in face:
                edges.add(tuple(sorted((f[0], f[1]))))
                edges.add(tuple(sorted((f[1], f[2]))))
                edges.add(tuple(sorted((f[2], f[0]))))
            edge_tensor = torch.tensor(list(edges), dtype=torch.long)
            i, j = edge_tensor[:, 0], edge_tensor[:, 1]
            laplacian_matrix[i, i] += 1
            laplacian_matrix[j, j] += 1
            laplacian_matrix[i, j] -= 1
            laplacian_matrix[j, i] -= 1
            return laplacian_matrix
        self.lap_w = lap_w
        if self.lap_w > 0:
            self.lap_mat = laplacian_operator(v,f).to(device)

        self.col_w = col_w

        self.MSE = torch.nn.MSELoss()

    def _getL2Loss(self, x, g_t):
        return torch.norm(x-g_t,dim=-1).mean()
        

    def _getLaplacianLoss(self, x, g_t):
        x_lap = torch.einsum("ij,...jk->...ik", self.lap_mat, x)
        pred_lap = torch.einsum("ij,...jk->...ik", self.lap_mat, g_t)


        return torch.norm(x_lap-pred_lap,dim=-1).mean()
    

    def _getCollisionLosses(self, x, b_v, b_f, threshold = 4e-5):
        body_vertex_normal = getVertexNormals(b_v, b_f.to(b_v.device))
        c_s = x.shape

        cloth_v = x.view(-1,*x.shape[-2:]).detach().cpu().numpy()
        body_v = b_v.view(-1,*b_v.shape[-2:]).cpu()
        tmp_dis = []
        tmp_ind = []
        for i in range(cloth_v.shape[0]):
            d, i = cKDTree(body_v[i]).query(cloth_v[i],workers=-1)
            tmp_dis.append(d)
            tmp_ind.append(i)
        # dis = np.array(tmp_dis).reshape(c_s[:-1])
        ind = np.array(tmp_ind).reshape(c_s[:-1])
        
        loss = []
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                col_i = ind[i][j]
                b_n = body_vertex_normal[i,j,col_i]
                b_c_vec = x[i,j] - b_v[i,j][col_i]
                norm_dist = torch.einsum('ab,ab->a',b_n,b_c_vec)
                cur_loss = torch.clamp(threshold - norm_dist, min= 0.)
                loss.append(torch.mean(cur_loss.view(-1)))
        return torch.mean(torch.stack(loss))

    
    def computeRMSE(self, x, g_t):
        return torch.sqrt(torch.mean(torch.sum((x - g_t) ** 2, dim=-1), dim=-1)).mean()

    def getLosses(self, x, g_t, b_v=None, b_f=None):
        l2_loss = self._getL2Loss(x,g_t)
        tot_loss = l2_loss

        losses = [l2_loss]

        losses.append(torch.tensor(0.0,dtype=torch.float32,device=x.device) if self.lap_w == 0 else  self._getLaplacianLoss(x,g_t))
        tot_loss = tot_loss + self.lap_w * losses[-1]

        losses.append(torch.tensor(0.0,dtype=torch.float32,device=x.device) if self.col_w == 0 else  self._getCollisionLosses(x,b_v,b_f))        
        tot_loss = tot_loss + self.col_w * losses[-1]

        return tot_loss, *losses
    
    def computeMSE(self,x, g_t):
        return self.MSE(x,g_t)
    




def computeRMSE(self,x:torch.Tensor, ground_truth: torch.Tensor):
    return torch.sqrt(torch.mean((x - ground_truth) ** 2))

def computeHausdorffDistance(self,x:torch.Tensor, ground_truth: torch.Tensor):
    x_shape = x.shape
    x = x.reshape(-1, *x_shape[-2:])
    ground_truth = ground_truth.reshape(-1, *x_shape[-2:])

    HD = []
    for i in range(x.shape[0]):
        diff = x[i].unsqueeze(0) - ground_truth[i].unsqueeze(1)
        dist_matrix  = diff.norm(dim=2)
        pred_to_gt_min = dist_matrix.min(dim=1).values
        gt_to_pred_min = dist_matrix.min(dim=0).values
        d_ab = pred_to_gt_min.max()
        d_ba = gt_to_pred_min.max()
        HD.append(max(d_ab,d_ba))
    return torch.stack(HD).mean()




def computeSTED(self,x:torch.Tensor, ground_truth: torch.Tensor, edges):
    x_shape = x.shape
    x = x.reshape(-1, *x_shape[-2:])
    ground_truth = ground_truth.reshape(-1, *x_shape[-2:])

    e0 = edges[:, 0]
    e1 = edges[:, 1]
    pred_e0 = x[:, e0, :]
    pred_e1 = x[:, e1, :]
    gt_e0   = ground_truth[:, e0, :]
    gt_e1   = ground_truth[:, e1, :]
    pred_diff = pred_e0 - pred_e1  # (T, E, 3)
    gt_diff   = gt_e0   - gt_e1    # (T, E, 3)
    pred_len = pred_diff.norm(dim=2)  # (T, E)
    gt_len   = gt_diff.norm(dim=2)    # (T, E)
    length_diff  = (pred_len - gt_len).abs()
    return length_diff.mean()
