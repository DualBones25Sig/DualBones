
import os
import torch
import numpy as np
from model.utils.lbs import lbs
from .o3dViewer import getO3dMesh,getO3dWeightMesh,getO3dText,o3dRender



class ClothMesh:
    '''
    joint is body joint
    bone is cloth virtual bone
    mixed_bone is the mixed of joints and bones
    '''
    def __init__(self,
                 name:str,
                 vertices:torch.Tensor,
                 faces:torch.Tensor,
                 joint_names:list,
                 joint_weights:torch.Tensor,
                 joint_offset_matrix:torch.Tensor,
                 cache_dir:str = "./model/precomputed"
                 ):
        self.device = vertices.device
        self.name = name
        self.vertices = vertices
        self.faces = faces

        self.cache_dir = cache_dir

        self.render = True

        # filter joints first
        self.joint_indices = torch.where(joint_weights.sum(dim=0)>0)[0]
        self.joint_names = np.array(joint_names)
        self.joint_weights = joint_weights
        self.joint_offset_matrix = joint_offset_matrix

    @property
    def edges(self):
        if self.faces.shape[-1] == 3:
            return self.faces[:, [0, 1, 1, 2, 2, 0]].flatten().reshape(-1, 2)
        else:
            return self.faces[:, [0,1,1,2, 2,3,3,0]].flatten().reshape(-1, 2)


    def getJointOnlySkinning(self):
        return self.joint_weights, self.joint_offset_matrix

    def getLooseOnlySkinning(self):
        # loose skinning is not for the entire cloth, but for the loose vertices
        return self.loose_v_indices, self.bone_weights, self.bone_offset_matrix

    def getLooseOnlyMesh(self):
        index_map = torch.full((self.vertices.shape[0],), -1, dtype=torch.long)
        index_map[self.loose_v_indices] = torch.arange(self.loose_v_indices.shape[0], dtype=torch.long)
        mask = (index_map[self.faces] != -1).all(dim=1)
        # o3dRender([getO3dMesh(self.vertices[self.loose_v_indices].cpu().numpy(),index_map[self.faces[mask]].cpu().numpy())])
        return self.vertices[self.loose_v_indices], index_map[self.faces[mask]]
    
    def getMixedSkinning(self):
        return self.mixed_weights, self.mixed_offset_matrix
    
    def getMixedJointNames(self):
        return np.concatenate([self.joint_names[self.joint_indices],\
            np.array([f'ClothBone{ind:05d}' for ind in range(self.mixed_weights.shape[1] - len(self.joint_indices))])])

    def renderWeights(self):
        interval = 0.5
        render_list = []
        v= self.vertices.cpu().numpy()
        f = self.faces.cpu().numpy()

        v_color = torch.cat([self.loose_propotion.unsqueeze(1), 1. - self.loose_propotion.unsqueeze(1), ((torch.eq(self.loose_propotion, 0) | torch.eq(self.loose_propotion, 1)).float()).unsqueeze(1)],dim=-1)
        _m = getO3dMesh(v + np.array([-interval,0,0]),f, v_color)
        render_list.append(_m)

        # render weights after Heat propation
        cloth_bone_pos = -self.bone_offset_matrix[:,:3,3]
        render_weights = np.zeros([v.shape[0],cloth_bone_pos.shape[0]],dtype=np.float32)
        render_weights[self.loose_v_indices] = self.bone_weights.cpu().numpy()
        _m = getO3dWeightMesh(v + np.array([interval,0,0]),cloth_bone_pos + np.array([interval,0,0]), render_weights, f)
        render_list += _m
        o3dRender(render_list)


    def computeMixedSkinning(self, transforms:list, ground_truth:list, num_bones:int,\
                             therthold, scale = 25, feather:int = 7):
        self.num_bones = num_bones

        file_name = os.path.join(self.cache_dir,f"{self.name}_AB_info_B{num_bones}_{therthold}_{scale}_{feather}.npz")
        if os.path.exists(file_name):
            print("load weights from file")
            data = np.load(file_name)
            mix_propotion = torch.from_numpy(data['propotion'])
            bone_weights = torch.from_numpy(data['weights'])
            bone_offset_matrix = torch.from_numpy(data['offset'])
            loose_v_indices = torch.where(mix_propotion>0)[0]
        else:
            mix_propotion, error = self._computeMixPropotion(transforms,ground_truth, therthold, scale)
            bone_weights, bone_offset_matrix = self._computeBoneWeights(num_bones, torch.clamp(error-therthold,0.0,0.6)**0.5, feather)
            # compute loose vertices and filter skinning by loose vertices
            loose_v_indices = torch.where(mix_propotion>0)[0]

            bone_weights = bone_weights[loose_v_indices] # filter loose indices (if have)
            w_bones_sum = bone_weights.sum(dim=0)

            choose_bones = torch.where(w_bones_sum>1e-4)[0]
            bone_weights = bone_weights[:, choose_bones] # filter unused bones (if have)
            bone_weights = bone_weights / bone_weights.sum(dim=1, keepdim=True)
            bone_offset_matrix = bone_offset_matrix[choose_bones]
            np.savez(file_name, propotion = mix_propotion.numpy(), weights = bone_weights.numpy(), offset = bone_offset_matrix.numpy())

        self.loose_v_indices = loose_v_indices
        self.loose_propotion = mix_propotion
        self.bone_weights = bone_weights
        self.bone_offset_matrix = bone_offset_matrix
        
        if self.render:
            self.renderWeights()

        # step3: mix the skinning info        
        j_o_m = self.joint_offset_matrix[self.joint_indices]
        self.mixed_weights = self._mixWeights()
        self.mixed_offset_matrix = torch.cat([j_o_m.to(self.device), self.bone_offset_matrix], dim = 0)

        return
    
    def UpdateBoneWeights(self, bone_weights:torch.Tensor):
        self.bone_weights = bone_weights
        self.mixed_weights = self._mixWeights()
        return
    
    def _mixWeights(self):
        j_w = self.joint_weights[:,self.joint_indices]
        losse_v_mix_propotion = self.loose_propotion[self.loose_v_indices]
        mixed_weights = torch.zeros([j_w.shape[0], j_w.shape[1] + self.bone_weights.shape[1]],\
                                dtype=torch.float32, device=self.device)
        mixed_weights[:,:j_w.shape[1]] = j_w
        mixed_weights[self.loose_v_indices, :j_w.shape[1]] = torch.einsum("ij,i->ij",j_w[self.loose_v_indices], (1. -losse_v_mix_propotion))
        mixed_weights[self.loose_v_indices, j_w.shape[1]:] = torch.einsum("ij,i->ij",self.bone_weights, losse_v_mix_propotion)
        return mixed_weights


    def _computeMixPropotion(self,transforms, ground_truth,therthold, scale):
        '''
        compute the propotion of cloth bones in the cloth
        value: [0, 1], larger value means this vertex is need loose (need more optimized)
        '''
        assert len(ground_truth) >0, "no data to compute mix propotion"
        lbs_vertices_error = analyzeClothByGroundTruth( self.vertices,
                                                        self.joint_offset_matrix,
                                                        self.joint_weights,
                                                        transforms,
                                                        ground_truth)

        error = (lbs_vertices_error - therthold) 
        propotion = torch.clamp(error * scale, min=0.0, max=1.0)

        return propotion, lbs_vertices_error
    
    def getMeshDescibe(self):
        return f"{self.name}_v{self.vertices.shape[0]}_f{self.faces.shape[0]}"
    
    def printDescribe(self):
        '''
        '''
        print("--------------------cloth  information----------------------")
        print("|| cloth name:", self.name, ", v",self.vertices.shape[0], ", f", self.faces.shape[0])
        print(f"|| bone num: {self.bone_weights.shape[1]}, joint num: {len(self.joint_indices)}({len(self.joint_names)} totaly)")
        print(f"|| loose: v num:{len(self.loose_v_indices)}({self.vertices.shape[0]} totaly)")
        pass

    def _computeBoneWeights(self,num_bones,loose_weights, feather):
        print("Initializing joints and weights")
        cloth_bones_pos,_WCMF_w     = initializeBonesByWFCM(\
            self.vertices.numpy(),self.faces.numpy(),loose_weights.numpy(),num_bones=num_bones)
        weights                     = initWeightsByHeatPropagation2(\
                                        self.vertices,
                                        self.faces,
                                        cloth_bones_pos,
                                        feather)

        cloth_bone_weights   = torch.from_numpy(weights).to(torch.float32).to(self.device)
        cloth_bone_offset_matrix = torch.eye(4).repeat(cloth_bones_pos.shape[0],1,1).to(torch.float32).to(self.device)
        cloth_bone_offset_matrix[:,:3,3] = -torch.from_numpy(cloth_bones_pos).to(torch.float32).to(self.device)

        return cloth_bone_weights, cloth_bone_offset_matrix




def InitializeWeightsByDistance(src_positions, des_positions, src_weight):
    """
    copy weight from nearest position
    """
    from scipy.spatial import KDTree
    des_weight = torch.empty([des_positions.shape[0],src_weight.shape[1]], dtype=torch.float32)
    distances, indices = KDTree(src_positions).query(des_positions)
    return src_weight[indices]


from psbody.mesh import Mesh as Obj
import os

import warnings


def initializeBonesByWFCM(vertices:np.array,
                            faces:np.array,
                            loose_error:np.array,
                            num_bones = 0.5,
                            is_joints_vertices = False):
    from .WFCM import WFCM

    l_v_indices = np.where(loose_error > 1e-4)[0]
    l_v = vertices[l_v_indices]
    l_w = loose_error [l_v_indices]
    # l_w = l_w ** 4
    bones, M = WFCM(l_v,num_bones,m_i=4,weights=l_w, loose_index=l_v_indices)

    weights = np.zeros([vertices.shape[0],num_bones],dtype=np.float32)
    weights[l_v_indices] = M
    return bones.astype(np.float32), weights






def initializeBonesByKmeans(vertices:np.array,
                            faces:np.array,
                            num_bones = 0.5,
                            is_bones_vertices = False):
    from scipy.cluster.vq import vq, kmeans, whiten

    # use kmeans to initailze bones
    vertices = vertices
    simplify_ratio = 1. * num_bones / vertices.shape[0]
    # num_bones = int(vertices.shape[0] * num_bones)

    whitened = whiten(vertices)
    codebook, _ = kmeans(whitened, num_bones)
    vq_code, vq_dist = vq(whitened, codebook)

    translation = np.empty([num_bones,3],dtype=np.float32)

    weights = np.zeros([vertices.shape[0],num_bones],dtype=np.float32)

    for bone_ind in range(num_bones):
        ind_array = vq_code == bone_ind
        weights[ind_array, bone_ind] = 1.0
        if is_bones_vertices:
            select_v_ind = np.flatnonzero(ind_array)[np.argmin(vq_dist[ind_array])]
            # if vq_dist[select_v_ind] > 5e-2: print("Warning: bone distance:",vq_dist[select_v_ind])
            translation[bone_ind] =  vertices[select_v_ind]
        else:
            translation[bone_ind] = np.mean(vertices[ind_array],axis=0)
            # translation[bone_ind] = np.mean(np.flatnonzero(ind_array))

    t = np.array(translation)

    return t, weights




def test_lbs(   vertices:torch.Tensor,
                cloth:Obj,
                offset_matrix,
                weights,
                transforms:list,
                ground_truth:list,
                threshold = 0.1):
    tmp_cloth_v = cloth.v.copy()
    

    for ind,t in enumerate(transforms):
        v = lbs(vertices,t.view(*t.shape[:-1], 4,4),weights,offset_matrix)
        
        for i in range(v.shape[0]):
            cloth.v = v[i].cpu().numpy()
            cloth.write_obj("./tmp/tmp_lbs/tmp%07d"% i+".obj")
            cloth.v = ground_truth[ind][i].cpu().numpy()
            cloth.write_obj("./tmp/tmp_gt/tmp%07d"% i+".obj")

        cloth.v = tmp_cloth_v

        return
    



def analyzeClothByGroundTruth( vertices:torch.Tensor,
                                offset_matrix,
                                weights,
                                transforms:list,
                                ground_truth:list,
                                threshold = 0.1):
    vertices_error = torch.zeros([vertices.shape[0]],dtype=torch.float32,device=vertices.device)
    data_len = 0


    for ind,t in enumerate(transforms):
        v = lbs(vertices,t.view(*t.shape[:-1], 4,4),weights,offset_matrix)
        vertices_error += torch.sum(torch.norm(v - ground_truth[ind],dim=-1),dim=0)
        data_len += t.shape[0]

    vertices_error /= data_len

    return vertices_error




def initWeightsByHeatPropagation2(  vertices: np.array,
                                    faces: np.array,
                                    heat_source,
                                    feather:int,
                                    max_influencer:int = 4,
                                    ):
    '''
    not port pattern
    Para:
        fixed_heat_weights: if set True, all heats should be source mesh vertices, and the weights of the vertices will set to 1.0 
        feather:            [1,10];  Heat decay rate, higher rate will give softer junction
        max_influencer:     One vertx can be influence by most four heatOrigin
    '''

    import potpourri3d as pp3d
    from psbody.mesh import Mesh as Obj
    from scipy.spatial import KDTree
    from tqdm import tqdm

    def normalized_weight(weight):
        row_sums = weight.sum(axis=1)
        if min(row_sums) == 0.:
            print("warning: these vtx has no wegiht:", end= " ")
        for i in range(weight.shape[0]):
            if row_sums[i] != 0.:
                weight[i, :] = weight [i, :] / row_sums[i]
            else:
                print(i,end=",")
        print("")
        return weight


    source_mesh = Obj(v=vertices, f = faces)
    # cache file name

    mesh_v = source_mesh.v
    mesh_f = source_mesh.f
    mesh_v_num = source_mesh.v.shape[0]
    
    heat_source = heat_source
    heat_source_num = heat_source.shape[0]


    # for every vertex of simplified_mesh，get corresponding index in mesh
    mesh_kdtree = KDTree(mesh_v)
    heat_dist , heat_ind = mesh_kdtree.query(heat_source)

    s_m_weight = np.zeros([mesh_v_num,heat_source_num],dtype=float)
    dist_solver = pp3d.MeshHeatMethodDistanceSolver(mesh_v,mesh_f)

    feather = 8
    j = np.clip((11-feather),1,10)
    for i,(h_i, h_d) in enumerate(zip(heat_ind, heat_dist)):

        dist = dist_solver.compute_distance(h_i)
        # o3dRender([getO3dMesh(mesh_v,mesh_f,np.concatenate([dist[:, np.newaxis], dist[:, np.newaxis], dist[:, np.newaxis]], axis=1)*2)]) # debug only

        d = dist + 0.005 * feather  # add value makes the weight more smooth
        d = (3 * d) ** (1 + 0.5 * j) # j control the weights delay speed
        d = np.clip(d , 1e-4, 1e5)
        s_m_weight[:,i] = 1/d

        s_m_weight[np.isinf(dist), i] = 0
    
    # select influencer
    filtered_weights = np.zeros_like(s_m_weight)
    for i in range(s_m_weight.shape[0]):
        row_indices = np.argsort(s_m_weight[i])[-4:]
        filtered_weights[i, row_indices] = s_m_weight[i, row_indices]

    weights = normalized_weight(filtered_weights)
    # o3dRender(getO3dWeightMesh(mesh_v, heat_source, weights, mesh_f)) # debug only

    assert len(np.where(np.isnan(weights))[0]) == 0, f"nan in weights, pos {np.where(np.isnan(weights))[0].tolist()}"

    return weights

