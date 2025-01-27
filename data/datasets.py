import os
import numpy as np
import torch
from torch.utils.data import Dataset
from psbody.mesh import Mesh as Obj
from torch_geometric.data import Data as graphData
from scipy.spatial.transform import Rotation as transRot
from .utils.clothMesh import ClothMesh

import sys
sys.path.append('.')

class ModelDataset(Dataset):
    def __init__(self,config_data, mode = "train"):
        self.config_data = config_data
        self.cloth_name =config_data['cloth']['cloth']

        
        self.use_local_transforms = False
        self.swap_YZ = False
        self.transpose_transform = True
        


        self.data_folder = config_data['train_folders' if mode == 'train' else 'test_folders']


        # property of body offset_matrix
        self._initBodyStaticData()
        if mode == 'train':
            self._initClothStaticData()


        transform_files = [os.path.join(config_data['path'], f, "transform.npz") for f in self.data_folder]
        self.data_folder = [t_d_f for t_d_f,b_t_f in zip(self.data_folder,transform_files) if os.path.exists(b_t_f)]
        transform_files = [f for f in transform_files if os.path.exists(f)]

        # simplify it
        self.preloaded = False
        self.hip_translation = []
        self.body_transforms = []
        self.body_vertices = []
        self.ground_truth = []

        self.prefix_lenths = []

        data_folder = []
        # load data
        print("Loding Data")
        for i in range(len(transform_files)):
            cloth_path = os.path.join(self.config_data['path'],self.data_folder[i],self.cloth_name)
            if os.path.exists(cloth_path) or os.path.exists(cloth_path+".npz"):
                transforms, vertices , hip_translations = self._loadTransformFile(transform_files[i])
                if "dress0" in self.cloth_name  and transforms.shape[0] != 500:
                    print(f"not 500 frames in {self.data_folder[i]}(skip)",end=",")
                    continue

                self.body_transforms.append(transforms.view(transforms.shape[0],-1,16))
                self.body_vertices.append(vertices)
                self.hip_translation.append(hip_translations)
                self.ground_truth.append(self._loadGroundTruth(os.path.join(self.config_data['path'],self.data_folder[i],self.cloth_name)))
                print(f"{self.data_folder[i]}:{len(transforms)}", end = ", ")

                self.prefix_lenths.append((0 if len(data_folder)==0 else self.prefix_lenths[-1]) + self.body_transforms[-1].shape[0])
                data_folder.append(self.data_folder[i])
            else:
                print(f"{self.data_folder[i]} have no ground truth, skip")
        print()

        self.data_folder = data_folder

        num_bones = self.config_data['num_bones']

        if mode == 'train':
            self.cloth.computeMixedSkinning(self.body_transforms, self.ground_truth, num_bones,therthold=config_data['thertholde'])
            self.cloth.printDescribe()
        return

         
    def _initBodyStaticData(self):
        body_information = np.load(os.path.join(self.config_data['path'],'body_info.npz'))
        body_joints = body_information['joints'].tolist()
        self.body_select_joint_indeices = [body_joints.index(j) for j in body_information['select_joints']]
        self.body_select_joints = body_information['joints'][self.body_select_joint_indeices].tolist()

        self.hip_index = body_information['select_joints'].tolist().index(body_information['hip_joint']) if 'hip_joint' in body_information else 0

        # body informatino
        self.body_v                 = torch.from_numpy(body_information['vertices']).to(torch.float32)
        self.body_f                 = torch.from_numpy(body_information['faces'].astype(np.int64)).to(torch.long)
        self.body_collision_indices = torch.arange(0,self.body_v.shape[0],dtype=torch.long)
        self.body_weights           = torch.from_numpy(body_information['weights'][:,self.body_select_joint_indeices]).to(torch.float32)
        self.body_weights = self.body_weights / torch.sum(self.body_weights, dim=1, keepdim=True)

        self.body_offset_matrix       = torch.from_numpy(body_information['offset'][self.body_select_joint_indeices]).reshape(-1,4,4).to(torch.float32)
        self.gravity                = torch.tensor([0,-9.8,0],dtype=torch.float32)
        self.v_offset               = torch.from_numpy(body_information['hip_offset']).to(torch.float32)


    def _initClothStaticData(self):
        cloth_information = np.load(os.path.join(self.config_data['path'],f"{self.cloth_name}_info.npz"))

        vertices = torch.from_numpy(cloth_information['vertices']).float()
        faces = torch.from_numpy(cloth_information['faces']).int()

        if 'weights' not in cloth_information:
            from scipy.spatial import cKDTree
            tree = cKDTree(self.body_v.numpy())
            _,idx = tree.query(vertices.numpy())
            joint_weights = self.body_weights[idx]
        else:
            joint_weights = torch.from_numpy(cloth_information['weights']).float()

        self.cloth = ClothMesh(self.cloth_name,
                               vertices,
                               faces,
                               self.body_select_joints,
                               joint_weights,
                               self.body_offset_matrix,
                               self.config_data['precompute_cache']
                               )


    def __len__(self):
        return len(self.body_transforms)

    def _loadGroundTruth(self, cloth_path):
        if os.path.exists(cloth_path+".npz"):
            G_T = torch.from_numpy(np.load(cloth_path+".npz")['vertices']).float()
        else:
            files = os.listdir(cloth_path)
            files.sort()
            files =  [os.path.join(cloth_path,f) for f in files if f.endswith('obj')]
            clothes = []
            for file in files:
                clothes.append(Obj(filename=file).v)
            G_T = torch.from_numpy(np.array(clothes)).float()
        return G_T

    def _loadTransformFile(self,file_name):
        npz_data = np.load(file_name)

        return torch.from_numpy(npz_data['transforms_local' if self.use_local_transforms else 'transforms']).to(torch.float32)[:,self.body_select_joint_indeices],\
                torch.from_numpy(npz_data['vertices']).to(torch.float32),\
                torch.from_numpy(npz_data['hip_translation'] if 'hip_translation' in npz_data else npz_data['trans']).to(torch.float32),\
    
    @property
    def data_info(self):
        return f"{self.cloth_name}_L{self.prefix_lenths[-1]}_MixedB{self.cloth.mixed_offset_matrix.shape[0]}"



    def __getitem__(self, index):
        return  self.body_transforms[index],\
                self.body_vertices[index],\
                self.hip_translation[index],\
                self.ground_truth[index]
