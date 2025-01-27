import sys
import torch
import os



def loadCheckPoint(path:str,
                   device):
    
    assert os.path.exists(path), f"Error! Load checkpoint failed, {path} not exists"
    checkpoint = torch.load(path, map_location=device)
    return checkpoint


def getModelSize2(model:torch.nn.Module, detail = False):
    # 计算模型参数数量, 假设均为float
    print("-------------------model  information----------------------")
    if detail:
        for name, param in model.named_parameters():
            print(name,",para num:",param.numel(),"para size:",4 * param.numel() /1024 / 1024,"MB")

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params,",Model size:",4 * total_params /1024 / 1024, "MB")



def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('model size: {:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)



import numpy as np

class StepOneResult:
    def __init__(self) -> None:
        self.cloth_vertices = np.zeros([1,3],dtype=np.ndarray)
        self.cloth_faces = np.zeros([1,3],dtype=np.ndarray)

        self.joint_names = []
        self.offset_matrix = np.zeros([1,16],dtype=np.ndarray)
        self.weights = np.zeros([1,1], dtype=np.ndarray)

        self.transforms = {}
        return

    def SetInfo(self,vertices:np.ndarray, faces:np.ndarray,joint_names:list, offset_matrix:np.ndarray, weights:np.ndarray):
        self.cloth_vertices = vertices
        self.cloth_faces = faces

        self.joint_names = joint_names
        self.offset_matrix = offset_matrix
        self.weights = weights

        return

    def Add(self, folder_name : str, transforms  : np.ndarray):
        self.transforms[folder_name] = transforms
        return

    def saveToFile(self, file_name:str):
        npz_data = {}
        
        npz_data['vertices'] = self.cloth_vertices
        npz_data['faces'] = self.cloth_faces
        npz_data['joint_names'] = np.array(self.joint_names)
        npz_data['offset_matrix'] = self.offset_matrix
        npz_data['weights'] = self.weights

        npz_data.update(self.transforms)

        np.savez(file_name, **npz_data)
        pass

    def loadFromFile(self, file_name:str):
        npz_data = np.load(file_name)
        self.cloth_vertices = npz_data['vertices']
        self.cloth_faces = npz_data['faces']
        self.joint_names = npz_data['joint_names']
        self.offset_matrix = npz_data['offset_matrix']
        self.weights = npz_data['weights']

        filter_key = ['vertices','faces','joint_names','offset_matrix','weights']
        self.transforms = {key: value for key, value in npz_data.items() if key not in filter_key}
        return


    def getCloth(self):
        return self.cloth_vertices, self.cloth_faces
    
    def getJointNames(self):
        return self.joint_names

    def getOffsetMatrix(self)->np.ndarray:
        return self.offset_matrix
    
    def getWeights(self):
        return self.weights

    def getTransforms(self, folder_name : str) -> np.ndarray:
        return self.transforms[folder_name]


    def __LBSExampel():
        '''
        A example of lbs
        '''
