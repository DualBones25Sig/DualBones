import os
import time
import math

from datetime import datetime

import numpy as np
import torch

from data.datasets import ModelDataset
from data.dataLoader import ModelDatasetLoader
from model.step1_coarse_loose import CoarseLooseModel
from model.step2_fine_cloth import FineClothModel
from model.LBSMoule import LBSLayer
from model.utils.model import *
from model.utils.lbs import lbs
from model.utils.transformUtil import   transformRestore, transformSimplify

from tensorboardX import SummaryWriter
from tqdm import tqdm

from .LossFunc import SupervisedLosses

class Runner():
    def __init__(self, config, model_step, mode='train'):
        super().__init__()

        self.config     = config
        self.config_model = config[f'model{model_step}']
        self.device     = config['device']
        self.writer     = None
        self.training   = mode=='train'

        torch.set_num_threads(4)
        # deterministic
        torch.manual_seed(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.model_step   = model_step
        self.train_data   = ModelDataset(self.config['data'], mode='train')
        self.train_loader = ModelDatasetLoader(self.train_data, batch_size=self.config_model['batch_size'] if mode =="train" else 1, shuffle=True if mode == "train" else False)
        self.test_data    = ModelDataset(self.config['data'], mode='test')
        self.test_loader  = ModelDatasetLoader(self.test_data, batch_size=1, shuffle=False)


        self.c_model    = self.createCoarseModel()
        self.f_model    = self.createFineModel() if model_step == 2 else None
        self.lbs        = self.createLBS()
        self.optimizer  = self.createAdam()
        self.lbs_optimizer = self.createLBSOptimzer() if model_step == 1 else None
        self.scheduler  = self.createScheduler()
        self.S_loss_func = self.createLossFunc()
        self.tbqq_lenth = config['data']['tbqq_lenth']

        self.print_basic_info()
        self.pbar       = tqdm(range(0, self.config_model['epochs']),ncols=200)


    
    def createModelDatasetLoader(self,data,batch_size):
        return ModelDatasetLoader(data, batch_size=batch_size, shuffle=True)
                        
    def createAdam(self):
        config_Adam = self.config_model['optimizer']
        return torch.optim.Adam(self.c_model.parameters() if self.model_step == 1\
                                 else self.f_model.parameters(),
                                lr=config_Adam["lr"],
                                weight_decay=config_Adam["weight_decay"])
    
    def createLBSOptimzer(self, lr = 1e-5):
        return torch.optim.Adam(list(self.lbs.parameters()), lr=lr)


    def createScheduler(self):
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.3)

    def createCoarseModel(self):
        c_model = CoarseLooseModel(self.train_data.cloth, self.device).to(self.device)
        getModelSize2(c_model)
        return c_model
    
    def createLossFunc(self):
        if self.model_step == 1:
            loose_v, loose_f = self.train_data.cloth.getLooseOnlyMesh()
            return SupervisedLosses(loose_v,loose_f, lap_w= 5, col_w = 0, device=self.device)
        else:
            return SupervisedLosses(self.train_data.cloth.vertices,self.train_data.cloth.faces, lap_w= 5, col_w = 0, device=self.device)
    
    
    def createFineModel(self):
        if self.training:
            step1_save_path = os.path.join(self.model_save_path, self.getModelFilename(self.config_model['model1_epoch'],1))
            checkpoint = loadCheckPoint(step1_save_path, self.device)
            self.c_model.load_state_dict(checkpoint['model_state_dict'])
            self.train_data.cloth.UpdateBoneWeights(checkpoint['weights'].to('cpu'))
        else:
            print("predicting, please load coarse model and set lbs weights")
            print("predicting, please load fine model")

        f_model = FineClothModel(self.train_data.cloth, self.config_model).to(self.device)
        getModelSize2(f_model)
        return f_model


    def createLBS(self):
        if self.model_step == 1:
            self.lbs_src = self.train_data.cloth.vertices[self.train_data.cloth.loose_v_indices].to(self.device)
            return LBSLayer(self.train_data.cloth.bone_weights, \
                            self.train_data.cloth.bone_offset_matrix,\
                            train_weights=self.training,\
                            weights_optimizer=self.config['lbs_model']['weights_optimizer'],\
                            device = self.device)
        else:
            self.lbs_src = self.train_data.cloth.vertices.to(self.device)
            return LBSLayer(
                self.train_data.cloth.mixed_weights,
                self.train_data.cloth.mixed_offset_matrix,
                train_weights=False,
                device= self.device
            )

    @property
    def model_save_path(self):
        return self.config['model_save_path']

    @property
    def cur_epoch(self):
        return self.pbar.n
    
    def getModelFilename(self, epoch, model_step = -1):
        return f"S{self.model_step if model_step == -1 else model_step}_"+"epoch{:05d}".format(epoch) + ".pth"
    
    def print_basic_info(self):
        print("-------------------runner information----------------------")
        print("- model Step: ", self.model_step)
        print("- cloth name: ", self.train_data.config_data['cloth']['cloth'])
        print("- device    : ", self.device)
        print("- batch size: ", self.config_model['batch_size'])
        print("- tbqq lenth : ", self.train_data.config_data['tbqq_lenth'])
        print("- train data lenth: ", self.train_data.prefix_lenths[-1])
        print("- test data lenth: ", self.test_data.prefix_lenths[-1])
        print("-----------------------------------------------------------")

    def saveModel(self, totaly_loss = 0):
        model_save_name = os.path.join(self.model_save_path, self.getModelFilename(self.cur_epoch))
        if self.model_step == 1:
            torch.save({ 'model_state_dict': self.c_model.state_dict(),
                        'weights': self.lbs.getWeights(),
                        'epoch': self.cur_epoch,
                        'loss': totaly_loss
                        },model_save_name)
        else:
            torch.save({ 'model_state_dict': self.f_model.state_dict(),
                        'step_1_epoch': self.config_model['model1_epoch'],
                        'epoch': self.cur_epoch,
                        'loss': totaly_loss
                        }, model_save_name)


    def saveFullModel(self, filename:str):
        for d_ind,(cloth, body_transforms, body_seq_v, hip_translation, ground_truth) in enumerate(self.dataLoader):
            model_input = body_transforms[0][0].unsqueeze(0).unsqueeze(0).view(-1,body_transforms.shape[-1] * body_transforms.shape[-2]).to(self.device)
            if filename.endswith("onnx"):
                torch.onnx.export(
                    self.model,                    # Model to export
                    model_input,                        # Example input
                    filename,         # Output ONNX file name
                    input_names=['InputParams'],    # Input names (can be customized)
                    output_names=['OutputPredictions'],  # Output names (can be customized)
                    opset_version=11,         # ONNX operator set version
                )
            break
        pass



    def updateEpoch(self):
        self.scheduler.step()
        save_interval = [-1,50,50]
        self.pbar.update(1)
        if self.cur_epoch % save_interval[self.model_step] == 0:
            self.saveModel()
        _ind = self.cur_epoch // 5
        if self.cur_epoch % 5 == 0 and (_ind & (_ind - 1)) == 0 and _ind > 0:
            print()


    def writeError(self,loss_names, cur_error):
        if self.writer is None:
            now_t = datetime.now()
            # tensor_log_folder = self.train_data.data_info + "T{:02d}_{:02d}_{:02d}_{:02d}_".format(now_t.month, now_t.day, now_t.hour, now_t.minute)
            tensor_log_folder = self.train_data.data_info
            self.writer = SummaryWriter(os.path.join("./result/logs", tensor_log_folder, "coarse" if self.model_step==1 else "fine"))

        if isinstance(loss_names, str):
            self.writer.add_scalar(f"{loss_names} error", cur_error, self.cur_epoch)
        else:
            for ind,c_e in enumerate(cur_error):
                self.writer.add_scalar(f"{loss_names[ind]} error", c_e, self.cur_epoch)
    

    def writeObj(self):
        pass
    

    def train(self):
        if self.model_step == 1:
            self.trainCoarse()
            pass
        elif self.model_step == 2:
            self.trainFine()


    def trainCoarse(self):
        self.c_model.train()
        self.lbs.train()
        self.info_str = ""
        
        cur_train_losses = []

        for d_ind,(body_transforms, body_seq_v, hip_translation, ground_truth) in enumerate(self.train_loader):
            # transform : batch, seq, joint_num, 4, 4
            batch_size = body_transforms.shape[0]
            seq_lenth = body_transforms.shape[1]
            step_times = math.ceil(seq_lenth/self.tbqq_lenth)
            gt_loose = ground_truth[:, :, self.train_data.cloth.loose_v_indices]
            hip_translation = hip_translation - hip_translation[:,0:1]

            coarse_hidden = None

            for s_i in range(step_times):
                b_t = body_transforms[:, s_i*self.tbqq_lenth:(s_i+1)*self.tbqq_lenth].to(self.device)
                g_t = gt_loose[:, s_i*self.tbqq_lenth:(s_i+1)*self.tbqq_lenth].to(self.device)
                h_t = hip_translation[:,  s_i*self.tbqq_lenth:(s_i+1)*self.tbqq_lenth].to(self.device)
                
                self.optimizer.zero_grad()

                b_f_f, bone_trans, next_coarse_hidden = self.c_model(b_t,h_t,coarse_hidden)
                coarse_hidden =  next_coarse_hidden.detach()

                bone_trans_restore = transformRestore(bone_trans)
                # bone_trans_restore[...,:3,:3] = torch.matmul(b_t.view(*b_t.shape[:-1],4,4)[...,0:1,:3,:3], bone_trans_restore[...,:3,:3])
                predict = self.lbs(self.lbs_src, bone_trans_restore)

                S_loss, l2_loss, lap_loss, _ = self.S_loss_func.getLosses(predict, g_t)
                rmse = self.S_loss_func.computeRMSE(predict, g_t)

                if self.cur_epoch > 50:
                    self.lbs_optimizer.zero_grad()

                S_loss.backward()
                self.optimizer.step()

                if self.cur_epoch > 50:
                    self.lbs_optimizer.step()

                if self.lbs.train_weights and self.lbs.weights_optimizer == 'mask_lsq' and\
                      self.cur_epoch > 40 and (self.cur_epoch + 1) % 30 == 0:
                    self.lbs.lsqAddCache(bone_trans_restore, g_t)

                cur_train_losses.append([S_loss.item(), l2_loss.item(), lap_loss.item(), rmse.item()])

        train_loss = torch.mean(torch.tensor(cur_train_losses),dim=0)
        self.writeError("train_loss",train_loss[0])
        self.writeError("train_rmse",train_loss[3])

        if self.lbs.train_weights and self.lbs.weights_optimizer == 'mask_lsq' and\
                self.cur_epoch > 40 and (self.cur_epoch + 1) % 30 == 0:
            self.lbs.LsqUpdateWeights(self.lbs_src)
            self.lbs_optimizer = self.createLBSOptimzer()


        if self.cur_epoch % 10 ==0:
            self.testCoarse()

        self.info_str = f"train={train_loss[0]:.3e},l2:{train_loss[1]:.3e},rmse:{train_loss[3]:.3e},||\
              test={self.test_loss[0]:.3e},l2:{self.test_loss[1]:.3e},rmse:{self.test_loss[3]:.3e}, mse{self.test_loss[4]:.3e}"
        # self.info_str += f"loose_rmse{self.test_loss[5]:.3e}"
        self.pbar.set_postfix_str(self.info_str)

        self.updateEpoch()

    def testCoarse(self):
        cur_test_losses = []
        self.c_model.eval()

        with torch.no_grad():
            for d_ind,(body_transforms, body_seq_v, h_t, ground_truth) in enumerate(self.test_loader):
                h_t = (h_t - h_t[:,0:1])
                coarse_hidden = None
                b_t = body_transforms.to(self.device)
                g_t_loose = ground_truth[:, :, self.train_data.cloth.loose_v_indices]
                g_t = g_t_loose.to(self.device)
                h_t = h_t.to(self.device)

                b_f_f, bone_trans, next_coarse_hidden = self.c_model(b_t,h_t,coarse_hidden)
                # coarse_hidden =  next_coarse_hidden.detach()

                bone_trans_restore = transformRestore(bone_trans)
                # bone_trans_restore[...,:3,:3] = torch.matmul(b_t.view(*b_t.shape[:-1],4,4)[...,0:1,:3,:3], bone_trans_restore[...,:3,:3])
                predict = self.lbs(self.lbs_src, bone_trans_restore)

                S_loss, l2_loss, lap_loss, _= self.S_loss_func.getLosses(predict, g_t)
                rmse = self.S_loss_func.computeRMSE(predict, g_t)
                mse = self.S_loss_func.computeMSE(predict, g_t)
                cur_test_losses.append([S_loss.item(), l2_loss.item(), lap_loss.item(),rmse.item(), mse.item()])



            self.test_loss = torch.mean(torch.tensor(cur_test_losses),dim=0)
            self.writeError("test_loss",self.test_loss[0])
            self.writeError("test_rmse",self.test_loss[3])


    def trainFine(self):
        self.c_model.eval()
        self.f_model.train()
        self.lbs
        self.info_str = ""
        
        supervised_loss = []

        for d_ind,(body_transforms, body_seq_v, hip_translation, ground_truth) in enumerate(self.train_loader):
            # transform : batch, seq, joint_num, 4, 4
            batch_size = body_transforms.shape[0]
            seq_lenth = body_transforms.shape[1]
            step_times = math.ceil(seq_lenth/self.tbqq_lenth)
            hip_translation = hip_translation - hip_translation[:,0:1]

            coarse_hidden = None
            fine_hidden = None

            for s_i in range(step_times):
                b_t = body_transforms[:, s_i*self.tbqq_lenth:(s_i+1)*self.tbqq_lenth].to(self.device)
                g_t = ground_truth[:, s_i*self.tbqq_lenth:(s_i+1)*self.tbqq_lenth].to(self.device)
                h_t = hip_translation[:,  s_i*self.tbqq_lenth:(s_i+1)*self.tbqq_lenth].to(self.device)
                
                self.optimizer.zero_grad()
                with torch.no_grad():
                    b_f_f, bone_trans_s, next_coarse_hidden = self.c_model(b_t,h_t,coarse_hidden)
                    coarse_hidden =  next_coarse_hidden.detach()

                    bone_trans_restore = transformRestore(bone_trans_s)
                    # bone_trans_restore[...,:3,:3] = torch.matmul(b_t.view(*b_t.shape[:-1],4,4)[...,0:1,:3,:3], bone_trans_restore[...,:3,:3])
                    AB_trans = torch.cat([b_f_f.view(*b_f_f.shape[:-1],4,4), bone_trans_restore],dim=-3)
                    coarse_predict = self.lbs(self.lbs_src, AB_trans)
        
                detail_res, next_fine_hidden = self.f_model(AB_trans[...,:3,:], fine_hidden)
                fine_hidden = next_fine_hidden.detach()

                predict =  coarse_predict + detail_res

                b_v = body_seq_v[:, s_i*self.tbqq_lenth:(s_i+1)*self.tbqq_lenth].to(self.device)
                S_loss, l2_loss, lap_loss, col_loss = self.S_loss_func.getLosses(predict, g_t, b_v, self.train_data.body_f)
                rmse = self.S_loss_func.computeRMSE(predict, g_t)

                S_loss.backward()
                self.optimizer.step()

                supervised_loss.append([S_loss.item(), l2_loss.item(), lap_loss.item(), rmse.item(), col_loss.item()])

        supervised_loss = torch.mean(torch.tensor(supervised_loss),dim=0)

        if self.cur_epoch % 10 ==0:
            self.testFine()

        self.writeError("train_loss",supervised_loss[0])
        self.writeError("train_rmse",supervised_loss[3])

        self.info_str = f"train={supervised_loss[0]:.3e},l2:{supervised_loss[1]:.3e},rmse:{supervised_loss[3]:.3e},col:{supervised_loss[4]:.3e}, ||\
              test={self.test_loss[0]:.3e},l2:{self.test_loss[1]:.3e},rmse:{self.test_loss[3]:.3e},col:{self.test_loss[4]:.3e},mse:{self.test_loss[5]:.3e}" 
        self.pbar.set_postfix_str(self.info_str)
        self.updateEpoch()

    def testFine(self):
        self.test_loss = []
        self.c_model.eval()
        self.f_model.eval()

        with torch.no_grad():
            for d_ind,(b_t, body_seq_v, h_t, g_t) in enumerate(self.test_loader):
                h_t = (h_t- h_t[:,:1])
                coarse_hidden = None
                fine_hidden = None  
                
                b_t = b_t.to(self.device)
                g_t = g_t.to(self.device)
                h_t = h_t.to(self.device)

                b_f_f, bone_trans_s, next_coarse_hidden = self.c_model(b_t,h_t,coarse_hidden)

                bone_trans_restore = transformRestore(bone_trans_s)
                # bone_trans_restore[...,:3,:3] = torch.matmul(b_t.view(*b_t.shape[:-1],4,4)[...,0:1,:3,:3], bone_trans_restore[...,:3,:3])
                AB_trans = torch.cat([b_f_f.view(*b_f_f.shape[:-1],4,4), bone_trans_restore],dim=-3)
                coarse_predict = self.lbs(self.lbs_src, AB_trans)
    
                detail_res, next_fine_hidden = self.f_model(AB_trans[...,:3,:], fine_hidden)

                predict =  coarse_predict + detail_res

                b_v = body_seq_v.to(self.device)
                S_loss, l2_loss, lap_loss, col_loss = self.S_loss_func.getLosses(predict, g_t, b_v, self.train_data.body_f)
                rmse = self.S_loss_func.computeRMSE(predict, g_t)
                mse = self.S_loss_func.computeMSE(predict,g_t)
                self.test_loss.append([S_loss.item(), l2_loss.item(), lap_loss.item(), rmse.item(), col_loss.item(), mse.item()])

            self.test_loss = torch.mean(torch.tensor(self.test_loss),dim=0)
            self.writeError("test_loss",self.test_loss[0])
            self.writeError("test_rmse",self.test_loss[3])


    def predict(self,save_file_path = "./result/predict"):
        from psbody.mesh import Mesh as Obj
        self.info_str = ""
        print("predicting...", "path:",save_file_path)

        self.c_model.eval()
        self.f_model.eval()

        avg_losses = []
        with torch.no_grad():
            for d_ind,(b_t, body_seq_v, h_t, g_t) in enumerate(self.test_loader):
                self.test_loss = []
                coarse_hidden = None
                fine_hidden = None  
                b_t = b_t.to(self.device)
                g_t = g_t.to(self.device)
                h_t = (h_t- h_t[:,:1]).to(self.device) 
                b_v = body_seq_v.to(self.device)

                b_f_f, bone_trans_s, next_coarse_hidden = self.c_model(b_t,h_t,coarse_hidden)
                # coarse_hidden =  next_coarse_hidden.detach()

                bone_trans_restore = transformRestore(bone_trans_s)
                # bone_trans_restore[...,:3,:3] = torch.matmul(b_t.view(*b_t.shape[:-1],4,4)[...,0:1,:3,:3], bone_trans_restore[...,:3,:3])
                AB_trans = torch.cat([b_f_f.view(*b_f_f.shape[:-1],4,4), bone_trans_restore],dim=-3)
                coarse_predict = self.lbs(self.lbs_src, AB_trans)
    
                detail_res, next_fine_hidden = self.f_model(AB_trans[...,:3,:], fine_hidden)
                # fine_hidden = next_fine_hidden.detach()

                predict =  coarse_predict + detail_res

                S_loss, l2_loss, lap_loss, col_loss = self.S_loss_func.getLosses(predict, g_t.to(self.device), b_v, self.train_data.body_f)
                rmse = self.S_loss_func.computeRMSE(predict, g_t.to(self.device))
                mse = self.S_loss_func.computeMSE(predict, g_t.to(self.device))
                self.test_loss.append([S_loss.item(), l2_loss.item(), lap_loss.item(),rmse.item(),mse.item()])
                self.test_loss = torch.mean(torch.tensor(self.test_loss),dim=0)
                
                avg_losses.append(self.test_loss[-1])

                print("predict", self.test_data.data_folder[d_ind],  f" test={self.test_loss[0]:.3e},rmse:{self.test_loss[3]:.3e}, mse{self.test_loss[4]:.3e}" )

                frame_num = predict.shape[1]

                #---------------------- save npz info---------------------
                v_arr = predict[0].cpu().numpy()
                f = self.train_data.cloth.faces.numpy()
                for v_i,v in enumerate(v_arr):
                    file_name  = os.path.join(self.config["predict_path"],self.test_data.data_folder[d_ind],self.train_data.cloth.name,f"c{v_i:07d}.obj")
                    Obj(v=v,f=f).write_obj(file_name)
