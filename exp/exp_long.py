from utils.tools import EarlyStopping, adjust_learning_rate, visual, save_to_csv, visual_weights
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.optim import lr_scheduler
from data_provider.data import data_provider
import csv
from layers.mode import Mode
# from Autostg.mode import Mode
from exp.exp_basic import Exp_Basic

warnings.filterwarnings('ignore')
 
class RunManager(Exp_Basic):
    def __init__(self, args):
        super(RunManager, self).__init__(args) 
        

    def _build_model(self): 
        #fu_arch = 'fusion_genotype_resnet50'
        #genotype_fu = eval("genotypes.%s" % fu_arch) 

        # model = self.model_dict['MsaSTG'].Model(self.args).float()   
        # model = self.model_dict['Autostg'].Model(self.args).float()   
        model = self.model_dict[self.args.model].Model(self.args).float()   

        if self.args.use_multi_gpu and self.args.use_gpu: 
            model = nn.DataParallel(model, device_ids=self.args.device_ids) 
        return model 

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        if self.args.data == 'PEMS':
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion
    

    def clear_records(self):
        self._best_epoch = -1
        self._valid_records = []

    def initialize(self, train_steps):
        # initialize for weight optimizer
        self._weight_optimizer = torch.optim.Adam(  #权重参数优化器 adma SGD优化算法 随机梯度下降 后续可以更改梯度下降算法？
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=0.00001

        )

        self._weight_optimizer_scheduler = torch.optim.lr_scheduler.OneCycleLR( #一个学习率调度器，在特定milestones上调整权重优化器的学习率
            optimizer=self._weight_optimizer,
            steps_per_epoch=train_steps,
            #steps_per_epoch=122,
            pct_start=self.args.pct_start, #学习率变化的起始百分比
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate #在一个训练周期内逐渐增加学习率，然后再逐渐减小学习率
        )

        # self._weight_optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR( #一个学习率调度器，在特定milestones上调整权重优化器的学习率
        #     optimizer=self._weight_optimizer,
        #     milestones=[50,60,70,80],
        #     gamma=0.1,
        # )

        # initialize for arch optimizer
       
        # initialize validation records
        # self.clear_records() #初始化验证记录，清空之前的记录

        if self.args.model == 'Testmodel' or self.args.model == 'AutoMixer':
            # initialize for arch optimizer
            self._arch_optimizer = torch.optim.Adam( #架构参数优化器
                self.model.arch_parameters(),
                lr=self.args.learning_rate, 
                weight_decay=0.00001
                #weight_decay=self._arch_decay
            )
            
            # # self._arch_optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            # #     optimizer=self._arch_optimizer,
            # #     milestones=[50,60,70,80],
            # #     gamma=0.1,
            # # )
            self._arch_optimizer_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self._arch_optimizer,
                steps_per_epoch=train_steps,
                #steps_per_epoch=122,
                pct_start=self.args.pct_start, #学习率变化的起始百分比
                epochs=self.args.train_epochs,
                max_lr=self.args.learning_rate
            )
        elif self.args.model != 'AutoCTS+':
            self._arch_optimizer = torch.optim.Adam( #架构参数优化器
            # self.model.arch_parameters(),
            self.model.module.arch_parameters(),
            lr=self.args.learning_rate, 
            weight_decay=0.00001
            #weight_decay=self._arch_decay
            )
        
        # self._arch_optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer=self._arch_optimizer,
        #     milestones=[50,60,70,80],
        #     gamma=0.1,
        # )
            self._arch_optimizer_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self._arch_optimizer,
                steps_per_epoch=train_steps,
                #steps_per_epoch=122,
                pct_start=self.args.pct_start, #学习率变化的起始百分比
                epochs=self.args.train_epochs,
                max_lr=self.args.learning_rate
            )
        # initialize validation records
        self.clear_records() #初始化验证记录，清空之前的记录
        # if self.args.model == "Autostg":
        self.adj_mats = np.ones((self.args.enc_in, self.args.enc_in, 4))
        self.node_fts = np.ones((self.args.enc_in, 2))
        # self.adj_mats = np.ones((self.args.enc_in, self.args.enc_in))


        if self.args.model == "Autostg":
            self.node_fts = np.ones((self.args.enc_in, 2))
            self.adj_mats = np.ones((self.args.enc_in, self.args.enc_in, 4))
            if self.args.model_id == 'PEMS03':
                self.adj_mats = self.get_adj_matrix('pems/PEMS03/PEMS03.csv', self.args.enc_in, id_filename='pems/PEMS03/PEMS03.txt')
                self.adj_mats = np.expand_dims(self.adj_mats, axis=-1)
                self.adj_mats = np.repeat(self.adj_mats, 4, axis=-1)
            elif self.args.model_id == 'PEMS04' or self.args.model_id == 'PEMS07' or self.args.model_id == 'PEMS08':
                path = 'pems/'+self.args.model_id+'/'+self.args.model_id+'.csv'
                self.adj_mats = self.get_adj_matrix(path, self.args.enc_in)
                self.adj_mats = np.expand_dims(self.adj_mats, axis=-1)
                self.adj_mats = np.repeat(self.adj_mats, 4, axis=-1)
            # adj_mats = self.get_adj_matrix('pems/PEMS07/PEMS07.csv', 883)
            # adj_mats = self.get_adj_matrix('pems/PEMS08/PEMS08.csv', 170)
            

    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                batch_x_mark = None
                batch_y_mark = None

                if self.args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None
                
                # encoder - decoder
                if self.args.model == 'AutoCTS+' or self.args.model == 'AutoCTS':
                    x_enc = batch_x.unsqueeze(-1).permute(0, 3, 2, 1)
                    out_x = self.model(x_enc)
                    outputs = out_x.squeeze(-1)
                elif self.args.model == "Autostg":
                    batch_x_a = ((batch_x.unsqueeze(-1)).repeat(1, 1, 1, 2)).permute(0, 3, 2, 1)
                    outputs_a = self.model(batch_x_a, self.node_fts, self.adj_mats, Mode.ALL_PATHS)
                    outputs = outputs_a.permute(0, 3, 2, 1).squeeze(-1)
                else:    
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, Mode.TWO_PATHS)
                # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0

                pred = outputs.detach()
                true = batch_y.detach()

                if self.args.data == 'PEMS':
                    B, T, C = pred.shape
                    pred = pred.cpu().numpy()
                    true = true.cpu().numpy()
                    pred = vali_data.inverse_transform(pred.reshape(-1, C)).reshape(B, T, C)
                    true = vali_data.inverse_transform(true.reshape(-1, C)).reshape(B, T, C)
                    mae, mse, rmse, mape, mspe, corr, rse = metric(pred, true)
                    total_loss.append(mae)

                else:
                    loss = criterion(pred, true)
                    total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    # def train(self, setting):
    #     train_data, train_loader = self._get_data(flag='train')
    #     vali_data, vali_loader = self._get_data(flag='val')
    #     test_data, test_loader = self._get_data(flag='test')

    #     self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

    def load_pth(self, path, train_steps = 122):

        self.initialize(train_steps)
        print(path)
        states = torch.load(path)
        self.model.load_state_dict(states['net']) #加载网络
        # load optimizer 加载优化器
        self._arch_optimizer.load_state_dict(states['arch_optimizer']) 
        self._arch_optimizer_scheduler.load_state_dict(states['arch_optimizer_scheduler']) 
        self._weight_optimizer.load_state_dict(states['weight_optimizer']) 
        self._weight_optimizer_scheduler.load_state_dict(states['weight_optimizer_scheduler'])


    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)

        self.initialize(train_steps)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # model_optim = self._select_optimizer()
        criterion = self._select_criterion()


        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader): 
                iter_count += 1
                  #架构参数和权重参数共同训练

                self.model.train()

                self._weight_optimizer.zero_grad()
                for p in self.model.arch_parameters(): #训练weight时不对arch求导
                    p.requires_grad_(False)
                #self._arch_optimizer.zero_grad() 

                batch_x = batch_x.float().to(self.device) #32*96*358 bt*input*num
                batch_y = batch_y.float().to(self.device) #32*12*358 bt*output*num
                

                batch_x_mark = None
                batch_y_mark = None
                dec_inp = None
                #outputs = self.model(batch_x) #这里是数据通过网络的地方 可以在此更换网络

                # if self.args.model == 'AutoCTS+':
                #     batch_x = batch_x.unsqueeze(-1).permute(0, 3, 2, 1)
                #     outputs = self.model(batch_x)
                #     outputs = outputs.squeeze(-1)
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, Mode.TWO_PATHS)
                loss = criterion(outputs, batch_y) 
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                #loss.backward(retain_graph=False)
                loss.backward()
                #是否需要梯度裁剪
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(),5)

                self._weight_optimizer.step()

                adjust_learning_rate(self._weight_optimizer, self._weight_optimizer_scheduler, epoch + 1, self.args, printout=False)

                self._weight_optimizer_scheduler.step() #更新权重优化器的学习率调度器



            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            #early_stopping(vali_loss, self.model, path)
            early_stopping(vali_loss, self.model, self, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def get_adj_matrix(self, distance_df_filename, num_of_vertices, type_='connectivity', id_filename=None):
        A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32) #358*358
        if id_filename:
            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx
                        for idx, i in enumerate(f.read().strip().split('\n'))}
            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2]) #distance没有用吗?
                    A[id_dict[i], id_dict[j]] = 1
                    A[id_dict[j], id_dict[i]] = 1
            return A

        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                if type_ == 'connectivity':
                    A[i, j] = 1
                    A[j, i] = 1
                elif type_ == 'distance':
                    A[i, j] = 1 / distance
                    A[j, i] = 1 / distance
                else:
                    raise ValueError("type_ error, must be connectivity or distance!")

        return A

    def search(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
    
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader) #488

        self.initialize(train_steps)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True) 

        #model_optim = self._select_optimizer() #Adam优化器
        
        criterion = self._select_criterion() #L1损失函数

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader): 
                iter_count += 1
                  #架构参数和权重参数共同训练

                self.model.train()

                for p in self.model.arch_parameters(): #训练weight时不对arch求导
                    p.requires_grad_(False)

                self._weight_optimizer.zero_grad()
                self._arch_optimizer.zero_grad() 

                batch_x = batch_x.float().to(self.device) #32*96*358 bt*input*num
                batch_y = batch_y.float().to(self.device) #32*12*358 bt*output*num

                batch_x_mark = None
                batch_y_mark = None
                dec_inp = None
                
                #outputs = self.model(batch_x) #这里是数据通过网络的地方 可以在此更换网络
                if self.args.model == 'AutoCTS':
                    batch_x_a = batch_x.unsqueeze(-1).permute(0, 3, 2, 1)
                    outputs_a = self.model(batch_x_a)
                    outputs = outputs_a.squeeze(-1)
                elif self.args.model == "Autostg":
                    batch_x_a = ((batch_x.unsqueeze(-1)).repeat(1, 1, 1, 2)).permute(0, 3, 2, 1)
                    outputs_a = self.model(batch_x_a, self.node_fts, self.adj_mats, Mode.ALL_PATHS)
                    outputs = outputs_a.permute(0, 3, 2, 1).squeeze(-1)
                else:    
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, Mode.ALL_PATHS)
                loss = criterion(outputs, batch_y) 
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward(retain_graph=False)
                #是否需要梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),5)

                self._weight_optimizer.step()

                adjust_learning_rate(self._weight_optimizer, self._weight_optimizer_scheduler, epoch + 1, self.args, printout=False)

                self._weight_optimizer_scheduler.step() #更新权重优化器的学习率调度器


                if epoch < 3: continue
                
                self.model.eval()
                self._weight_optimizer.zero_grad()
                self._arch_optimizer.zero_grad()

                for p in self.model.arch_parameters(): #对arch求导
                    p.requires_grad_(True)
                x_search, y_search, batch_x_mark1, batch_y_mark1 = next(iter(vali_loader))

                x_search = x_search.float().to(self.device)
                y_search = y_search.float().to(self.device)

                #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
                if self.args.model == 'AutoCTS':
                    batch_x_a = x_search.unsqueeze(-1).permute(0, 3, 2, 1)
                    outputs_a = self.model(batch_x_a)
                    outputs_s = outputs_a.squeeze(-1)
                elif self.args.model == "Autostg":
                    x_search_a = ((x_search.unsqueeze(-1)).repeat(1, 1, 1, 2)).permute(0, 3, 2, 1)
                    outputs_a = self.model(x_search_a, self.node_fts, self.adj_mats, Mode.ALL_PATHS)
                    outputs_s = outputs_a.permute(0, 3, 2, 1).squeeze(-1)
                else:    
                    outputs_s = self.model(x_search, batch_x_mark, dec_inp, batch_y_mark, Mode.ALL_PATHS)

                # outputs_s = self.model(x_search, batch_x_mark, dec_inp, batch_y_mark)

                loss_s = criterion(outputs_s, y_search) 
                #print("#####################################################\n")

                loss_s.backward(retain_graph=False)
                #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
                # if self.args.model != 'AutoCTS':
                #     torch.nn.utils.clip_grad_norm_(self.model.module.arch_parameters(),5)
                # else:
                #     torch.nn.utils.clip_grad_norm_(self.model.arch_parameters(),5)

                torch.nn.utils.clip_grad_norm_(self.model.arch_parameters(),5)

                self._arch_optimizer.step()
                adjust_learning_rate(self._arch_optimizer, self._arch_optimizer_scheduler, epoch + 1, self.args, printout=False)
                #print("*****************************************\n")
                self._arch_optimizer_scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, self, path)
            # early_stopping(vali_loss, self.model, self._weight_optimizer, self._weight_optimizer_scheduler, self._arch_optimizer, self._arch_optimizer_scheduler, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            #print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        #self.model.load_state_dict(torch.load(best_model_path))

        #return self.model
        return


    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            #self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            # self.load_pth(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth'))
            # path = os.path.join(self.args.checkpoints + setting, 'train.pth')
            # states = torch.load(path)
            # self.model.load_state_dict(states['net'])
            self.load_pth(os.path.join(self.args.checkpoints + setting, 'train.pth'))
            # for index, p in enumerate(self.model.arch_parameters()):
            #      print(f'alpha: {p}')

        #checkpoints_path = './checkpoints/' + setting + '/'
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                batch_x_mark = None
                batch_y_mark = None

                if self.args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None

                # encoder - decoder
                if self.args.model == 'AutoCTS+' or self.args.model == 'AutoCTS':
                    x_enc = batch_x.unsqueeze(-1).permute(0, 3, 2, 1)
                    out_x = self.model(x_enc)
                    outputs= out_x.squeeze(-1)
                elif self.args.model == "Autostg":
                    batch_x_a = ((batch_x.unsqueeze(-1)).repeat(1, 1, 1, 2)).permute(0, 3, 2, 1)
                    outputs_a = self.model(batch_x_a, self.node_fts, self.adj_mats, Mode.ALL_PATHS)
                    outputs = outputs_a.permute(0, 3, 2, 1).squeeze(-1)
                else:    
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, Mode.TWO_PATHS)
                # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        if self.args.data == 'PEMS':
            B, T, C = preds.shape
            preds = test_data.inverse_transform(preds.reshape(-1, C)).reshape(B, T, C)
            trues = test_data.inverse_transform(trues.reshape(-1, C)).reshape(B, T, C)

        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, corr, rse = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        print('rmse:{}, mape:{}, mspe:{}'.format(rmse, mape, mspe))
        print('corr:{}, rse:{}'.format(corr, rse))

        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        if self.args.data == 'PEMS':
            f.write('mae:{}, mape:{}, rmse:{}'.format(mae, mape, rmse))
        else:
            f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        return
