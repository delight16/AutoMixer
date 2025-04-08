from torch.optim import lr_scheduler

from data_provider.data import data_provider
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, save_to_csv
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.m4_summary import M4Summary
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas
# from model.mode import Mode
warnings.filterwarnings('ignore')


class Exp_Short_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Short_Term_Forecast, self).__init__(args)

    def _build_model(self):
        if self.args.data == 'm4':
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]  # Up to M4 config
            self.args.seq_len = 2 * self.args.pred_len  # input_len = 2*pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
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

    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()

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

        if self.args.model != 'AutoCTS+':
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
        # initialize validation records
        self.clear_records() #初始化验证记录，清空之前的记录
        # if self.args.model == "Autostg":
        self.adj_mats = np.ones((self.args.enc_in, self.args.enc_in, 4))
        self.node_fts = np.ones((self.args.enc_in, 2))
        # self.adj_mats = np.ones((self.args.enc_in, self.args.enc_in))
        

    def load_pth(self, path, train_steps = 122):

        self.initialize(train_steps)
        states = torch.load(path)
        self.model.load_state_dict(states['net']) #加载网络

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)

        self.initialize(train_steps)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)


        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                self.model.train()

                self._weight_optimizer.zero_grad()
                # self._arch_optimizer.zero_grad() 

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                if self.args.model == 'AutoCTS+':
                    batch_x = batch_x.unsqueeze(-1).permute(0, 3, 2, 1)
                    outputs = self.model(batch_x)
                    outputs = outputs.squeeze(-1)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss_value = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
                # loss_sharpness = mse((outputs[:, 1:, :] - outputs[:, :-1, :]), (batch_y[:, 1:, :] - batch_y[:, :-1, :]))
                loss = loss_value  # + loss_sharpness * 1e-5
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),5)

                self._weight_optimizer.step()

                adjust_learning_rate(self._weight_optimizer, self._weight_optimizer_scheduler, epoch + 1, self.args, printout=False)

                self._weight_optimizer_scheduler.step() #更新权重优化器的学习率调度器


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(train_loader, vali_loader, criterion)
            test_loss = vali_loss
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

   
        best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))

        # return self.model
        return 

    def search(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)

        self.initialize(train_steps)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)


        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                self.model.train()

                self._weight_optimizer.zero_grad()
                self._arch_optimizer.zero_grad() 

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.model == 'AutoCTS':
                    batch_x_a = batch_x.unsqueeze(-1).permute(0, 3, 2, 1)
                    outputs_a = self.model(batch_x_a)
                    outputs = outputs_a.squeeze(-1)
                elif self.args.model == "Autostg":
                    batch_x_a = ((batch_x.unsqueeze(-1)).repeat(1, 1, 1, 2)).permute(0, 3, 2, 1)
                    outputs_a = self.model(batch_x_a, self.node_fts, self.adj_mats, Mode.TWO_PATHS)
                    outputs = outputs_a.permute(0, 3, 2, 1).squeeze(-1)
                else:
                    outputs = self.model(batch_x, None, dec_inp, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss_value = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
                # loss_sharpness = mse((outputs[:, 1:, :] - outputs[:, :-1, :]), (batch_y[:, 1:, :] - batch_y[:, :-1, :]))
                loss = loss_value  # + loss_sharpness * 1e-5
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),5)

                self._weight_optimizer.step()

                adjust_learning_rate(self._weight_optimizer, self._weight_optimizer_scheduler, epoch + 1, self.args, printout=False)

                self._weight_optimizer_scheduler.step() #更新权重优化器的学习率调度器

                if epoch < 3: continue
                
                self.model.eval()
                
                x_search, y_search, batch_x_mark1, batch_y_mark1 = next(iter(vali_loader))

                x_search = x_search.float().to(self.device)
                y_search = y_search.float().to(self.device)
                #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")

                batch_y_mark1 = batch_y_mark1.float().to(self.device)

                dec_inp = torch.zeros_like(y_search[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([y_search[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                if self.args.model == 'AutoCTS':
                    batch_x_a = batch_x.unsqueeze(-1).permute(0, 3, 2, 1)
                    outputs_a = self.model(batch_x_a)
                    outputs_s = outputs_a.squeeze(-1)
                elif self.args.model == "Autostg":
                    batch_x_a = ((batch_x.unsqueeze(-1)).repeat(1, 1, 1, 2)).permute(0, 3, 2, 1)
                    outputs_a = self.model(batch_x_a, self.node_fts, self.adj_mats, Mode.TWO_PATHS)
                    outputs_s = outputs_a.permute(0, 3, 2, 1).squeeze(-1)
                else:
                    outputs_s = self.model(x_search, None, dec_inp, None)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs_s = outputs_s[:, -self.args.pred_len:, f_dim:]
                y_search = y_search[:, -self.args.pred_len:, f_dim:].to(self.device)

                batch_y_mark1 = batch_y_mark1[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss_value_s = criterion(x_search, self.args.frequency_map, outputs_s, y_search, batch_y_mark1)
                # loss_sharpness = mse((outputs[:, 1:, :] - outputs[:, :-1, :]), (batch_y[:, 1:, :] - batch_y[:, :-1, :]))
                loss_s = loss_value_s  # + loss_sharpness * 1e-5

                #print("#####################################################\n")

                loss_s.backward(retain_graph=False)
                #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
                torch.nn.utils.clip_grad_norm_(self.model.arch_parameters(),5)

                self._arch_optimizer.step()
                adjust_learning_rate(self._arch_optimizer, self._arch_optimizer_scheduler, epoch + 1, self.args, printout=False)
                #print("*****************************************\n")
                self._arch_optimizer_scheduler.step()


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(train_loader, vali_loader, criterion)
            test_loss = vali_loss
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

   
        best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))

        # return self.model
        return 

    def vali(self, train_loader, vali_loader, criterion):
        x, _ = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        self.model.eval()
        with torch.no_grad():
            # decoder input
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()

            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float()  # .to(self.device)
            id_list = np.arange(0, B, 500)  # validation set size
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                x_enc = x[id_list[i]:id_list[i + 1]]

                if self.args.model == 'AutoCTS+' or self.args.model == 'AutoCTS':
                    x_enc = x_enc.unsqueeze(-1).permute(0, 3, 2, 1)
                    out_x = self.model(x_enc).detach().cpu()
                    out_x = out_x.squeeze(-1)
                elif self.args.model == "Autostg":
                    batch_x_a = ((x_enc.unsqueeze(-1)).repeat(1, 1, 1, 2)).permute(0, 3, 2, 1)
                    outputs_a = self.model(batch_x_a, self.node_fts, self.adj_mats, Mode.TWO_PATHS).detach().cpu()
                    out_x = outputs_a.permute(0, 3, 2, 1).squeeze(-1)
                else:
                    out_x = self.model(x_enc, None, dec_inp[id_list[i]:id_list[i + 1]], None).detach().cpu()
                outputs[id_list[i]:id_list[i + 1], :, :] = out_x
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            pred = outputs
            true = torch.tensor(y, dtype=torch.float32).cpu()
            batch_y_mark = torch.ones(true.shape)

            loss = criterion(x.detach().cpu()[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)

        self.model.train()
        return loss

    def test(self, setting, test=0):
        _, train_loader = self._get_data(flag='train')
        _, test_loader = self._get_data(flag='test')
        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        if test:
            print('loading model')
            self.load_pth(os.path.join(self.args.checkpoints + setting, 'train.pth'))
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            id_list = np.arange(0, B, 1)
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                x_enc = x[id_list[i]:id_list[i + 1]]
                if self.args.model == 'AutoCTS+' or self.args.model == 'AutoCTS':
                    x_enc = x_enc.unsqueeze(-1).permute(0, 3, 2, 1)
                    out_x = self.model(x_enc)
                    out_x = out_x.squeeze(-1)
                elif self.args.model == "Autostg":
                    batch_x_a = ((x_enc.unsqueeze(-1)).repeat(1, 1, 1, 2)).permute(0, 3, 2, 1)
                    outputs_a = self.model(batch_x_a, self.node_fts, self.adj_mats, Mode.TWO_PATHS)
                    out_x = outputs_a.permute(0, 3, 2, 1).squeeze(-1)
                else:
                    out_x = self.model(x_enc, None, dec_inp[id_list[i]:id_list[i + 1]], None)
                outputs[id_list[i]:id_list[i + 1], :, :] = out_x

                if id_list[i] % 1000 == 0:
                    print(id_list[i])

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            outputs = outputs.detach().cpu().numpy()

            preds = outputs
            trues = y
            x = x.detach().cpu().numpy()

            for i in range(0, preds.shape[0], preds.shape[0] // 10):
                gt = np.concatenate((x[i, :, 0], trues[i]), axis=0)
                pd = np.concatenate((x[i, :, 0], preds[i, :, 0]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                save_to_csv(gt, pd, os.path.join(folder_path, str(i) + '.csv'))

        print('test shape:', preds.shape)

        # result save
        folder_path = './m4_results/' + self.args.model + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        forecasts_df = pandas.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(self.args.pred_len)])
        forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
        forecasts_df.index.name = 'id'
        forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
        forecasts_df.to_csv(folder_path + self.args.seasonal_patterns + '_forecast.csv')

        print(self.args.model)
        file_path = './m4_results/' + self.args.model + '/'
        if 'Weekly_forecast.csv' in os.listdir(file_path) \
                and 'Monthly_forecast.csv' in os.listdir(file_path) \
                and 'Yearly_forecast.csv' in os.listdir(file_path) \
                and 'Daily_forecast.csv' in os.listdir(file_path) \
                and 'Hourly_forecast.csv' in os.listdir(file_path) \
                and 'Quarterly_forecast.csv' in os.listdir(file_path):
            m4_summary = M4Summary(file_path, self.args.root_path)
            # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
            smape_results, owa_results, mape, mase = m4_summary.evaluate()
            print('smape:', smape_results)
            print('mape:', mape)
            print('mase:', mase)
            print('owa:', owa_results)
        else:
            print('After all 6 tasks are finished, you can calculate the averaged index')
        return

