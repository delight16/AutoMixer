import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.Op1 import candidate_op_profiles, Cell
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from layers.Autoformer_EncDec import series_decomp, DFT_series_decomp
import pywt
import numpy as np

class Model(nn.Module):  #继承自nn.Module类，用于定义一个神经网络模型

    def __init__(self, configs): 
        super(Model, self).__init__() 
        self.configs = configs #传入的配置信息
        self.seq_len = configs.seq_len  #96
        self.batch_size = configs.batch_size  
        self.label_len = configs.label_len  #0
        self.pred_len = configs.pred_len # 12
        self.down_sampling_window = configs.down_sampling_window # 2
        self.channel_independence = configs.channel_independence #0
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                          for _ in range(configs.e_layers)]) #5层
        
        #self.pdm_blocks = PastDecomposableMixing(configs)

        self.preprocess = series_decomp(configs.moving_avg)#平均池化
        self.enc_in = configs.enc_in #358 编码器输入 节点的数量

        self.Nas_mix_i = nn.ModuleList([Cell(configs, configs.op_num, candidate_op_profiles)
                                          for _ in range(configs.e_layers)]) #5层


        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers
        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i), #96 48
                    configs.pred_len,#12
                )
                for i in range(configs.down_sampling_layers + 1) #2
            ]
        )

        if self.channel_independence == 1:
            self.projection_layer = nn.Linear(
                configs.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(
                configs.d_model, configs.c_out, bias=True)

            self.out_res_layers = torch.nn.ModuleList([
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.seq_len // (configs.down_sampling_window ** i),
                )
                for i in range(configs.down_sampling_layers + 1)
            ])

            self.regression_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

    def arch_parameters(self):
        for l in self.pdm_blocks:
            for p in l.arch_parameters():
                yield p


    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)  #32*12*358
        out_res = out_res.permute(0, 2, 1) #32*358*96
        out_res = self.out_res_layers[i](out_res)   #FC层 不变 
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1) #96-12
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1) 
                out2_list.append(x_2) 
            return (out1_list, out2_list) 

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T 
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec): #重要步骤 预测过程 

        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
                x_mark = x_mark.repeat(N, 1, 1)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ): #归一化
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1) #672*336*1
                x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list) #res,moving_mean 
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]): #季节 672*336*16
                enc_out = self.enc_embedding(x, None)  # [B,T,C] 32*96*128 32*48*128
                enc_out_list.append(enc_out) #季节enc


       # Past Decomposable Mixing as encoder for past 这里是混合层
                
        for i in range(self.layer):  
            enc_out_list = self.pdm_blocks[i](enc_out_list)  

        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list) 

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1) #相加
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)
                break
        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):  #out_res是平均值
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension 通过网络层时，中间的是特征通道数128，线性层改变的是最后一维 96-12
                dec_out = self.out_projection(dec_out, i, out_res) #恢复通道数 128->358
                dec_out_list.append(dec_out)
                # break

        return dec_out_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out_list = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out_list




class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len # 96
        self.batch_size = configs.batch_size # 96
        self.pred_len = configs.pred_len #12
        self.down_sampling_window = configs.down_sampling_window #2

        self.layer_norm = nn.LayerNorm(configs.d_model) #LN128
        self.dropout = nn.Dropout(configs.dropout) #0.1
        self.channel_independence = configs.channel_independence #0

        if configs.decomp_method == 'moving_avg': #平均池化
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('decompsition is error')

        if configs.channel_independence == 0: 
            self.cross_layer = nn.Sequential( 
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff), #128-256
                nn.GELU(), 
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model), #256-128 
            )

        
        self.Nas_mix = Cell(configs, configs.op_num, candidate_op_profiles)

        self.Nas_mix_i = nn.ModuleList([Cell(configs, configs.op_num, candidate_op_profiles)
                                          for _ in range(configs.e_layers)])

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )
 

        input_sizes = [configs.seq_len // (2 ** i) for i in range(4)] 
        output_sizes = input_sizes

        self.uni_scale = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(out_size, in_size),
                    nn.GELU(),
                    nn.Linear(in_size, in_size)
                )
                for in_size in input_sizes
                for out_size in output_sizes
                if out_size != in_size
            ]
        )
        self.wavelet = 'cgau8'
        self.totalscal = 10

    # def Nas(self, x_list, t=0):
    #     out_x_list = []
    #     num_inputs = len(x_list)

    #     # 定义映射表，表示每层需要使用的 self.uni_scale 索引
    #     scale_mapping = [
    #         [None, 0, 1, 2],  # 第 0 层
    #         [3, None, 4, 5],  # 第 1 层
    #         [6, 7, None, 8],  # 第 2 层
    #         [9, 10, 11, None] # 第 3 层
    #     ]

    #     for i in range(num_inputs):
    #         scale_x_list = x_list[:]

    #         # 根据映射表动态应用 self.uni_scale
    #         for j in range(num_inputs):
    #             if scale_mapping[i][j] is not None:
    #                 scale_x_list[j] = self.uni_scale[scale_mapping[i][j]](x_list[j])

    #         # 计算混合结果并添加到输出列表
    #         mix_sum = sum(self.Nas_mix(scale) for scale in scale_x_list[:num_inputs])
    #         out_x_list.append(mix_sum.permute(0, 2, 1))

    #     return out_x_list
            
    
    def Nas(self, x_list, t = 0):
        out_x_list = []
        scale_x_list = []
        for i in range(len(x_list)):
            copy_list = []
            for j in range(len(x_list)):
                copy_list.append(x_list[j])
            scale_x_list = copy_list
            if i == 0:
                scale_x_list[1] = self.uni_scale[0](copy_list[1])
                if(len(x_list) >= 3):
                    scale_x_list[2] = self.uni_scale[1](copy_list[2])
                if(len(x_list) == 4):
                    scale_x_list[3] = self.uni_scale[2](copy_list[3])
                
            if i == 1:
                scale_x_list[0] = self.uni_scale[3](copy_list[0])
                if(len(x_list) >= 3):
                    scale_x_list[2] = self.uni_scale[4](copy_list[2])
                if(len(x_list) == 4):
                    scale_x_list[3] = self.uni_scale[5](copy_list[3])


            if i == 2:
                scale_x_list[0] = self.uni_scale[6](copy_list[0])
                scale_x_list[1] = self.uni_scale[7](copy_list[1])
                if(len(x_list) == 4):
                    scale_x_list[3] = self.uni_scale[8](copy_list[3])

            if i == 3:
                scale_x_list[0] = self.uni_scale[9](copy_list[0])
                scale_x_list[1] = self.uni_scale[10](copy_list[1])
                scale_x_list[2] = self.uni_scale[11](copy_list[2])
            

            if(len(x_list) == 4):
                out_x_list.append((self.Nas_mix(scale_x_list[0])+self.Nas_mix(scale_x_list[1])+self.Nas_mix(scale_x_list[2])+self.Nas_mix(scale_x_list[3])).permute(0, 2, 1))
            elif(len(x_list) == 3):
                out_x_list.append((self.Nas_mix(scale_x_list[0])+self.Nas_mix(scale_x_list[1])+self.Nas_mix(scale_x_list[2])).permute(0, 2, 1))
            else:
                out_x_list.append((self.Nas_mix(scale_x_list[0])+self.Nas_mix(scale_x_list[1])).permute(0, 2, 1))

        return out_x_list
    
    def arch_parameters(self):                
        # for m in self.Nas_mix:
        #     for p in m.arch_parameters():
        #         yield p   

        for p in self.Nas_mix.arch_parameters():
            yield p           

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x) #季节和趋势 分别是差值和均值
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1)) #32*128*96 32*128*48 672*16*336
            trend_list.append(trend.permute(0, 2, 1)) #32*128*96 32*128*48

                # Perform wavelet transform on the season_list
        # wavelet_features = []
        # for season in season_list:
        #     tensor_3d = season.permute(0, 2, 1)
        #     ndarray_3d = tensor_3d.detach().cpu().numpy()
        #     wfc = pywt.central_frequency(self.wavelet)
        #     period = 1.0
        #     a = 2 * wfc * self.totalscal / (np.arange(self.totalscal, 0, -1))
        #     amp, _ = pywt.cwt(ndarray_3d, a, self.wavelet, period)
        
        #     amp_tensor = torch.tensor(np.real(amp), dtype=torch.float32).to(season.device)

        #     # 平均化 scales 维度，或者选择其中一个 scale，这里取平均
        #     amp_tensor = amp_tensor.mean(dim=0)  # 平均化 scales 维度，得到 [batch_size, seq_len, N]

        #     # 调整为 [batch_size, N, seq_len]，与 season 一致
        #     amp_tensor = amp_tensor.permute(0, 2, 1)  # 变为 [batch_size, N, seq_len] = [672, 16, 336]

        #     # 确保形状匹配，虽然这不应该是必要的
        #     if amp_tensor.size(1) != season.size(1):
        #         amp_tensor = amp_tensor[:, :season.size(1), :]
        #     if amp_tensor.size(2) != season.size(2):
        #         amp_tensor = amp_tensor[:, :, :season.size(2)]

        #     # 将季节性特征与频域特征相加
        #     wavelet_features.append(amp_tensor)

        # # Combine wavelet features with original seasonal features
        # combined_season_list = [season + wavelet for season, wavelet in zip(season_list, wavelet_features)]

        # out_season_list = self.Nas(combined_season_list)  
        out_season_list = self.Nas(season_list)  
        out_trend_list = self.Nas(trend_list) 
        
        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list #32*96*128 32*48*128

