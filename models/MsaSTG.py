import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#from operations import *
from models.mxt import candidate_op_profiles, Cell


class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features #358
        self.eps = eps   #数值稳定性的增量
        self.affine = affine #是否使用可学习的仿射变换参数
        self.subtract_last = subtract_last #是否减去最后一个特征
        self.non_norm = non_norm  #是否进行非归一化处理
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features)) #缩放因子 1
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features)) #偏置项 0

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu') #权重初始化  kaiming正态分布 非线性激活函数

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2) #送进卷积层的是32*358*96 输出 32*128*96  
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000): 
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float() #创建一个形状为(max_len, d_model)的零张量pe，用于存储位置嵌入的值
        pe.require_grad = False  

        position = torch.arange(0, max_len).float().unsqueeze(1)#创建一个形状为(max_len, 1)的位置张量position，其中包含从0到max_len-1的连续数值
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()   #用于计算位置嵌入的分母项

        pe[:, 0::2] = torch.sin(position * div_term) #对偶数索引和奇数索引的维度进行sin和cos操作，将结果存储在pe的相应位置上
        pe[:, 1::2] = torch.cos(position * div_term) 

        pe = pe.unsqueeze(0)  #在第0维上添加一个维度，得到形状为(1, max_len, d_model)的位置嵌入张量
        self.register_buffer('pe', pe) 

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x

 
class TimeFeatureEmbedding(nn.Module): #?
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq] #4
        self.embed = nn.Linear(d_inp, d_model, bias=False) # 

    def forward(self, x):
        return self.embed(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size #25  1
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0) #平均池化模型

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1) #填充
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)#填充 
        x = torch.cat([front, x, end], dim=1)  #连接 32*120*358
        x = self.avg(x.permute(0, 2, 1)) #池化 32*358*96
        x = x.permute(0, 2, 1) #32*96*358
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block 时序分解模块
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1) 

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean #差值，均值

class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, 5)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module): 
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i), #96/2^0 96
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)), #96/2^1 48
                    ),
                    nn.GELU(), #非线性映射
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ), #维度恢复

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list): #从细到粗混合 在这里修改为架构搜索

        # mixing high->low 
        out_high = season_list[0]   #32*128*96
        out_low = season_list[1]  #48
        out_season_list = [out_high.permute(0, 2, 1)] #32*96*128
 
        for i in range(len(season_list) - 1): 
            out_low_res = self.down_sampling_layers[i](out_high)  #32*128*48  再次下采样
            out_low = out_low + out_low_res
            out_high = out_low 
            if i + 2 <= len(season_list) - 1: 
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list 


class MultiScaleTrendMixing(nn.Module): 
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)), #48
                        configs.seq_len // (configs.down_sampling_window ** i), #96
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i), #96
                        configs.seq_len // (configs.down_sampling_window ** i),#96
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class MultiScaleSeasonMixing(nn.Module): 
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i), #96/2^0 96
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)), #96/2^1 48
                    ),
                    nn.GELU(), #非线性映射
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                    ), #维度恢复

                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, season_list): #从细到粗混合 在这里修改为架构搜索

        # mixing high->low 
        out_high = season_list[0]   #32*128*96
        out_low = season_list[1]  #48
        out_season_list = [out_high.permute(0, 2, 1)] #32*96*128
 
        for i in range(len(season_list) - 1): 
            out_low_res = self.down_sampling_layers[i](out_high)  #32*128*48  再次下采样
            out_low = out_low + out_low_res
            out_high = out_low 
            if i + 2 <= len(season_list) - 1: 
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list 


class MultiScaleTrendMixing(nn.Module): 
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)), #48
                        configs.seq_len // (configs.down_sampling_window ** i), #96
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i), #96
                        configs.seq_len // (configs.down_sampling_window ** i),#96
                    ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list



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
            self.cross_layer = nn.Sequential( #这一步操作的作用是什么？
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff), #128-256
                nn.GELU(), 
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model), #256-128 
            )

        # self.Nas_mix =  nn.ModuleList([Cell(configs, configs.op_num, candidate_op_profiles)
        #                                   for _ in range(configs.down_sampling_layers + 1)])
        
        self.Nas_mix = Cell(configs, configs.op_num, candidate_op_profiles)

        self.Nas_mix_i = nn.ModuleList([Cell(configs, configs.op_num, candidate_op_profiles)
                                          for _ in range(configs.e_layers)])
        # Mixing season 
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(configs)  #下采样 季节性

        
        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(configs) #上采样 趋势性

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )
        # self.uni_scale = torch.nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             torch.nn.Linear(6,12),
        #             nn.GELU(),
        #             torch.nn.Linear(12,12),
        #         ) ,
        #         nn.Sequential(
        #             torch.nn.Linear(3,12),
        #             nn.GELU(),
        #             torch.nn.Linear(12,12),
        #         ) , 
        #         nn.Sequential(
        #             torch.nn.Linear(12,6),
        #             nn.GELU(),
        #             torch.nn.Linear(6,6),
        #         ) ,
        #         nn.Sequential(
        #             torch.nn.Linear(3,6),
        #             nn.GELU(),
        #             torch.nn.Linear(6,6),
        #         ) ,
        #         nn.Sequential(
        #             torch.nn.Linear(12,3),
        #             nn.GELU(),
        #             torch.nn.Linear(3,3),
        #         ) ,
        #         nn.Sequential(
        #             torch.nn.Linear(6,3),
        #             nn.GELU(),
        #             torch.nn.Linear(3,3),
        #         ) 
        #     ]
        #     )
        # self.uni_scale = torch.nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             torch.nn.Linear(24,48),
        #             nn.GELU(),
        #             torch.nn.Linear(48,48),
        #         ) ,
        #         nn.Sequential(
        #             torch.nn.Linear(12,48),
        #             nn.GELU(),
        #             torch.nn.Linear(48,48),
        #         ) ,

        #         nn.Sequential(
        #             torch.nn.Linear(48,24),
        #             nn.GELU(),
        #             torch.nn.Linear(24,24),
        #         ),
        #         nn.Sequential(
        #             torch.nn.Linear(12,24),
        #             nn.GELU(),
        #             torch.nn.Linear(24,24),
        #         ),

        #         nn.Sequential(
        #             torch.nn.Linear(48,12),
        #             nn.GELU(),
        #             torch.nn.Linear(12,12),
        #         ) ,
                
        #         nn.Sequential(
        #             torch.nn.Linear(24,12),
        #             nn.GELU(),
        #             torch.nn.Linear(12,12),
        #         ) ,
                
        #     ]
        #     )
        # self.uni_scale = torch.nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             torch.nn.Linear(48,96),
        #             nn.GELU(),
        #             torch.nn.Linear(96,96),
        #         ) ,
        #         nn.Sequential(
        #             torch.nn.Linear(24,96),
        #             nn.GELU(),
        #             torch.nn.Linear(96,96),
        #         ) , 
        #         nn.Sequential(
        #             torch.nn.Linear(96,48),
        #             nn.GELU(),
        #             torch.nn.Linear(48,48),
        #         ) ,
        #         nn.Sequential(
        #             torch.nn.Linear(24,48),
        #             nn.GELU(),
        #             torch.nn.Linear(48,48),
        #         ) ,
        #         nn.Sequential(
        #             torch.nn.Linear(96,24),
        #             nn.GELU(),
        #             torch.nn.Linear(24,24),
        #         ) ,
        #         nn.Sequential(
        #             torch.nn.Linear(48,24),
        #             nn.GELU(),
        #             torch.nn.Linear(24,24),
        #         ) 
        #     ]
        #     )

        input_sizes = [configs.seq_len // (2 ** i) for i in range(4)] 
        # input_sizes = [96, 48, 24, 12]
        # output_sizes = [96, 48, 24, 12]
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


        #print(uni_scale)



        # if(configs.seq_len == 168):
        #     self.uni_scale = torch.nn.ModuleList(
        #     [
        #         nn.Sequential(
        #                 torch.nn.Linear(84,168),
        #                 nn.GELU(),
        #                 torch.nn.Linear(168,168),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(42,168),
        #                 nn.GELU(),
        #                 torch.nn.Linear(168,168),
        #             ) , 
        #             nn.Sequential(
        #                 torch.nn.Linear(21,168),
        #                 nn.GELU(),
        #                 torch.nn.Linear(168,168),
        #             ) , 
        #             nn.Sequential(
        #                 torch.nn.Linear(168,84),
        #                 nn.GELU(),
        #                 torch.nn.Linear(84,84),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(42,84),
        #                 nn.GELU(),
        #                 torch.nn.Linear(84,84),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(21,84),
        #                 nn.GELU(),
        #                 torch.nn.Linear(84,84),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(168,42),
        #                 nn.GELU(),
        #                 torch.nn.Linear(42,42),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(84,42),
        #                 nn.GELU(),
        #                 torch.nn.Linear(42,42),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(21,42),
        #                 nn.GELU(),
        #                 torch.nn.Linear(42,42),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(168,21),
        #                 nn.GELU(),
        #                 torch.nn.Linear(21,21),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(84,21),
        #                 nn.GELU(),
        #                 torch.nn.Linear(21,21),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(42,21),
        #                 nn.GELU(),
        #                 torch.nn.Linear(21,21),
        #             ) ,
        #     ]
        #     )
        # elif(configs.seq_len == 96):
        #     self.uni_scale = torch.nn.ModuleList(
        #         [
        #             nn.Sequential(
        #                 torch.nn.Linear(48,96),
        #                 nn.GELU(),
        #                 torch.nn.Linear(96,96),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(24,96),
        #                 nn.GELU(),
        #                 torch.nn.Linear(96,96),
        #             ) , 
        #             nn.Sequential(
        #                 torch.nn.Linear(12,96),
        #                 nn.GELU(),
        #                 torch.nn.Linear(96,96),
        #             ) , 
        #             nn.Sequential(
        #                 torch.nn.Linear(96,48),
        #                 nn.GELU(),
        #                 torch.nn.Linear(48,48),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(24,48),
        #                 nn.GELU(),
        #                 torch.nn.Linear(48,48),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(12,48),
        #                 nn.GELU(),
        #                 torch.nn.Linear(48,48),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(96,24),
        #                 nn.GELU(),
        #                 torch.nn.Linear(24,24),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(48,24),
        #                 nn.GELU(),
        #                 torch.nn.Linear(24,24),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(12,24),
        #                 nn.GELU(),
        #                 torch.nn.Linear(24,24),
        #             ),
        #             nn.Sequential(
        #                 torch.nn.Linear(96,12),
        #                 nn.GELU(),
        #                 torch.nn.Linear(12,12),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(48,12),
        #                 nn.GELU(),
        #                 torch.nn.Linear(12,12),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(24,12),
        #                 nn.GELU(),
        #                 torch.nn.Linear(12,12),
        #             )  
        #         ]
        #     )
        # elif(configs.seq_len == 36):
        #     self.uni_scale = torch.nn.ModuleList(
        #         [
        #             nn.Sequential(
        #                 torch.nn.Linear(18,36),
        #                 nn.GELU(),
        #                 torch.nn.Linear(36,36),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(9,36),
        #                 nn.GELU(),
        #                 torch.nn.Linear(36,36),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(4,36),
        #                 nn.GELU(),
        #                 torch.nn.Linear(36,36),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(36,18),
        #                 nn.GELU(),
        #                 torch.nn.Linear(18,18),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(9,18),
        #                 nn.GELU(),
        #                 torch.nn.Linear(18,18),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(4,18),
        #                 nn.GELU(),
        #                 torch.nn.Linear(18,18),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(36,9),
        #                 nn.GELU(),
        #                 torch.nn.Linear(9,9),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(18,9),
        #                 nn.GELU(),
        #                 torch.nn.Linear(9,9),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(4,9),
        #                 nn.GELU(),
        #                 torch.nn.Linear(9,9),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(36,4),
        #                 nn.GELU(),
        #                 torch.nn.Linear(4,4),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(18,4),
        #                 nn.GELU(),
        #                 torch.nn.Linear(4,4),
        #             ) ,
        #             nn.Sequential(
        #                 torch.nn.Linear(9,4),
        #                 nn.GELU(),
        #                 torch.nn.Linear(4,4),
        #             ) ,
        #         ]
        #     )
            
    
    def Nas(self, x_list, t = 0):
        out_x_list = []
        #copy_list = x_list
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
            
            #out_x_list.append((self.Nas_mix(scale_x_list[0])+self.Nas_mix(scale_x_list[1])+self.Nas_mix(scale_x_list[2])+self.Nas_mix(scale_x_list[3])).permute(0, 2, 1))
            # out_x_list.append((self.Nas_mix(scale_x_list[0])+self.Nas_mix(scale_x_list[1])+self.Nas_mix(scale_x_list[2])).permute(0, 2, 1))
            
            if(len(x_list) == 4):
                out_x_list.append((self.Nas_mix(scale_x_list[0])+self.Nas_mix(scale_x_list[1])+self.Nas_mix(scale_x_list[2])+self.Nas_mix(scale_x_list[3])).permute(0, 2, 1))
            elif(len(x_list) == 3):
                out_x_list.append((self.Nas_mix(scale_x_list[0])+self.Nas_mix(scale_x_list[1])+self.Nas_mix(scale_x_list[2])).permute(0, 2, 1))
            else:
                out_x_list.append((self.Nas_mix(scale_x_list[0])+self.Nas_mix(scale_x_list[1])).permute(0, 2, 1))

            #out_x_list.append((self.Nas_mix(scale_x_list[0])+self.Nas_mix(scale_x_list[1])).permute(0, 2, 1))
            # if t == 4:
            #     out_x_list.append((self.Nas_mix_i[t](scale_x_list[0])+self.Nas_mix_i[t](scale_x_list[1])+self.Nas_mix_i[t](scale_x_list[2])).permute(0, 2, 1))
            # else:
            #     out_x_list.append((self.Nas_mix_i[t](scale_x_list[0])+self.Nas_mix_i[t](scale_x_list[1])+self.Nas_mix_i[t](scale_x_list[2])))

        #x = 1
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
            season_list.append(season.permute(0, 2, 1)) #32*128*96 32*128*48
            trend_list.append(trend.permute(0, 2, 1)) #32*128*96 32*128*48
            #season_list.append(season) #32*96*128 32*48*128
            #trend_list.append(trend) #32*96*128 32*48*128


        # bottom-up season mixing 
        #out_season_list = self.mixing_multi_scale_season(season_list) #自底向上季节混合 32*96*128 32*48*128
            
        # out_season_list = season_list
            
        # out_season_list[1] = self.Nas(season_list)  
        # out_season_list[0] = out_season_list[0].permute(0, 2, 1)
        # out_season_list[1] = out_season_list[1].permute(0, 2, 1)

        out_season_list = self.Nas(season_list)  
        out_trend_list = self.Nas(trend_list) 
            
        #out_trend_list = self.mixing_multi_scale_trend(trend_list) #自上而下趋势混合
        # top-down trend mixing
        #out_trend_list = trend_list
        #out_trend_list = self.Nas_mix(trend_list)

        #for i in range(5):
        # out_season_list = self.Nas(season_list)  
        # out_trend_list = self.Nas(trend_list) 
        

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list #32*96*128 32*48*128


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
        # for m in self.Nas_mix:
        #     if isinstance(m, Cell):
        #         for p in m.arch_parameters():
        #             yield p

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
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list) #res,moving_mean 
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]): #季节
                enc_out = self.enc_embedding(x, None)  # [B,T,C] 32*96*128 32*48*128
                enc_out_list.append(enc_out) #季节enc

        # enc_out_list[0] = enc_out_list[0].permute(0, 2, 1)
        # enc_out_list[1] = enc_out_list[1].permute(0, 2, 1)
       # Past Decomposable Mixing as encoder for past 这里其实就是混合层了
                
        for i in range(self.layer): #5层 一模一样的层 通过5次 
            enc_out_list = self.pdm_blocks[i](enc_out_list)  
        # enc_out_list = self.pdm_blocks(enc_out_list)  

            #enc_out_list[0], enc_out_list[1] = self.Nas_mix[i](enc_out_list[0], enc_out_list[1])  
            #enc_out_list[1] = self.Nas_mix[i](enc_out_list[0], enc_out_list[1])  
        
        # enc_out_list[0] = enc_out_list[0].permute(0, 2, 1)
        # enc_out_list[1] = enc_out_list[1].permute(0, 2, 1)

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