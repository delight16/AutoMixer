import sys,os
os.environ['CUDA_VISIBLE_DEVICES']='0, 1, 2, 3, 4, 5'
import argparse
import random
import torch
import numpy as np
from exp.exp_long import RunManager
from exp.exp_m4 import Exp_Short_Term_Forecast
import time
import logging
from data_provider.m4 import M4Meta
import os, sys
from layers.mode import Mode
os.chdir(sys.path[0])

print(torch.cuda.device_count())

parser = argparse.ArgumentParser(description='AutoMixer')  #创建一个ArgumentParser对象，用于解析命令行参数。description参数是关于脚本功能的简短描述。

# basic config

parser.add_argument('--is_training', type=int, required=True, default=1, help='status') #训练状态
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id') #
parser.add_argument('--model', type=str, required=True, default='Automixer',
                    help='model name, options: [Automixer]') #模型名称 可选 待补充

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type') #数据
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file') #数据文件路径
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file') #目录文件
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate') #特征类型：多对多，单对单，多对单
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task') #在单对单和多对单任务中的目标特征
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h') #尺度 秒分时日周月
parser.add_argument('--checkpoints', type=str, default='./checknew1/', help='location of model checkpoints') #模型保存路径

# forecasting task 
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length') #输入序列长度
parser.add_argument('--label_len', type=int, default=48, help='start token length') #起始标记的长度
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')  #预测序列的长度
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4') #季节性模式
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False) # 是否对输出数据进行反转

# basic model setting
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')  #编码器输入大小 变量个数N
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size') #解码器输入大小 变量个数N
parser.add_argument('--d_model', type=int, default=16, help='dimension of model') #模型的维度
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')  #编码器层数
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers') #解码器层数
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn') #全连接网络的维度

# Automixer

# model define
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock') #TimesBlock模型中的top_k参数
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')  #Inception模型中的num_kernels参数
parser.add_argument('--c_out', type=int, default=7, help='output size') #输出大小
parser.add_argument('--n_heads', type=int, default=4, help='num of heads') #注意力头的数量
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average') #移动平均窗口的大小
parser.add_argument('--factor', type=int, default=1, help='attn factor') #注意力因子
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)  #是否在编码器中使用蒸馏
parser.add_argument('--dropout', type=float, default=0.1, help='dropout') #dropout的比例
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]') #时间特征编码方式
parser.add_argument('--activation', type=str, default='gelu', help='activation') #激活函数
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')  #是否在编码器中输出注意力
parser.add_argument('--channel_independence', type=int, default=1,
                    help='0: channel dependence 1: channel independence for FreTS model') #FreTS模型中的通道独立性
parser.add_argument('--decomp_method', type=str, default='moving_avg', 
                    help='method of series decompsition, only support moving_avg or dft_decomp')  #时间序列分解方法
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0') #是否使用归一化
parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')  #下采样层数
parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size') #下采样窗口的大小
parser.add_argument('--down_sampling_method', type=str, default='avg', 
                    help='down sampling method, only support avg, max, conv') #下采样方法 平均，最大，卷积

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')  #数据加载器的工作线程数
parser.add_argument('--itr', type=int, default=1, help='experiments times') #实验的次数
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs') #训练的轮数
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data') #
parser.add_argument('--patience', type=int, default=10, help='early stopping patience') # 早停机制
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate') #优化器的学习率
parser.add_argument('--des', type=str, default='test', help='exp description')  #实验描述
parser.add_argument('--loss', type=str, default='MSE', help='loss function') # 损失函数
parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate') #学习率调整方式
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start') #学习率变化的起始百分比
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)#是否使用自动混合精度训练
parser.add_argument('--comment', type=str, default='none', help='com')#注释

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu') #是否使用GPU
parser.add_argument('--gpu', type=int, default=1, help='gpu') #选择的GPU编号
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False) #是否使用多个GPU
# parser.add_argument('--use_multi_gpu', type=bool, help='use multiple gpus', default=True) #是否使用多个GPU
parser.add_argument('--devices', type=str, default='0,1,2,3,4,5', help='device ids of multile gpus') #多个GPU的设备ID

# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='hidden layer dimensions of projector (List)') #投影器的隐藏层维度
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')#投影器中的隐藏层数量

parser.add_argument('--op_num', type=int, default=6)#候选操作层数 //1 1+2 1+2+3 1+2+3+4
parser.add_argument('--seed', type=int, default=21)

args = parser.parse_args()
fix_seed = args.seed #个固定的种子值，用于在后续的随机操作中生成可重复的结果。
#fix_seed = 2021
random.seed(fix_seed) 
torch.manual_seed(fix_seed) #使用固定种子值初始化PyTorch库中的随机数生成器。
np.random.seed(fix_seed)



if args.data == 'm4':
    Exp = Exp_Short_Term_Forecast
else:
    Exp = RunManager


# if args.data == 'm4':
#     args.pred_len = M4Meta.horizons_map[args.seasonal_patterns]  # Up to M4 config
#     args.seq_len = 2 * args.pred_len  # input_len = 2*pred_len
#     args.label_len = args.pred_len
#     args.frequency_map = M4Meta.frequency_map[args.seasonal_patterns]


setting = '{}_{}_seed{}_batch{}_lr{}_epoch{}_in{}_out{}_dm{}_el{}_dl{}_df{}_op{}'.format(
        args.model,
        args.model_id,
        fix_seed,
        args.batch_size,
        args.learning_rate,
        args.train_epochs,
        args.seq_len,
        args.pred_len,
        args.d_model,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.op_num,
        )


path = os.path.join(args.checkpoints, setting)
if not os.path.exists(path):
    os.makedirs(path)



args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]


start_time = time.time()



exp = Exp(args)  # set experiments

if args.is_training == 1:
    
    file_name = path + '/' + 'search.txt'

    f = open(file_name, 'w')
    sys.stdout = f
    print('>>>>>>>start search : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    # exp.search(setting)
    exp.search(setting)
    exp.train(setting)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
elif args.is_training == 2:
    file_name = path + '/' + 'train.txt'
    f = open(file_name, 'w')
    sys.stdout = f
    print('>>>>>>>start train : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    # exp.search(setting)
    exp.train(setting)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()

else:
    file_name = path + '/' + 'test.txt'

    f = open(file_name, 'w')
    sys.stdout = f
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()


print("total time = ", time.time()-start_time,"s")

f.close()