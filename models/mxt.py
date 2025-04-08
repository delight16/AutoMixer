import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys
from collections import namedtuple
import ruamel.yaml as yaml


os.chdir(sys.path[0])

class BasicOp(nn.Module): #一个基类 所有候选操作都继承这个基类
    def __init__(self, **kwargs):
        super(BasicOp, self).__init__() 

    def forward(self, inputs, **kwargs): 
        raise NotImplementedError 

    def __repr__(self): 
        cfg = [] 
        for (key, value) in self.setting:  
            cfg += [str(key) + ': ' + str(value)] 
        return str(self.type) + '(' + ', '.join(cfg) + ')'

    @property
    def type(self):
        raise NotImplementedError

    @property
    def setting(self):
        raise NotImplementedError

def create_op(op_name, d_model, d_ff):
    name2op = {
        'Zero': lambda: Zero(),
        'Identity': lambda: Identity(),
        'Linear': lambda: Linear(d_model, d_ff)
        # 'FcDown': lambda: FcDown(),
        # 'MaxDown': lambda: MaxDown(),
        # 'AvgDown': lambda: AvgDown(),
        # 'ConvDown': lambda: ConvDown(),
        
        
    }
    op = name2op[op_name]()
    return op

def add_indent(str_, num_spaces): #用于给字符串添加指定数量的空格缩进
    s = str_.split('\n')
    s = [(num_spaces * ' ') + line for line in s]
    return '\n'.join(s)

#config = Config()
class MixedOp(BasicOp):
    #def __init__(self, in_channels, out_channels, candidate_op_profiles):
    def __init__(self, configs, candidate_op_profiles):
        super(MixedOp, self).__init__()
       # self._in_channels = in_channels
       # self._out_channels = out_channels

        self._num_ops = len(candidate_op_profiles)
        self._candidate_op_profiles = candidate_op_profiles #表示候选操作配置列表，用于选择要执行的操作及其配置
        self._candidate_ops = nn.ModuleList()
        for (op_name, profile) in self._candidate_op_profiles:
            #self._candidate_ops += [create_op(op_name, self._in_channels, self._out_channels, profile)]
            self._candidate_ops += [create_op(op_name, configs.d_model, configs.d_ff)]
        #所有的操作实例
        # self._candidate_alphas = nn.Parameter(torch.normal(mean=torch.zeros(self._num_ops), std=1), requires_grad=True)
        self._candidate_alphas = nn.Parameter(torch.zeros(self._num_ops), requires_grad=True) #使用 nn.Parameter 创建一个与候选操作数量相同大小的零张量，要求该参数可进行梯度计算。
        
        # 搜索设置
        # probs = F.softmax(self._candidate_alphas.data, dim=0)
        # self._sample_idx = torch.multinomial(probs, 3, replacement=True).cpu().numpy() #抽样
#        self._sample_idx =  np.array([2,2])
        # if configs.is_training == 1:
        self._sample_idx = np.arange(self._num_ops)
        #x = 1
        # 训练时设置的
        if configs.is_training == 2:
            probs = F.softmax(self._candidate_alphas.data, dim=0)
            op = torch.argmax(probs).item() #返回概率最高的操作
            self._sample_idx = np.array([op], dtype=np.int32)



    def forward(self, inputs):
        probs = F.softmax(self._candidate_alphas[self._sample_idx], dim=0)
        output = 0
        for i, idx in enumerate(self._sample_idx):
            output += probs[i] * self._candidate_ops[idx](inputs)
        return output

    def arch_parameters(self):
        yield self._candidate_alphas


    def __repr__(self):
        # mode info
        out_str = ''
        #out_str += 'mode: ' + str(self._mode) + str(self._sample_idx) + ',\n'
        # probability of each op & its info
        probs = F.softmax(self._candidate_alphas.data, dim=0)
        for i in range(self._num_ops):
            out_str += 'op:%d, prob: %.3f, info: %s,' % (i, probs[i].item(), self._candidate_ops[i])
            if i + 1 < self._num_ops:
                out_str += '\n'

        out_str = 'mixed_op {\n%s\n}' % add_indent(out_str, 4)
        return out_str


class Cell(nn.Module):
    def __init__(self, configs, num_mixed_ops, candidate_op_profiles):
        super(Cell, self).__init__()
        #self._channels = channels   
        self._num_mixed_ops = num_mixed_ops    
        self._mixed_ops = nn.ModuleList() #为什么要这个混合模块呢？ 搜索不能实现这一步吗？ 还是说darts就是要定义这种混合块
        for i in range(self._num_mixed_ops):   
            self._mixed_ops += [MixedOp(configs, candidate_op_profiles)] 


    # def forward(self, x, s):
    #     # calculate outputs
    #     node_idx = 0  
    #     #current_output = 0   
    #     current_output = s  

    #     node_outputs = [x]   
    #     #这里是Darts的网络结构  应该在这里做修改 ？
    #     for i in range(self._num_mixed_ops): #经过六个混合操作
    #         current_output = current_output + self._mixed_ops[i]([node_outputs[node_idx]]) #这一步是网络预测过程 
    #         if node_idx + 1 >= len(node_outputs):  
    #             node_outputs += [current_output] #长度为4  那么这四个分别是h0-h3 h0就是输入
    #             #current_output = 0
    #             current_output = s
    #             node_idx = 0
    #         else:
    #             node_idx += 1

    #     if node_idx != 0:
    #         node_outputs += [current_output]

    #     ret = 0 #ret是每一个cell的输出 要混合h0-h3的值
    #     # for x in node_outputs[:]:
    #     for x in node_outputs[1:]:
    #         ret = ret + x
    #     return ret
    
    def forward(self, x):
        # calculate outputs
        node_idx = 0  
        #current_output = 0   
        current_output = 0 

        node_outputs = [x]   
        #这里是Darts的网络结构  应该在这里做修改 ？
        for i in range(self._num_mixed_ops): #经过六个混合操作
            current_output += self._mixed_ops[i]([node_outputs[node_idx]]) #这一步是网络预测过程 
            if node_idx + 1 >= len(node_outputs):  
                node_outputs += [current_output] #长度为4  那么这四个分别是h0-h3 h0就是输入
                #current_output = 0
                current_output = 0
                node_idx = 0
            else:
                node_idx += 1

        if node_idx != 0:
            node_outputs += [current_output]

        ret = 0 #ret是每一个cell的输出 要混合h0-h3的值
        # for x in node_outputs[:]:
        for x in node_outputs[:]:
            ret = ret + x
        return ret

    
    def arch_parameters(self):
        for i in range(self._num_mixed_ops):
            for p in self._mixed_ops[i].arch_parameters():
                yield p

    def __repr__(self):
        edge_cnt = 0
        out_str = []
        for i in range(self._num_mixed_ops):
            out_str += ['mixed_op: %d\n%s' % (i, self._mixed_ops[i])]

        out_str = 'Cell {\n%s\n}' % add_indent('\n'.join(out_str), 4)
        return out_str



class Identity(BasicOp): #恒等操作
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs, **kwargs):
        x = 0 
        for i in inputs: x += i   #将输入的所有元素相加并返回结果
        return x 

    @property
    def type(self):
        return 'identity'

    @property
    def setting(self):
        return [] #没有额外设置项

class Zero(BasicOp): #将输入置零
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, inputs, **kwargs):
        return torch.zeros_like(inputs[0]) #创建一个和输入 inputs 的第一个元素形状相同的全零张量

    @property
    def type(self):
        return 'zero'

    @property
    def setting(self):
        return []

class Linear(BasicOp): #将输入置零
    def __init__(self, d_model, d_ff):
        super(Linear, self).__init__()
        # self._linelayer = nn.Linear(96, 96)
        self._linelayer = nn.Sequential( 
                nn.Linear(in_features=d_model, out_features=d_ff),
                nn.GELU(),
                nn.Linear(in_features=d_ff, out_features=d_model),
            )

    def forward(self, inputs, **kwargs):
        x = 0 
        for i in inputs: x += i
        x = x.permute(0, 2, 1)
        out = self._linelayer(x)
        out = out.permute(0, 2, 1)
        return out 

    @property
    def type(self):
        return 'Linear'

    @property
    def setting(self):
        return []

class FcDown(BasicOp): #将输入置零
    def __init__(self):
        super(FcDown, self).__init__()
        # self._linelayer = nn.Linear(96, 96)
        self._layer = nn.Sequential(
                    torch.nn.Linear(
                        96,
                        48,
                    ),
                    nn.GELU(), #非线性映射
                    torch.nn.Linear(
                        48,
                        48,
                    ), #维度恢复
                )

    def forward(self, inputs, **kwargs):
        x = 0 
        for i in inputs: x += i
        out = self._layer(x)
        return out 

    @property
    def type(self):
        return 'FcDown'

    @property
    def setting(self):
        return []

class MaxDown(BasicOp): 
    def __init__(self):
        super(MaxDown, self).__init__()
        self._layer = torch.nn.MaxPool1d(2, return_indices=False)

    def forward(self, inputs, **kwargs):
        x = 0 
        for i in inputs: x += i
        out = self._layer(x)
        return out 

    @property
    def type(self):
        return 'MaxDown'

    @property
    def setting(self):
        return []

class AvgDown(BasicOp): 
    def __init__(self):
        super(AvgDown, self).__init__()
        self._layer = torch.nn.AvgPool1d(2)

    def forward(self, inputs, **kwargs):
        x = 0 
        for i in inputs: x += i
        out = self._layer(x)
        return out 

    @property
    def type(self):
        return 'AvgDown'

    @property
    def setting(self):
        return []

class ConvDown(BasicOp): 
    def __init__(self):
        super(ConvDown, self).__init__()
        self._layer = nn.Conv1d(in_channels=128, out_channels=128,
                                  kernel_size=3, padding=1,
                                  stride=2,
                                  padding_mode='circular',
                                  bias=False)

    def forward(self, inputs, **kwargs):
        x = 0 
        for i in inputs: x += i
        out = self._layer(x)
        return out 

    @property
    def type(self):
        return 'ConvDown'

    @property
    def setting(self):
        return []
    

def dict_to_namedtuple(dic: dict): #接受一个dict类型的参数dic。
    return namedtuple('tuple', dic.keys())(**dic) 
    #使用namedtuple函数创建一个命名元组，并将dic字典的键作为元组的字段名称。
    #通过**dic将字典的值作为关键字参数传递给命名元组的构造函数，创建一个命名元组对象并返回。

class Config:  #Config 类 在main中使用
    def __init__(self): #构造函数
        pass

    def load_config(self, config):  #config应该是一个路径 加载yaml文件
        with open(config, 'r') as f: #使用open函数打开config文件 
            setting = yaml.load(f, Loader=yaml.RoundTripLoader) #字典
            #用yaml模块的load函数从文件中加载配置数据。
            #yaml是一种用于序列化数据的格式，load函数将配置文件解析为一个Python对象，并将其赋值给setting变量。
        
        self.model = dict_to_namedtuple(setting['model'])



config = Config()


config.load_config('layers/train.yaml')

candidate_op_profiles = config.model.candidate_op_profiles

# enc_out_list = torch.load('tensor_file.pt') # 32*96*128 32*48*128



# layer = Cell(2, candidate_op_profiles).cuda() 

# out = layer(enc_out_list[0].permute(0, 2, 1), enc_out_list[1].permute(0, 2, 1)) 






# self.down_sampling_layers = torch.nn.ModuleList(
#     [
#         nn.Sequential(
#             torch.nn.Linear(
#                 configs.seq_len // (configs.down_sampling_window ** i), #96/2^0 96
#                 configs.seq_len // (configs.down_sampling_window ** (i + 1)), #96/2^1 48
#             ),
#             nn.GELU(), #非线性映射
#             torch.nn.Linear(
#                 configs.seq_len // (configs.down_sampling_window ** (i + 1)),
#                 configs.seq_len // (configs.down_sampling_window ** (i + 1)),
#             ), #维度恢复

#         )
#         for i in range(configs.down_sampling_layers)
#     ]
# )
# self.up_sampling_layers = torch.nn.ModuleList(
#     [
#         nn.Sequential(
#             torch.nn.Linear(
#                 configs.seq_len // (configs.down_sampling_window ** (i + 1)), #48
#                 configs.seq_len // (configs.down_sampling_window ** i), #96
#             ),
#             nn.GELU(),
#             torch.nn.Linear(
#                 configs.seq_len // (configs.down_sampling_window ** i), #96
#                 configs.seq_len // (configs.down_sampling_window ** i),#96
#             ),
#         )
#         for i in reversed(range(configs.down_sampling_layers))
#     ])







