import torch.nn as nn
import torch


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
