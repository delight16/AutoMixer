import numpy as np


class Scaler:
    def __init__(self, data, missing_value=np.inf): #计算数据中非缺失值的均值和标准差，并将其存储在 mean 和 std 成员变量中
        values = data[data != missing_value] 
        self.mean = values.mean()
        self.std = values.std()

    def transform(self, data): #减去均值并除以标准差进行归一化
        return (data - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, data): #将缩放后的数据进行逆操作，将数据恢复到原始的尺度
        return data * self.std + self.mean


def add_indent(str_, num_spaces): #用于给字符串添加指定数量的空格缩进
    s = str_.split('\n')
    s = [(num_spaces * ' ') + line for line in s]
    return '\n'.join(s)


def num_parameters(layer): #计算神经网络层中的参数数量。它接受一个神经网络层 layer，遍历层中的参数，并将每个参数的大小相乘以获得总参数数量。
    def prod(arr):
        cnt = 1
        for i in arr:
            cnt = cnt * i
        return cnt

    cnt = 0
    for p in layer.parameters():
        cnt += prod(p.size())
    return cnt
