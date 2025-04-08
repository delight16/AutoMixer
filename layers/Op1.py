import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys
from collections import namedtuple
import ruamel.yaml as yaml
import math

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
        # 'Linear': lambda: Linear(d_model, d_ff)
        # 'FcDown': lambda: FcDown(),
        # 'MaxDown': lambda: MaxDown(),
        # 'AvgDown': lambda: AvgDown(),
        # 'ConvDown': lambda: ConvDown(),
        # 'trans': lambda: InformerLayer(d_model),
        # 's_trans': lambda: SpatialInformerLayer(d_model),
        
        
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
        self.configs = configs
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
        # self._sample_idx =  np.array([2,2])
        # if configs.is_training == 1:
        self._sample_idx = np.arange(self._num_ops)
        #x = 1
        # # 训练时设置的
        # if configs.is_training == 2:
        #     probs = F.softmax(self._candidate_alphas.data, dim=0)
        #     op = torch.argmax(probs).item() #返回概率最高的操作
        #     self._sample_idx = np.array([op], dtype=np.int32)
        # if(self.configs.is_training == 2): 



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
    
    def forward(self, x):
        # calculate outputs
        
        node_idx = 0  
        #current_output = 0   
        current_output = 0 

        node_outputs = [x]   
        #这里是Darts的网络结构  应该在这里做修改
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


######################################################################
# Informer encoder layer
######################################################################
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TriangularCausalMask():
    def __init__(self, B, L):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(DEVICE)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=3, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        """
        :param Q: [b, heads, T, d_k]
        :param K: 采样的K? 长度为Ln(L_K)?
        :param sample_k: c*ln(L_k), set c=3 for now
        :param n_top: top_u queries?
        :return: Q_K and Top_k query index
        """
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

        # kernel_size = 3
        # pad = (kernel_size - 1) // 2
        # self.query_projection = SepConv1d(d_model, d_model, kernel_size, padding=pad)
        # self.key_projection = SepConv1d(d_model, d_model, kernel_size, padding=pad)
        # self.value_projection = SepConv1d(d_model, d_model, kernel_size, padding=pad)

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # queries = queries.transpose(-1, 1)
        # keys = keys.transpose(-1, 1)
        # values = values.transpose(-1, 1)
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn



class InformerLayer(nn.Module):
    def __init__(self, d_model, d_ff=32, dropout=0., n_heads=4, activation="relu", output_attention=False):
        super(InformerLayer, self).__init__()
        # d_ff = d_ff or 4*d_model
        self.attention = AttentionLayer(
            ProbAttention(False, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads)
        self.mlp = nn.Conv2d(1, d_model, kernel_size=(1, 1)) #change
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        # self.pe = PositionalEmbedding(d_model)
        self.d_model = d_model

    def forward(self, x, attn_mask=None):
        x = x[0]
        x = x.unsqueeze(-1).permute(0, 3, 1, 2) #change 
        x = self.mlp(x)
        b, C, N, T = x.shape
        x = x.permute(0, 2, 3, 1)  # [64, 207, 12, 32]
        x = x.reshape(-1, T, C)  # [64*207, 12, 32]
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        # x = x * math.sqrt(self.d_model)
        # x = x + self.pe(x)
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x+y)

        output = output.reshape(b, -1, T, C)
        output = output.permute(0, 3, 2, 1)
        
        output = output.sum(dim=-1) #change
        return output


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff=32, dropout=0., n_heads=4, activation="relu", output_attention=False):
        super(TransformerLayer, self).__init__()
        # d_ff = d_ff or 4*d_model
        self.attention = AttentionLayer(
            FullAttention(False, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        # self.pe = PositionalEmbedding(d_model)
        self.d_model = d_model

    def forward(self, x, attn_mask=None):
        b, C, N, T = x.shape
        x = x.permute(0, 2, 3, 1)  # [64, 207, 12, 32]
        x = x.reshape(-1, T, C)  # [64*207, 12, 32]
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        # x = x * math.sqrt(self.d_model)
        # x = x + self.pe(x)
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x+y)

        output = output.reshape(b, -1, T, C)
        output = output.permute(0, 3, 1, 2)

        return output


######################################################################
# Spatial Transformer
######################################################################


class SpatialTransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff=32, dropout=0., n_heads=4, activation="relu", output_attention=False):
        super(SpatialTransformerLayer, self).__init__()
        self.attention = SpatialAttentionLayer(
            SpatialFullAttention(attention_dropout=dropout, output_attention=output_attention), d_model, n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.d_model = d_model

    def forward(self, x):
        b, C, N, T = x.shape
        x = x.permute(0, 3, 2, 1)  # [64, 12, 207, 32]
        x = x.reshape(-1, N, C)  # [64*12, 207, 32]
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(x, x, x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x+y)

        output = output.reshape(b, -1, N, C)
        output = output.permute(0, 3, 2, 1)

        return output




class SpatialInformerLayer(nn.Module):
    def __init__(self, d_model, d_ff=32, dropout=0., n_heads=4, activation="relu", output_attention=False):
        super(SpatialInformerLayer, self).__init__()
        self.attention = SpatialAttentionLayer(
            SpatialProbAttention(attention_dropout=dropout, output_attention=output_attention), d_model, n_heads)
        self.mlp = nn.Conv2d(1, d_model, kernel_size=(1, 1)) #change
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        # self.pe = PositionalEmbedding(d_model)
        self.d_model = d_model

    def forward(self, x):
        x = x[0]
        x = x.unsqueeze(-1).permute(0, 3, 1, 2) #change 
        x = self.mlp(x)
        # x = x.unsqueeze(-1).permute(0, 3, 2, 1) #change
        b, C, N, T = x.shape
        x = x.permute(0, 3, 2, 1)  # [64, 12, 207, 32]
        x = x.reshape(-1, N, C)  # [64*12, 207, 32]
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(x, x, x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x+y)

        output = output.reshape(b, -1, N, C)
        output = output.permute(0, 2, 1, 3)

        # output = output.squeeze(-1) #change
        output = output.sum(dim=-1) #change
        return output


class SpatialAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(SpatialAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values):
        # shape=[b*T, N, C]
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values)
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class SpatialFullAttention(nn.Module):
    def __init__(self, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(SpatialFullAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # 在这里加上fixed邻接矩阵？
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class SpatialProbAttention(nn.Module):
    def __init__(self, factor=3, scale=None, attention_dropout=0.1, output_attention=False):
        super(SpatialProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        """
        :param Q: [b, heads, T, d_k]
        :param K: 采样的K? 长度为Ln(L_K)?
        :param sample_k: c*ln(L_k), set c=3 for now
        :param n_top: top_u queries?
        :return: Q_K and Top_k query index
        """
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        # V_sum = V.sum(dim=-2)
        V_sum = V.mean(dim=-2)  # # [256*12, 4, 8]
        contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()  # [256*12, 4, 207, 8]
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores) [256*12, 4, 18, 207]

        # print(context_in.shape)  # [256*12, 4, 207, 8]
        # print(torch.matmul(attn, V).shape)  # [256*12, 4, 18, 8]
        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        # print(index.shape)  # [256*12, 4, 18]

        # add scale factor
        scale = self.scale or 1. / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale  # [256*12, 4, 18, 207] 18=sqrt(207)*3
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q)

        return context.transpose(2, 1).contiguous(), attn






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

