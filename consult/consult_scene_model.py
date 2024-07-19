import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import math
import time
from torch.autograd import Variable
import copy
import random
from torchtext.legacy import data
class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(InputEmbeddings, self).__init__()
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, x):
        return self.embed(x) * math.sqrt(self.embedding_dim)

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, embedding_dim)
        
        position = torch.arange(0., max_len).unsqueeze(1)   # [max_len, 1], 位置编码
        div_term = torch.exp(torch.arange(0., embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)              # 增加维度
        # print(pe.shape)
        
        self.register_buffer('pe', pe)    # 内存中定一个常量，模型保存和加载的时候，可以写入和读出
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False) # Embedding + PositionalEncoding
        return self.dropout(x)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None): # q,k,v: [batch, h, seq_len, d_k]    
    d_k = query.size(-1)                                                  # query的维度
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # 打分机制 [batch, h, seq_len, seq_len]
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9) # mask==0的内容填充-1e9, 使计算softmax时概率接近0
    p_atten = F.softmax(scores, dim = -1)            # 对最后一个维度归一化得分, [batch, h, seq_len, seq_len]
    
    if dropout is not None:
        p_atten = dropout(p_atten)
        
    return torch.matmul(p_atten, value), p_atten  # [batch, h, seq_len, d_k]
    

# 建立一个全连接的网络结构
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, embedding_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert embedding_dim % h == 0 
        
        self.d_k = embedding_dim // h   # 将 embedding_dim 分割成 h份 后的维度
        self.h = h                      # h 指的是 head数量
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)  
        self.dropout = nn.Dropout(p = dropout)
    
    def forward(self, query, key, value, mask = None):  # q,k,v: [batch, seq_len, embedding_dim]
        
        if mask is not None:
            mask = mask.unsqueeze(1)     # [batch, seq_len, 1]
        nbatches = query.size(0)             
        
        # 1. Do all the linear projections(线性预测) in batch from embeddding_dim => h x d_k
        # [batch, seq_len, h, d_k] -> [batch, h, seq_len, d_k]
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) 
                                 for l, x in zip(self.linears, (query, key, value))]
        
        # 2. Apply attention on all the projected vectors in batch.
        # atten:[batch, h, seq_len, d_k], p_atten: [batch, h, seq_len, seq_len]
        attn, p_atten = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # 3. "Concat" using a view and apply a final linear.
        # [batch, h, seq_len, d_k]->[batch, seq_len, embedding_dim]
        attn = attn.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        
        return self.linears[-1](attn)

class MyTransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, p_drop, h, output_size):
        super(MyTransformerModel, self).__init__()
        self.drop = nn.Dropout(p_drop)
        
        # Embeddings, 
        self.embeddings = InputEmbeddings(vocab_size=vocab_size, embedding_dim=embedding_dim)
        # H: [e_x1 + p_1, e_x2 + p_2, ....]
        self.position = PositionalEncoding(embedding_dim, p_drop)
        # Multi-Head Attention
        self.atten = MultiHeadedAttention(h, embedding_dim)       # self-attention-->建立一个全连接的网络结构
        # 层归一化(LayerNorm)
        self.norm = nn.LayerNorm(embedding_dim)
        # Feed Forward
        self.linear = nn.Linear(embedding_dim, output_size)
        # 初始化参数
        self.init_weights()
        
    def init_weights(self):
        init_range = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, inputs, mask):      # 维度均为: [batch, seq_len]
        
        embeded = self.embeddings(inputs) # 1. InputEmbedding [batch, seq_len, embedding_dim] 
#         print(embeded.shape)              # torch.Size([36, 104, 100])
        
        embeded = self.position(embeded)  # 2. PosionalEncoding [batch, seq_len, embedding_dim]
#         print(embeded.shape)              # torch.Size([36, 104, 100])
        
        mask = mask.unsqueeze(2)          # [batch, seq_len, 1]

        # 3.1 MultiHeadedAttention [batch, seq_len. embedding_dim]
        inp_atten = self.atten(embeded, embeded, embeded, mask)  
        # 3.2 LayerNorm [batch, seq_len, embedding_dim]
        inp_atten = self.norm(inp_atten + embeded)
#         print(inp_atten.shape)            # torch.Size([36, 104, 100])
        
        # 4. Masked, [batch, seq_len, embedding_dim]
        inp_atten = inp_atten * mask        # torch.Size([36, 104, 100])         

#         print(inp_atten.sum(1).shape, mask.sum(1).shape)  # [batch, emb_dim], [batch, 1]
        b_avg = inp_atten.sum(1) / (mask.sum(1) + 1e-5)  # [batch, embedding_dim]
        
        return self.linear(b_avg).squeeze()              # [batch, 1] -> [batch]