import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import math
torch.cuda.is_available()
DEVICE = torch.device("cuda:0")


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, k_dim, v_dim, n_layers, n_heads, hidden_nodes):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, embed_dim)  ## 这个其实就是去定义生成一个矩阵，大小是 src_vocab_size * d_model
        self.pos_emb = PositionalEncoding(embed_dim)  ## 位置编码情况，这里是固定的正余弦函数，也可以使用类似词向量的nn.Embedding获得一个可以更新学习的位置编码

        self.layers = nn.ModuleList([EncoderLayer(embed_dim, k_dim, v_dim, n_heads, hidden_nodes) for _ in range(n_layers)])  ## 使用ModuleList对多个encoder进行堆叠，因为后续的encoder并没有使用词向量和位置编码，所以抽离出来；
        self.output_num = 128
        self.out = nn.Linear(self.output_num, 1).to(DEVICE)
        self.sigmoid = nn.Sigmoid()

        self.embed_dim = embed_dim

    def forward(self, enc_inputs):
        enc_inputs = enc_inputs.long()
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        # enc_outputs =  [batch_size*2, sentence_len, embed_dim]
        enc_outputs = enc_outputs.contiguous()
        enc_outputs = enc_outputs.view(-1, enc_outputs.size(1) * self.embed_dim)
        enc_outputs = enc_outputs.to(DEVICE)
        enc_outputs = nn.Linear(enc_outputs.size(1), self.output_num)(enc_outputs)
        features = enc_outputs
        enc_outputs = self.out(enc_outputs)
        y_score = self.sigmoid(enc_outputs)
        y_pred = torch.round(y_score)
        return y_score, y_pred, features


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, k_dim, v_dim, n_heads, hidden_nodes):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(embed_dim, k_dim, v_dim, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(embed_dim, hidden_nodes)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn



class MultiHeadAttention(nn.Module):
    def __init__(self, f_dim, k_dim, v_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        ## 输入进来的QKV是相等的，我们会使用映射linear做一个映射得到参数矩阵Wq, Wk,Wv
        self.W_Q = nn.Linear(f_dim, k_dim * n_heads)
        self.W_K = nn.Linear(f_dim, k_dim * n_heads)
        self.W_V = nn.Linear(f_dim, v_dim * n_heads)
        self.linear = nn.Linear(n_heads * v_dim, f_dim)
        self.layer_norm = nn.LayerNorm(f_dim)
        self.n_heads = n_heads
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.embed_dim = f_dim

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.k_dim).transpose(1, 2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.k_dim).transpose(1, 2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.v_dim).transpose(1, 2)  # v_s: [batch_size x n_heads x len_k x d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        context, attn = ScaledDotProductAttention(self.embed_dim)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.v_dim)  # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn  # output: [batch_size x len_q x d_model]


class ScaledDotProductAttention(nn.Module):
    def __init__(self, f_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.f_dim = f_dim

    def forward(self, Q, K, V, attn_mask):
        ## 输入进来的维度分别是 [batch_size x n_heads x len_q x d_k]  K： [batch_size x n_heads x len_k x d_k]  V: [batch_size x n_heads x len_k x d_v]
        ##首先经过matmul函数得到的scores形状是 : [batch_size x n_heads x len_q x len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.f_dim)

        ## 然后关键词地方来了，下面这个就是用到了我们之前重点讲的attn_mask，把被mask的地方置为无限小，softmax之后基本就是0，对q的单词不起作用
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        ## 位置编码的实现其实很简单，直接对照着公式去敲代码就可以，下面这个代码只是其中一种实现方式；
        ## 从理解来讲，需要注意的就是偶数和奇数在公式上有一个共同部分，我们使用log函数把次方拿下来，方便计算；
        ## pos代表的是单词在句子中的索引，这点需要注意；比如max_len是128个，那么索引就是从0，1，2，...,127
        ##假设我的demodel是512，2i那个符号中i从0取到了255，那么2i对应取值就是0,2,4...510
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)## 这里需要注意的是pe[:, 0::2]这个用法，就是从0开始到最后面，补长为2，其实代表的就是偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)##这里需要注意的是pe[:, 1::2]这个用法，就是从1开始到最后面，补长为2，其实代表的就是奇数位置
        ## 上面代码获取之后得到的pe:[max_len*d_model]

        ## 下面这个代码之后，我们得到的pe形状是：[max_len*1*d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)  ## 定一个缓冲区，其实简单理解为这个参数不更新就可以

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, f_dim, hidden_nodes):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=f_dim, out_channels=hidden_nodes, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_nodes, out_channels=f_dim, kernel_size=1)
        self.layer_norm = nn.LayerNorm(f_dim)

    def forward(self, inputs):
        residual = inputs # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)