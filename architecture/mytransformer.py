import math
import os

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from architecture.network import Classifier_1fc, DimReduction, DimReduction1
from einops import repeat
from .nystrom_attention import NystromAttention
from modules.emb_position import *

def pos_enc_1d(D, len_seq):
    
    if D % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(D))
    pe = torch.zeros(len_seq, D)
    position = torch.arange(0, len_seq).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, D, 2, dtype=torch.float) *
                         -(math.log(10000.0) / D)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def sigmax(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    Applies element-wise sigmoid to x and normalizes along the specified dim.
    Returns a probability vector (non-negative, sums to 1 along dim).

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension to normalize over.
        eps (float): Small value to prevent division by zero.

    Returns:
        torch.Tensor: Probability vector.
    """
    sigmoid_x = torch.sigmoid(x)
    norm = sigmoid_x.sum(dim=dim, keepdim=True)
    return sigmoid_x / (norm + eps)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MLP_single_layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_single_layer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x


class ACMIL_MHA(nn.Module):
    def __init__(self, conf, n_token=1, n_masked_patch=0, mask_drop=0):
        super(ACMIL_MHA, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.sub_attention = nn.ModuleList()
        for i in range(n_token):
            self.sub_attention.append(MutiHeadAttention(conf.D_inner, 8, n_masked_patch=n_masked_patch, mask_drop=mask_drop))
        self.bag_attention = MutiHeadAttention_modify(conf.D_inner, 8)
        self.q = nn.Parameter(torch.zeros((1, n_token, conf.D_inner)))
        nn.init.normal_(self.q, std=1e-6)
        self.n_class = conf.n_class

        self.classifier = nn.ModuleList()
        for i in range(n_token):
            self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, 0.0))
        self.n_token = n_token
        self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, 0.0)

    def forward(self, input):
        input = self.dimreduction(input)
        q = self.q
        k = input
        v = input
        outputs = []
        attns = []
        for i in range(self.n_token):
            feat_i, attn_i = self.sub_attention[i](q[:, i].unsqueeze(0), k, v)
            outputs.append(self.classifier[i](feat_i))
            attns.append(attn_i)

        attns = torch.cat(attns, 1)
        feat_bag = self.bag_attention(v, attns.softmax(dim=-1).mean(1, keepdim=True))

        return torch.cat(outputs, dim=0), self.Slide_classifier(feat_bag), attns


class MHA(nn.Module):
    def __init__(self, conf):
        super(MHA, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = MutiHeadAttention(conf.D_inner, 8)
        self.q = nn.Parameter(torch.zeros((1, 1, conf.D_inner)))
        nn.init.normal_(self.q, std=1e-6)
        self.n_class = conf.n_class
        self.classifier = Classifier_1fc(conf.D_inner, conf.n_class, 0.0)

    def forward(self, input):
        input = self.dimreduction(input)
        q = self.q
        k = input
        v = input
        feat, attn = self.attention(q, k, v)
        output = self.classifier(feat)

        return output



class MutiHeadAttention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.1,
        softmax: bool = True
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.softmax = softmax
    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        # breakpoint()
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)

        attn_out = attn
        if self.softmax:
            attn = F.softmax(attn, dim=-1)
        else:
            attn = sigmax(attn, dim=-1)
        # Get output
        out1 = attn @ v
        out1 = self._recombine_heads(out1)
        out1 = self.out_proj(out1)
        out1 = self.dropout(out1)
        out1 = self.layer_norm(out1)
        return out1[0], attn_out[0]



class ACMIL_MYMHA(nn.Module):
    def __init__(self, conf, n_token=1, n_drop=0):
        super(ACMIL_MYMHA, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner, numLayer_Res=0)
        self.sub_attention = nn.ModuleList()
        for i in range(n_token):
            self.sub_attention.append(MutiHeadAttention(conf.D_inner, 8,  softmax=True))

        # self.bag_attention = MutiHeadAttention_modify(conf.D_inner, 8)
        self.bag_attention = MutiHeadAttention(conf.D_inner, 8, softmax=True)
        
        self.q = nn.Parameter(torch.zeros((1, n_token, conf.D_inner)))
        self.cls = nn.Parameter(torch.zeros((1, 1, conf.D_inner)))
        nn.init.normal_(self.q, std=1e-6)
        self.n_class = conf.n_class
        self.n_drop = n_drop
        self.classifier = nn.ModuleList()
        for i in range(n_token):
            self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, 0.0))
        self.n_token = n_token
        self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, 0.0)

    def forward(self, input):
        input = self.dimreduction(input)
        q = self.q # shape [1, N_token, 128]
        k = input # [1, 15550, 128]
        v = input
        cls = self.cls
        outputs = []
        attns = []
        feats = []
        Feature_Drop = torch.randperm(self.n_token)[:self.n_drop]
        for i in range(self.n_token):
            feat_i, attn_i = self.sub_attention[i](q[:, i].unsqueeze(0), k, v) # q[:, i].unsqueeze(0) shape: [1, 1, 128]
            outputs.append(self.classifier[i](feat_i))
            attns.append(attn_i) # attn_i shape [8, 1, 15550]
            if self.training:
                if i not in Feature_Drop:
                    feats.append(feat_i)
            else:
                feats.append(feat_i)
            
        feats = torch.cat(feats, 0)[None,:] # shape 5, 128

        attns = torch.cat(attns, 1) # shape 8, 5, 15550

        feat_bag, _ = self.bag_attention(cls, feats, feats)
        
        return torch.cat(outputs, dim=0), self.Slide_classifier(feat_bag), attns


class ACMIL_MYMHA_EMA(nn.Module):
    def __init__(self, conf, n_token=1, n_drop=0):
        super(ACMIL_MYMHA_EMA, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner, numLayer_Res=0)
        self.sub_attention = nn.ModuleList()
        for i in range(n_token):
            self.sub_attention.append(MutiHeadAttention(conf.D_inner, 8, softmax=False))

        # self.bag_attention = MutiHeadAttention_modify(conf.D_inner, 8)
        self.bag_attention = MutiHeadAttention(conf.D_inner, 8, softmax=True)

        self.q = nn.Parameter(torch.zeros((1, n_token, conf.D_inner)))
        self.cls = nn.Parameter(torch.zeros((1, 1, conf.D_inner)))
        nn.init.normal_(self.q, std=1e-6)
        self.n_class = conf.n_class

        self.classifier = nn.ModuleList()
        for i in range(n_token):
            self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, 0.0))
        self.n_token = n_token
        self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, 0.0)
        self.n_drop = n_drop
    def forward(self, input):
        input = self.dimreduction(input)
        q = self.q # shape [1, N_token, 128]
        k = input # [1, 15550, 128]
        v = input
        attns = []
        for i in range(self.n_token):
            _, attn_i = self.sub_attention[i](q[:, i].unsqueeze(0), k, v) # q[:, i].unsqueeze(0) shape: [1, 1, 128]
            attns.append(attn_i) # attn_i shape [8, 1, 15550]
        
        attns = torch.cat(attns, 1) # shape 8, 5, 15550

        return attns
    