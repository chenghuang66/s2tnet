from utilies import _get_clones, _get_activation_fn
import torch
import torch.nn as nn
import numpy as np
import math
from functools import partial
import torch.nn.functional as F

from sttf_layer import STAttention,TransitionFunction

class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead,
                 feedforward_ratio=1, dropout=0.1, module='encoder',activation="relu"):
        super().__init__()
        
        # Implementation of Feedforward model
        ff=int(d_model*feedforward_ratio)
        self.linear1 = nn.Linear(d_model, ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.module=module
        if module is 'decoder':
            self.self_attn = STAttention(
                nhead, d_model, mode = 'temporal', dropout=dropout, attn_type = 'norm')
            self.mh_attn = STAttention(
                nhead, d_model, mode = 'temporal', dropout=dropout, attn_type = 'norm')
            self.norm3 = nn.LayerNorm(d_model)
            self.dropout3 = nn.Dropout(dropout)
            self.positionwise_feed_forward = TransitionFunction(
                d_model, ff, d_model,
                layer_config='cc', padding = 'left', dropout=dropout)

        else:

            self.positionwise_feed_forward = TransitionFunction(
                d_model, ff, d_model,
                layer_config='cc', padding = 'both', dropout=dropout)            
            self.s_attn = STAttention(nhead, d_model, mode = 'temporal', dropout=dropout)

    def forward(self, query, key, att_mask=None, key_padding_mask=None):

        if self.module is 'encoder':

            src2, attn1 = self.s_attn(query, query, query)            
            src = self.norm1(query + self.dropout1(src2))
            src2 = self.positionwise_feed_forward(src)
            src = self.norm2(src + self.dropout2(src2))

        else:

            src2, attn1 = self.self_attn(
                query, query, query, att_mask=att_mask,key_padding_mask=key_padding_mask)
            query = query + self.dropout1(src2)
            src = self.norm1(query) 
            src3, attn2 = self.mh_attn(src, key, key)
            src = src + self.dropout3(src3)
            src = self.norm3(src)
            src2 = self.positionwise_feed_forward(src)

            src = self.norm2(src + self.dropout2(src2))

        return src, None, None
            

class TransformerModel(nn.Module):
    def __init__(self, layer, num_layers, module, norm=None):
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.module = module

    def forward(self, query, key, att_mask=None, key_padding_mask=None):
        
        output = query

        atts1 = []
        atts2 = []

        for i in range(self.num_layers):
            if self.module is 'encoder':
                key = output
            
            output, attn1, attn2 = self.layers[i](
                output, key, att_mask=att_mask, key_padding_mask=key_padding_mask)

            atts1.append(attn1)
            atts2.append(attn2)
        
        if self.norm:
            output = self.norm(output)

        return output, atts1, atts2