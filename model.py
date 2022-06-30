# -*- coding: utf-8 -*-
"""
Created on Tue May 31 10:49:16 2022

@author: njucx
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import math
import copy
from data_process import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda')

#%% Transformer
        
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        print('postion', position.size())
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) #[max_len, 1, d_model]
        print('pe', pe.size())
        self.register_buffer('pe', pe)


    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> x = torch.randn(50, 128, 18)
            >>> pos_encoder = PositionalEncoding(18, 0.1)
            >>> output = pos_encoder(x)
        """
        
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)                     


class Transformer(nn.Module):
    r"""
    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        nlayers: the number of sublayers of both encoder and decoder
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).

    """

    def __init__(self, d_model, nhead, dim_feedforward, nlayers, dropout) -> None:
        super(Transformer, self).__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, norm_first=True)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers, encoder_norm)
        
        # decoder
        decoder_layer = DecoderLayer(d_model, nhead, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(decoder_layer, nlayers, decoder_norm)
        self.linear = nn.Linear(18, 1)

        self.d_model = d_model
        self.nhead = nhead
    
    def init_weights(self):
        initrange = 0.1
        nn.init.xavier_uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, src_mask) -> Tensor:
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).

        Shape:
            - src: :(S, N, E)

            - output: :(T, N, E)

            S is the source sequence length, 
            T is the target sequence length, 
            N is the batch size, 
            E is the feature number
        """

        if src.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src must be equal to d_model")

        memory = self.encoder(src, mask=src_mask)
        output = self.decoder(memory)
        output = output.permute(2, 1, 0)
        output = self.linear(output)
        output = output.permute(2, 1, 0)
        return output
    
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout) -> None:
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, memory) -> Tensor:

        x = memory
        x = x + self._sa_block(self.norm1(x))
        x = x + self._mha_block(self.norm2(x))
        x = x + self._ff_block(self.norm3(x))

        return x

    # self-attention block
    def _sa_block(self, x) -> Tensor:
        x = self.self_attn(x, x, x, need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x) -> Tensor:
        x = self.multihead_attn(x, x, x, need_weights=False)[0]
        return self.dropout2(x)
    
    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class Decoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(Decoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, memory: Tensor) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        for mod in self.layers:
            output = mod(memory)

        if self.norm is not None:
            output = self.norm(output)

        return output
 
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss