from utilies import _get_clones, _get_activation_fn,conv_init,bn_init
import torch
import torch.nn as nn
import numpy as np
import math
from functools import partial
import torch.nn.functional as F

from multi_attention_forward import *
from sttf_layer import *
from seq2seq_transformer import TransformerLayer, TransformerModel

class s2tnet(nn.Module):
    def __init__(self, in_chans=6,output_chans=2, d_model=32, nhead=8, feedforward_ratio=1, drop_rate=0.1):
        super().__init__()

        self.input_embedding = Embeddings(in_chans, d_model)
        self.output_embedding = Embeddings(output_chans, d_model)
        self.en_position = PositionalEncoding(d_model, drop_rate)
        self.de_position = PositionalEncoding(d_model, drop_rate)
        self.PostionEmbedding = TimeEmbeddingSine(d_model)
        self.output_linear = nn.Linear(d_model, output_chans)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.pos_drop1 = nn.Dropout(drop_rate)
        self.pos_drop2 = nn.Dropout(drop_rate)

        encoder_num_layers=6
        decoder_num_layers=6

        self.encoder_layer=TransformerLayer(d_model,nhead,feedforward_ratio,drop_rate,module='encoder')
        self.decoder_layer=TransformerLayer(d_model,nhead,feedforward_ratio,drop_rate,module='decoder')

        self.encoder=TransformerModel(self.encoder_layer,encoder_num_layers,module='encoder')
        self.decoder=TransformerModel(self.decoder_layer,decoder_num_layers,module='decoder')

        depth = 6
        #-------------------- ssa_tcn----------------------------
        self.Spatial_Tcn_blocks = nn.ModuleList([
            STLayer(d_model, nhead, drop_rate)  for i in range(depth)])

        self.enc_weights_embed = nn.Linear(d_model, 1)
        self.futr_weights_embed = nn.Linear(d_model, 1)

        self.tgt_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True))

    def forward(self, batch,current_frame,device,
                decoder_input=None,is_train=True):

        features, masks, neighbors=batch
        encoder_input_ = features[:, 1:current_frame, :, 2:].to(device)
        
        b,t1,v,c = encoder_input_.shape     
          
        if is_train:
            de_padding_mask = masks[:, current_frame-1:-1].to(device)
            b,t2,v,c = de_padding_mask.shape
            de_att_mask = subsequent_mask(t2).repeat(b*v, 1, 1).to(device)
            decoder_input_= features[:, current_frame:-1, :, -2:].to(device)
            start_of_seq = torch.zeros((b, 1, v, 2)).to(device)                                              
            decoder_input_ = torch.cat((start_of_seq, decoder_input_), 1).to(device)
        else:
            decoder_input_ = decoder_input.to(device)
            de_padding_mask = masks[:, 5:current_frame].to(device)
            b,t2,v,c = de_padding_mask.shape
            de_att_mask = subsequent_mask(t2).repeat(b*v, 1, 1).to(device)

        decoder_inputs=self.output_embedding(decoder_input_)
        decoder_pe = self._pos_embed(decoder_inputs)
        decoder_inputs = decoder_inputs + decoder_pe
        decoder_inputs = self.pos_drop2(decoder_inputs)
        decoder_inputs = self.norm2(decoder_inputs)

        # print(encoder_input_.shape,decoder_inputs.shape)
        encoder_inputs=self.input_embedding(encoder_input_)
        # encoder_pe = self._pos_embed(encoder_inputs)
        # encoder_inputs = encoder_inputs + encoder_pe
        # encoder_inputs = self.pos_drop2(encoder_inputs)
        # encoder_inputs = self.norm2(encoder_inputs)
    
        # # encoder 1
        for st_layer in self.Spatial_Tcn_blocks:
            encoder_inputs = st_layer(encoder_inputs)

        # encoder 2
        encoder_output, _, _= self.encoder(encoder_inputs, encoder_inputs)
        
        # decoder
        decoder_output, _, _= self.decoder(
            decoder_inputs, encoder_output, att_mask=de_att_mask)
        output=self.output_linear(decoder_output)

        return output, None
    
    def _pos_embed(self, x):
    
        bsize, seq_len, obj_len = x.shape[:-1]

        idx = torch.arange(
            seq_len,device=x.device).reshape(1, seq_len, 1, 1).repeat(bsize, 1, obj_len, 1)
        pos_embed = self.PostionEmbedding(idx)

        return pos_embed
    
    def _tgt_generate(self, hist_embed, enc_output, query_pos_embed):
        _, hist_len, obj_len, d_model = hist_embed.shape
        futr_len = query_pos_embed.shape[1]

        futr_weights_embed = self.futr_weights_embed(
            query_pos_embed).permute(0, 2, 1, 3).reshape(-1, futr_len, 1)
        enc_weights_embed = self.enc_weights_embed(
            enc_output).permute(0, 2, 1, 3).reshape(-1, hist_len, 1)
        futr_weights = torch.bmm(
            futr_weights_embed, enc_weights_embed.transpose(-1, -2))
        tgt_seq = torch.bmm(
            futr_weights, self.tgt_embed(hist_embed).permute(0, 2, 1, 3).reshape(-1, hist_len, d_model))
        
        tgt_seq = tgt_seq.reshape(-1, obj_len, futr_len, d_model).permute(0, 2, 1, 3)

        return tgt_seq