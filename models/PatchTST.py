__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp

# Try to import offline version components
try:
    from layers.Transformer_EncDec import Encoder, EncoderLayer
    from layers.SelfAttention_Family import FullAttention, AttentionLayer
    from layers.Embed import PatchEmbedding
    OFFLINE_COMPONENTS_AVAILABLE = True
except ImportError:
    OFFLINE_COMPONENTS_AVAILABLE = False


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # Detect learning environment
        learning_env = getattr(configs, 'learning_environment', 'online')
        use_offline_version = (learning_env == 'offline' and OFFLINE_COMPONENTS_AVAILABLE and 
                              hasattr(configs, 'task_name'))
        
        if use_offline_version:
            self._init_offline_version(configs)
        else:
            self._init_online_version(configs, max_seq_len, d_k, d_v, norm, attn_dropout, 
                                    act, key_padding_mask, padding_var, attn_mask, res_attention, 
                                    pre_norm, store_attn, pe, learn_pe, pretrain_head, head_type, verbose, **kwargs)
        
        self.use_offline_version = use_offline_version

    def _init_offline_version(self, configs):
        """Initialize offline version of PatchTST"""
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Set default values
        patch_len = getattr(configs, 'patch_len', 16)
        stride = getattr(configs, 'stride', 8)
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        head_dropout = getattr(configs, 'head_dropout', 0.0)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len, head_dropout=head_dropout)

    def _init_online_version(self, configs, max_seq_len, d_k, d_v, norm, attn_dropout, 
                           act, key_padding_mask, padding_var, attn_mask, res_attention, 
                           pre_norm, store_attn, pe, learn_pe, pretrain_head, head_type, verbose, **kwargs):
        """Initialize online version of PatchTST"""
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = getattr(configs, 'fc_dropout', 0.05)
        head_dropout = getattr(configs, 'head_dropout', 0.0)
        
        individual = getattr(configs, 'individual', False)
    
        patch_len = getattr(configs, 'patch_len', 16)
        stride = getattr(configs, 'stride', 8)
        padding_patch = getattr(configs, 'padding_patch', 'end')
        
        revin = getattr(configs, 'revin', 1)
        affine = getattr(configs, 'affine', 0)
        subtract_last = getattr(configs, 'subtract_last', 0)
        
        decomposition = getattr(configs, 'decomposition', 0)
        kernel_size = getattr(configs, 'kernel_size', 25)

        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Offline version forecast method"""
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, return_emb=False):
        """Unified forward interface"""
        if self.use_offline_version:
            # Offline version forward
            if x_mark_enc is None and x_dec is None and x_mark_dec is None:
                # If only x_enc provided, create default params
                batch_size, seq_len, n_vars = x_enc.shape
                x_mark_enc = torch.zeros(batch_size, seq_len, 1).to(x_enc.device)
                x_dec = torch.zeros(batch_size, self.pred_len, n_vars).to(x_enc.device)
                x_mark_dec = torch.zeros(batch_size, self.pred_len, 1).to(x_enc.device)
            
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            # Online version forward
            x = x_enc  # Use first argument as input
            if self.decomposition:
                res_init, trend_init = self.decomp_module(x)
                res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
                res = self.model_res(res_init)
                trend = self.model_trend(trend_init)
                x = res + trend
                x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
            else:
                x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
                x = self.model(x, return_emb=return_emb)
                if return_emb:
                    x, z = x
                x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
                if return_emb:
                    return x, z
            return x