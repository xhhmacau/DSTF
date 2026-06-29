"""
Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting
(AAAI 2023)

Paper: https://arxiv.org/abs/2302.14829

Dish-TS addresses distribution shift in TSF via learnable Coefficient Nets (CONETs):
  - BackCONET: learns the distribution of the input (lookback) space
  - HoriCONET: learns the distribution of the output (horizon) space
By separately modeling input/output distributions, it handles both intra-space
and inter-space distribution shifts.

This implementation wraps Dish-TS normalization around a configurable backbone
(DLinear or Informer) for integration into the Unified TSF Benchmark.
"""

import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F


class DishTSNorm(nn.Module):
    """
    Dish-TS Coefficient Net (CONET) normalization layer.
    
    Learns distribution coefficients (level + scaling) from input sequences
    via a learnable linear mapping, replacing fixed mean/variance normalization.
    
    Uses dual coefficients:
      - phil/xil: for normalizing (BackCONET direction)
      - phih/xih: for denormalizing (HoriCONET direction)
    """

    def __init__(self, num_features, seq_len, init='standard', activate=True):
        """
        Args:
            num_features: number of variates (channels) D
            seq_len: lookback length L
            init: initialization method for reduce_mlayer ('standard', 'avg', 'uniform')
            activate: whether to apply GELU activation to the learned coefficients
        """
        super().__init__()
        self.activate = activate

        # Learnable coefficient mapping: [D, L, 2] -> maps each series' lookback
        # into 2 distribution coefficients (level_low, level_high)
        if init == 'standard':
            self.reduce_mlayer = nn.Parameter(
                torch.rand(num_features, seq_len, 2) / seq_len
            )
        elif init == 'avg':
            self.reduce_mlayer = nn.Parameter(
                torch.ones(num_features, seq_len, 2) / seq_len
            )
        elif init == 'uniform':
            self.reduce_mlayer = nn.Parameter(
                torch.ones(num_features, seq_len, 2) / seq_len
                + torch.rand(num_features, seq_len, 2) / seq_len
            )
        else:
            raise ValueError(f"Unknown DishTS init method: {init}")

        # Affine parameters (per-variate)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def _preget(self, batch_x):
        """
        Compute distribution coefficients from input batch.
        
        Args:
            batch_x: [B, L, D] input time series
        """
        # x_transpose: [D, B, L]
        x_transpose = batch_x.permute(2, 0, 1)
        # theta: bmm([D, B, L] x [D, L, 2]) -> [D, B, 2] -> permute -> [B, 2, D]
        theta = torch.bmm(x_transpose, self.reduce_mlayer).permute(1, 2, 0)
        if self.activate:
            theta = F.gelu(theta)

        # Split into two level coefficients
        # phil: BackCONET level (for normalization), [B, 1, D]
        # phih: HoriCONET level (for denormalization), [B, 1, D]
        self.phil = theta[:, :1, :]
        self.phih = theta[:, 1:, :]

        # Compute scaling coefficients (std-like)
        # xil: BackCONET scaling, xih: HoriCONET scaling
        self.xil = torch.sum(
            torch.pow(batch_x - self.phil, 2), dim=1, keepdim=True
        ) / (batch_x.shape[1] - 1)
        self.xih = torch.sum(
            torch.pow(batch_x - self.phih, 2), dim=1, keepdim=True
        ) / (batch_x.shape[1] - 1)

    def normalize(self, batch_x, dec_inp=None):
        """
        Normalize input (and optionally decoder input) using BackCONET coefficients.
        
        Args:
            batch_x: [B, L, D] encoder input
            dec_inp: [B, label_len+pred_len, D] decoder input (optional, for transformer backbones)
            
        Returns:
            normalized batch_x, normalized dec_inp (or None)
        """
        self._preget(batch_x)
        batch_x_norm = self._forward_process(batch_x)
        dec_inp_norm = None if dec_inp is None else self._forward_process(dec_inp)
        return batch_x_norm, dec_inp_norm

    def denormalize(self, batch_y):
        """
        Denormalize output using HoriCONET coefficients.
        
        Args:
            batch_y: [B, pred_len, D] model predictions
            
        Returns:
            denormalized predictions
        """
        return self._inverse_process(batch_y)

    def _forward_process(self, batch_input):
        """Normalize using BackCONET (phil, xil)."""
        temp = (batch_input - self.phil) / torch.sqrt(self.xil + 1e-8)
        return temp.mul(self.gamma) + self.beta

    def _inverse_process(self, batch_input):
        """Denormalize using HoriCONET (phih, xih)."""
        return ((batch_input - self.beta) / self.gamma) * torch.sqrt(self.xih + 1e-8) + self.phih


class Model(nn.Module):
    """
    Dish-TS model with configurable backbone.
    
    Wraps DishTS normalization around a backbone forecasting model.
    Supports both simple (DLinear) and encoder-decoder (Informer) backbones.
    
    Usage:
        --model DishTS --dish_backbone DLinear    (default)
        --model DishTS --dish_backbone Informer
    """

    # Backbone models that use encoder-decoder architecture
    # and require (x_enc, x_mark_enc, x_dec, x_mark_dec) interface
    TRANSFORMER_BACKBONES = {'Informer', 'Autoformer', 'Transformer'}

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = getattr(configs, 'label_len', 48)

        # Determine backbone
        backbone_name = getattr(configs, 'dish_backbone', 'DLinear')
        self.backbone_name = backbone_name
        self.is_transformer = backbone_name in self.TRANSFORMER_BACKBONES

        # Build DishTS normalization (CONET)
        dish_init = getattr(configs, 'dish_init', 'standard')
        dish_activate = getattr(configs, 'dish_activate', True)
        self.dishts = DishTSNorm(
            num_features=configs.enc_in,
            seq_len=configs.seq_len,
            init=dish_init,
            activate=dish_activate,
        )

        # Build backbone model
        backbone_module = importlib.import_module(f'models.{backbone_name}')
        self.backbone = backbone_module.Model(configs)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        Forward pass with DishTS normalization.
        
        Args:
            x_enc: [B, seq_len, D] encoder input
            x_mark_enc: [B, seq_len, T] encoder time marks (for transformer backbones)
            x_dec: [B, label_len+pred_len, D] decoder input (for transformer backbones)
            x_mark_dec: [B, label_len+pred_len, T] decoder time marks (for transformer backbones)
            
        Returns:
            output: [B, pred_len, D] forecasting result
        """
        # Step 1: Normalize encoder input (and decoder input for transformers)
        if self.is_transformer and x_dec is not None:
            x_enc_norm, x_dec_norm = self.dishts.normalize(x_enc, dec_inp=x_dec)
            # Step 2: Forward through backbone
            output = self.backbone(x_enc_norm, x_mark_enc, x_dec_norm, x_mark_dec)
        else:
            x_enc_norm, _ = self.dishts.normalize(x_enc)
            # Step 2: Forward through backbone
            output = self.backbone(x_enc_norm)

        # Handle tuple output (some backbones return (output, attention))
        if isinstance(output, tuple):
            output = output[0]

        # Step 3: Denormalize output
        output = self.dishts.denormalize(output)

        return output
