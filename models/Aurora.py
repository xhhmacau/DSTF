"""
Aurora: Towards Universal Generative Multimodal Time Series Forecasting (ICLR 2026)
Adapter wrapper for the Unified TSF Benchmark framework.

Aurora is a pretrained multimodal foundation model that uses:
- Channel-Independence: each variable is processed independently as [B, L] input
- Flow Matching for generative probabilistic forecasting
- Built-in RevIN normalization
- ViT (vision) and BERT (text) encoders for multimodal fusion

This wrapper handles the dimension conversion between the framework's [B, L, C]
format and Aurora's [B*C, L] format.
"""

import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Add aurora_modules to path for internal relative imports
_aurora_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'aurora_modules')
if _aurora_dir not in sys.path:
    sys.path.insert(0, _aurora_dir)

from models.aurora_modules.configuration_aurora import AuroraConfig
from models.aurora_modules.modeling_aurora import AuroraForPrediction


class Model(nn.Module):
    """
    Adapter that wraps AuroraForPrediction for the Unified TSF Benchmark.
    
    Handles:
    1. [B, L, C] -> [B*C, L] dimension conversion (Channel-Independence)
    2. Training: forward pass with flow matching loss
    3. Inference: generate() with probabilistic predictions, return mean
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # Aurora-specific params
        self.inference_token_len = getattr(configs, 'inference_token_len', 48)
        self.num_samples = getattr(configs, 'aurora_num_samples', 100)
        pretrained_path = getattr(configs, 'aurora_pretrained_path', None)
        
        # Build AuroraConfig
        aurora_config = AuroraConfig(
            token_len=self.inference_token_len,
            hidden_size=getattr(configs, 'aurora_hidden_size', 512),
            intermediate_size=getattr(configs, 'aurora_intermediate_size', 1024),
            num_enc_layers=getattr(configs, 'aurora_enc_layers', 6),
            num_dec_layers=getattr(configs, 'aurora_dec_layers', 6),
            num_attention_heads=getattr(configs, 'n_heads', 8),
            dropout_rate=getattr(configs, 'dropout', 0.2),
            num_sampling_steps=getattr(configs, 'aurora_sampling_steps', 50),
            flow_loss_depth=3,
            diffusion_batch_mul=4,
        )
        
        # Load model
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading Aurora from pretrained: {pretrained_path}")
            self.aurora = AuroraForPrediction.from_pretrained(
                pretrained_path, trust_remote_code=True
            )
        else:
            print("Initializing Aurora from scratch.")
            self.aurora = AuroraForPrediction(aurora_config)
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        Args:
            x_enc: [Batch, Seq_len, Channels]
        Returns:
            predictions: [Batch, Pred_len, Channels]
        """
        B, L, C = x_enc.shape
        
        # Channel-Independence: reshape to [B*C, L]
        x_ci = rearrange(x_enc, 'b l c -> (b c) l')
        
        if self.training:
            return self._train_forward(x_ci, B, L, C)
        else:
            return self._inference_forward(x_ci, B, C)
    
    def _train_forward(self, x_ci, B, L, C):
        """
        Training: use teacher forcing with labels.
        Aurora's forward returns MoeCausalLMOutputWithPast with loss.
        
        For framework compatibility, we need to return predictions shaped [B, pred_len, C].
        The loss is computed internally by Aurora (flow + point), but our framework
        computes MSE externally. So we return point predictions for external MSE,
        and store the internal loss for optional use.
        """
        # For training, we do a forward pass without labels to get predictions
        # (Aurora's internal loss requires labels in the same [B*C, pred_len] format)
        # We use the linear head for deterministic point predictions
        predict_token_num = math.ceil(self.pred_len / self.aurora.config.token_len)
        
        # RevIN
        means = x_ci.mean(1, keepdim=True).detach()
        stdev = x_ci.std(dim=1, keepdim=True, unbiased=False).detach() + 1e-5
        x_normed = (x_ci - means) / stdev
        
        # Forward through encoder-decoder
        outputs = self.aurora.model(
            input_ids=x_normed,
            predict_token_num=predict_token_num,
            inference_token_len=self.aurora.config.token_len,
        )
        
        hidden_states = outputs.last_hidden_state
        x_rec, x_dec, from_text, from_vision = hidden_states
        
        # Get point predictions via linear head
        fore = rearrange(self.aurora.linear_head(x_dec), 'b n p -> b (n p)')
        fore = fore[:, :self.pred_len]
        
        # De-normalize
        fore = fore * stdev + means
        
        # Reshape back: [B*C, pred_len] -> [B, pred_len, C]
        predictions = rearrange(fore, '(b c) l -> b l c', b=B, c=C)
        
        return predictions
    
    def _inference_forward(self, x_ci, B, C):
        """
        Inference: use generate() for probabilistic predictions, return mean.
        """
        with torch.no_grad():
            # generate() returns [B*C, num_samples, pred_len]
            raw_output = self.aurora.generate(
                inputs=x_ci,
                max_output_length=self.pred_len,
                num_samples=self.num_samples,
                inference_token_len=self.inference_token_len,
            )
            
            # Take mean across samples: [B*C, pred_len]
            if raw_output.dim() == 3:
                predictions = raw_output.mean(dim=1)
            else:
                predictions = raw_output
            
            # Reshape: [B*C, pred_len] -> [B, pred_len, C]
            predictions = rearrange(predictions, '(b c) l -> b l c', b=B, c=C)
        
        return predictions
