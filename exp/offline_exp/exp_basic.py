import os
import torch
import numpy as np

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        # Complete model dictionary with all supported models
        self.model_dict = {
            # Linear models
            'NLinear': 'NLinear',
            'DLinear': 'DLinear',
            'Linear': 'Linear',
            'RLinear': 'RLinear',
            
            # Transformer-based models
            'PatchTST': 'PatchTST',
            'iTransformer': 'iTransformer',
            'Transformer': 'Transformer',
            'Informer': 'Informer',
            'Autoformer': 'Autoformer',
            'Crossformer': 'Crossformer',
            
            # Time series specific models
            'TimesNet': 'TimesNet',
            'TimeMixer': 'TimeMixer',
            'TimeBridge': 'TimeBridge',
            
            # CNN-based models
            'TCN': 'TCN',
            'ModernTCN': 'ModernTCN',
            'SCINet': 'SCINet',
            
            # Specialized models
            'MICN': 'MICN',
            'Leddam': 'Leddam',
            'DeformableTST': 'DeformableTST',
            'DUET': 'duet',  # Note: filename is lowercase
            'duet': 'duet',
            'S_Mamba': 'S_Mamba',
            'BiMamba4TS': 'BiMamba4TS', # Add BiMamba4TS
            'OLinear': 'OLinear',       # Add OLinear
            
            # Graph-based models
            'MTGNN': 'MTGNN',
            
            # Other models
            'GPT4TS': 'GPT4TS',
            'OneNet': 'OneNet',
            'FSNet': 'FSNet',
            'LIFT': 'LIFT',
            'LightMTS': 'LightMTS',
            'LightGTS': 'LightGTS',
            'Koopa': 'Koopa',
            'Stat_models': 'Stat_models',
            'DishTS': 'DishTS',
            
            # LLM / Foundation models
            'TimeLLM': 'TimeLLM',
            'Aurora': 'Aurora',
            
            # MoE models
            'SREMC_MoE': 'SREMC_MoE',
        }
        
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            # Check gpu_type param, default to cuda if not present
            gpu_type = getattr(self.args, 'gpu_type', 'cuda')
            
            if gpu_type == 'cuda' and torch.cuda.is_available():
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use GPU: cuda:{}'.format(self.args.gpu))
            elif gpu_type == 'mps' and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device('mps')
                print('Use GPU: mps')
            else:
                device = torch.device('cpu')
                print('Use CPU (GPU not available or not requested)')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
