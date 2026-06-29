import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN


from models.Statistics_prediction import Statistics_prediction

class ForecastModel(nn.Module):
    def __init__(self, backbone, num_features, seq_len, process_method='RevIN', configs=None, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.process_method = process_method.lower()
        if self.process_method == 'revin':
            self.processor = RevIN(num_features=num_features, **kwargs)
        elif self.process_method == 'dishts':
            self.processor = DishTS(num_features=num_features, seq_len=seq_len, **kwargs)
        elif self.process_method == 'san':
            self.processor = Statistics_prediction(configs)
        else:
            raise NotImplementedError

    def forward(self, x, *args, process=True, **kwargs):
        station_pred = None
        if process:
            if self.process_method == 'san':
                x, station_pred = self.processor.normalize(x)
            else:
                x = self.processor(x, mode='norm')
                
        outputs = self.backbone(x, *args, **kwargs)
        
        if not process:
            return outputs
            
        pred_out = outputs[0] if isinstance(outputs, tuple) else outputs
        
        if self.process_method == 'san':
            pred = self.processor.de_normalize(pred_out, station_pred)
        else:
            pred = self.processor(pred_out, mode='denorm')
            
        if isinstance(outputs, tuple):
            return [pred] + [o for o in outputs[1:]]
        else:
            return pred



class DishTS(RevIN):
    def __init__(self, num_features: int, eps=1e-8, affine=True, seq_len=None, init='standard', **kwargs):
        super().__init__(num_features, eps, affine, **kwargs)
        if init == 'standard':
            self.reduce_mlayer = nn.Parameter(torch.rand(num_features, seq_len, 2) / seq_len)
        elif init == 'avg':
            self.reduce_mlayer = nn.Parameter(torch.ones(num_features, seq_len, 2) / seq_len)
        elif init == 'uniform':
            self.reduce_mlayer = nn.Parameter(
                torch.ones(num_features, seq_len, 2) / seq_len + torch.rand(num_features, seq_len, 2) / seq_len)

    def _get_statistics(self, x):
        x_transpose = x.permute(2, 0, 1)
        theta = torch.bmm(x_transpose, self.reduce_mlayer).permute(1, 2, 0)
        theta = F.gelu(theta)  # Paper uses GELU activation in Coefficient Net
        self.phil, self.phih = theta[:, :1, :], theta[:, 1:, :]
        self.xil = torch.sqrt(torch.sum(torch.pow(x - self.phil, 2), axis=1, keepdim=True) / (x.shape[1] - 1) + self.eps)
        self.xih = torch.sqrt(torch.sum(torch.pow(x - self.phih, 2), axis=1, keepdim=True) / (x.shape[1] - 1) + self.eps)

    def _normalize(self, x):
        x = (x - self.phil) / self.xil
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        x = x * self.xih
        x = x + self.phih
        return x