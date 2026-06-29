import torch
import torch.nn as nn

from layers.ts2vec.encoder import TS2VecEncoderWrapper, TSEncoder


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.learning_environment = getattr(args, 'learning_environment', 'online')
        
        # Fix: dynamically get actual input dimensions
        if self.learning_environment == 'offline':
            input_dims = args.enc_in
        else:
            # In online mode, dynamically determine input_dims at forward time
            # Use default value first, adjust based on actual data at forward time
            input_dims = args.enc_in + 7
            
        encoder = TSEncoder(input_dims=input_dims,
                          output_dims=320,
                          hidden_dims=64,
                          depth=10)
        self.encoder = TS2VecEncoderWrapper(encoder, mask='all_true')
        self.pred_len = args.pred_len
        self.dim = args.c_out * args.pred_len
        self.regressor = nn.Linear(320, self.dim)
        
        # Add flag to track whether input_dims has been adjusted
        self.input_dims_adjusted = False

    def forward(self, x, x_mark=None):
        if self.learning_environment == 'offline':
            rep = self.encoder(x)
        else:
            if x_mark is None:
                x_mark = torch.zeros(*x.shape[:2], 7, device=x.device)
            
            # Dynamically adjust input_dims
            if not self.input_dims_adjusted:
                actual_input_dims = x.shape[-1] + x_mark.shape[-1]
                if actual_input_dims != self.encoder.encoder.input_dims:
                    # Recreate encoder to match actual input dimensions
                    new_encoder = TSEncoder(input_dims=actual_input_dims,
                                          output_dims=320,
                                          hidden_dims=64,
                                          depth=10)
                    # Fix: move newly created encoder to the correct device
                    new_encoder = new_encoder.to(x.device)
                    self.encoder = TS2VecEncoderWrapper(new_encoder, mask='all_true')
                    self.input_dims_adjusted = True
            
            x = torch.cat([x, x_mark], dim=-1)
            rep = self.encoder(x)
            
        y = self.regressor(rep)
        y = y.reshape(len(y), self.pred_len, -1)
        return y

class Model_Ensemble(Model):
    def __init__(self, args):
        super().__init__(args)
        depth = 10
        encoder = TSEncoder(input_dims=args.seq_len,
                            output_dims=320,  # standard ts2vec backbone value
                            hidden_dims=64,  # standard ts2vec backbone value
                            depth=depth)
        self.encoder_time = TS2VecEncoderWrapper(encoder, mask='all_true')
        self.regressor_time = nn.Linear(320, args.pred_len)

    def forward_individual(self, x, x_mark):
        rep = self.encoder_time.encoder.forward(x.transpose(1, 2))
        y1 = self.regressor_time(rep).transpose(1, 2)
        y2 = super().forward(x, x_mark)
        return y1, y2

    def forward(self, x, x_mark, w1=0.5, w2=0.5):
        y1, y2 = self.forward_individual(x, x_mark)
        return y1 * w1 + y2 * w2, y1, y2



