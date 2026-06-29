import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import csv
# Use offline-specific data_provider
from data_provider.offline.data_factory import data_provider
from exp.offline_exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from settings import need_x_y_mark, need_x_mark
from models.normalization import ForecastModel

warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        self.train_historical_data = None
        self.efficiency_metrics = None
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        # Pass learning environment info during model initialization
        self.args.learning_environment = 'offline'
        
        # For Koopa model, compute the mask_spectrum dynamically before loading the model
        if self.args.model == 'Koopa':
            print("Calculating mask_spectrum for Koopa model...")
            train_data, train_loader = self._get_data(flag='train')
            amps = 0.0
            for i, data in enumerate(train_loader):
                lookback_window = data[0]
                amps += abs(torch.fft.rfft(lookback_window, dim=1)).mean(dim=0).mean(dim=1)
            alpha = getattr(self.args, 'alpha', 0.2)
            self.args.mask_spectrum = amps.topk(int(amps.shape[0] * alpha)).indices
            print(f"Calculated mask_spectrum: {self.args.mask_spectrum}")

        # Fix model import logic
        if self.args.model in self.model_dict:
            model_name = self.model_dict[self.args.model]
            try:
                # Correct dynamic import approach
                model_module = __import__(f'models.{model_name}', fromlist=['Model'])
                model = model_module.Model(self.args).float()
                
                # Wrap model with ForecastModel if a normalization method is specified
                normalization = getattr(self.args, 'normalization', '')
                if normalization:
                    print(f"Wrapping model with {normalization} normalization layer.")
                    model = ForecastModel(model, num_features=self.args.enc_in, seq_len=self.args.seq_len, process_method=normalization, configs=self.args).float()
                    
                print(f"Successfully loaded offline model: {self.args.model}")
            except ImportError as e:
                raise ValueError(f"Model {self.args.model} not available. Error: {e}")
        else:
            raise ValueError(f"Unknown model: {self.args.model}. Available models: {list(self.model_dict.keys())}")
            
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        # Use offline data_provider
        data_set, data_loader = data_provider(self.args, flag)
        
        # If training set, save history data for MASE computation
        if flag == 'train' and self.train_historical_data is None:
            # Get complete training set history data
            if hasattr(data_set, 'data_y'):
                # For ETT datasets, use data_y as history data
                self.train_historical_data = data_set.data_y.copy()
            elif hasattr(data_set, 'data_x'):
                # Fallback: use data_x as history data
                self.train_historical_data = data_set.data_x.copy()
        
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Select appropriate input parameters based on model type
                if self.args.model in need_x_y_mark:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                elif self.args.model in need_x_mark:
                    outputs = self.model(batch_x, batch_x_mark)
                else:
                    outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        station_optim = optim.Adam(self.model.processor.parameters(), lr=self.args.station_lr) if ((getattr(self.args, 'normalization', '') or '').lower() == 'san') else None
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                if station_optim is not None:
                    station_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Select appropriate input parameters based on model type
                if self.args.model in need_x_y_mark:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                elif self.args.model in need_x_mark:
                    outputs = self.model(batch_x, batch_x_mark)
                else:
                    outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()
                if station_optim is not None:
                    station_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, None, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth'), map_location=self.device)
            )

        self.model_optim = self._select_optimizer()

        # Record efficiency metrics before testing
        if self.efficiency_metrics is None:
            # Record parameter count
            total_params = sum(p.numel() for p in self.model.parameters())

            sample_batch = next(iter(test_loader))
            sample_x, sample_y, sample_x_mark, sample_y_mark = [
                item.float().to(self.device) for item in sample_batch
            ]

            def run_model(batch_x, batch_y, batch_x_mark, batch_y_mark):
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                self.model_optim.zero_grad()
                if self.args.model in need_x_y_mark:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                elif self.args.model in need_x_mark:
                    outputs = self.model(batch_x, batch_x_mark)
                else:
                    outputs = self.model(batch_x)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                return outputs

            # Test training speed on a real batch so time-feature dimensions match the dataset.
            self.model.train()
            train_start = time.time()
            criterion = self._select_criterion()
            for _ in range(5):
                outputs = run_model(sample_x, sample_y, sample_x_mark, sample_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = sample_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = self._select_criterion()(outputs, batch_y)
                loss.backward()
                self.model_optim.step()
            train_time = (time.time() - train_start) / 5
            
            # Test inference speed
            infer_start = time.time()
            with torch.no_grad():
                for _ in range(100):
                    run_model(sample_x[:1], sample_y[:1], sample_x_mark[:1], sample_y_mark[:1])
            infer_time = (time.time() - infer_start) / 100
            
            self.efficiency_metrics = {
                'parameter_count': total_params,
                'training_speed_ms': train_time * 1000,  # Convert to milliseconds
                'inference_speed_ms': infer_time * 1000
            }

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Select appropriate input parameters based on model type
                if self.args.model in need_x_y_mark:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                elif self.args.model in need_x_mark:
                    outputs = self.model(batch_x, batch_x_mark)
                else:
                    outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        # Fix: convert to numpy array first, then check shape
        preds = np.array(preds, dtype=object)  # Use object dtype to avoid shape inconsistency issues
        trues = np.array(trues, dtype=object)
        
        print('test shape before reshape:', len(preds), preds[0].shape if len(preds) > 0 else 'empty')
        
        # Fix: correct reshape approach
        if len(preds) > 0:
            # Concatenate all batches
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            print('test shape after concatenate:', preds.shape, trues.shape)
        else:
            print('No predictions generated!')
            return None, None

        # Get training data for MASE computation
        try:
            train_data, _ = self._get_data('train')
            # Get raw training data values (for MASE computation)
            if hasattr(train_data, 'data_x'):
                # For multivariate, select based on features param
                if self.args.features == 'MS':
                    # Multivariate-to-univariate: use target variable
                    if hasattr(train_data, 'data_y'):
                        historical_data = train_data.data_y.flatten()
                    else:
                        historical_data = train_data.data_x[:, :, -1].flatten()  # Use the last variable
                else:
                    # Univariate or multivariate-to-multivariate
                    historical_data = train_data.data_x.flatten()
            else:
                historical_data = None
        except Exception as e:
            print(f"Warning: Could not get training data for MASE: {e}")
            historical_data = None

        # Set seasonality parameter (based on data frequency)
        seasonality = 1  # Default value
        if hasattr(self.args, 'freq'):
            if self.args.freq == 'h':  # Hourly data
                seasonality = 24  # Daily seasonality
            elif self.args.freq == 't':  # Minute data
                seasonality = 60  # Hourly seasonality
            elif self.args.freq == 'd':  # Daily data
                seasonality = 7   # Weekly seasonality

        # Compute overall performance metrics
        if historical_data is not None:
            mae, mse, rmse, wape, msmape, mase, rse, corr = metric(preds, trues, historical_data, seasonality)
        else:
            mae, mse, rmse, wape, msmape, mase, rse, corr = metric(preds, trues)

        # Print results (kept for debugging)
        print('mse:{}, mae:{}, rmse:{}, wape:{}, msmape:{}, mase:{}'.format(
            mse, mae, rmse, wape, msmape, mase))
        print('Efficiency metrics:')
        print(f'  Parameter Count: {self.efficiency_metrics["parameter_count"]:,}')
        print(f'  Training Speed: {self.efficiency_metrics["training_speed_ms"]:.2f} ms/batch')
        print(f'  Inference Speed: {self.efficiency_metrics["inference_speed_ms"]:.2f} ms/sample')

        # Return all metrics
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'wape': wape,
            'msmape': msmape,
            'mase': mase
        }
