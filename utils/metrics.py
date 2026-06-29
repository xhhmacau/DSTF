import math

import numpy as np
import pandas as pd


def update_metrics(pred, label, statistics, target_variate=None):
    if isinstance(pred, tuple):
        pred = pred[0]
    if target_variate is not None:
        pred = pred[:, :, target_variate]
        if label.dim() == 3:
            label = label[:, :, target_variate]

    balance = pred - label
    # statistics['all_preds'].append(pred)
    statistics['y_sum'] += label.abs().sum().item()
    statistics['total'] += len(label.view(-1))
    statistics['MAE'] += balance.abs().sum().item()
    statistics['MSE'] += (balance ** 2).sum().item()
    # RRSE += (balance ** 2).sum()
    # x2_sum += (target_batch ** 2).sum()
    # x_sum += target_batch.sum()

    # Add new metrics (removed MAPE and SMAPE)
    if 'RMSE' not in statistics:
        statistics['RMSE'] = 0
    if 'WAPE' not in statistics:
        statistics['WAPE'] = 0
    if 'MSMAPE' not in statistics:
        statistics['MSMAPE'] = 0
    if 'MASE' not in statistics:
        statistics['MASE'] = 0
        
    # Update new metric values
    pred_np = pred.detach().cpu().numpy()
    label_np = label.detach().cpu().numpy()
    
    statistics['RMSE'] += RMSE(pred_np, label_np) * len(label.view(-1))
    statistics['WAPE'] += WAPE(pred_np, label_np) * len(label.view(-1))
    statistics['MSMAPE'] += MSMAPE(pred_np, label_np) * len(label.view(-1))



def calculate_metrics(statistics):
    # MSE, MAE, total, y_sum = statistics['MSE'], statistics['MAE'], statistics['total'], statistics['y_sum']
    # metrics = {'MSE': MSE / total, 'MAE': MAE / total}
    # metrics['NMAE'] = MAE / y_sum
    # metrics['NRMSE'] = math.sqrt((MSE / total)) / (y_sum / total)
    # var = x2_sum / total - (x_sum / total) ** 2
    # RRSE = math.sqrt(RRSE.item() / total) / var.item()
    # return metrics

    total = statistics['total']
    metrics = {
        'MSE': statistics['MSE'] / total,
        'MAE': statistics['MAE'] / total,
        'RMSE': statistics['RMSE'] / total if 'RMSE' in statistics else 0,
        'WAPE': statistics['WAPE'] / total if 'WAPE' in statistics else 0,
        'MSMAPE': statistics['MSMAPE'] / total if 'MSMAPE' in statistics else 0,
        # 'MASE': statistics['MASE'] / total if 'MASE' in statistics else -1
    }
    return metrics

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def WAPE(pred, true):
    return np.sum(np.abs(true - pred)) / np.sum(np.abs(true)) * 100

def MSMAPE(pred, true, epsilon=0.1):
    comparator = np.full_like(true, 0.5 + epsilon)
    denom = np.maximum(comparator, np.abs(pred) + np.abs(true) + epsilon)
    return np.mean(2 * np.abs(pred - true) / denom) * 100

# def MASE(pred, true, hist_data, seasonality=2):
#     if seasonality == 2:
#         return -1
#     scale = len(pred) / (len(hist_data) - seasonality)
    
#     dif = 0
#     for i in range((seasonality + 1), len(hist_data)):
#         dif = dif + abs(hist_data[i] - hist_data[i - seasonality])
    
#     scale = scale * dif
#     return (sum(abs(true - pred)) / scale)[0]

def MASE(pred, true, historical_data, seasonality=1):
    """
    Compute MASE over the entire dataset
    pred: all model predictions (array-like)
    true: all corresponding ground truth values (array-like)
    historical_data: complete history for computing naive forecast baseline (array-like)
    seasonality: seasonality period of the data (e.g., 1 for no seasonality, 12 for monthly data)
    """
    # Compute naive forecast MAE
    # For seasonal data, naive forecast is the value at t-seasonality
    naive_forecast_errors = np.abs(historical_data[seasonality:] - historical_data[:-seasonality])
    mae_naive = np.mean(naive_forecast_errors)

    # Compute model MAE
    mae_model = np.mean(np.abs(true - pred))

    # Avoid division by zero
    if mae_naive == 0:
        return float('inf') if mae_model > 0 else 0 # or a very large number, or 0

    return mae_model / mae_naive

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true, hist_data=None, seasonality=1):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    wape = WAPE(pred, true)
    msmape = MSMAPE(pred, true)
    mase = MASE(pred, true, hist_data, seasonality) if hist_data is not None else -1
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, wape, msmape, mase, rse, corr


def calc_ic(pred=None, label=None, index=None, df=None, return_type='all', reduction='sum'):
    if df is None:
        if isinstance(pred, tuple):
            pred = pred[0]
        df = pd.DataFrame({'pred': pred, 'label': label}, index=index)
    if index is None:
        res = []
        if return_type != 'ric':
            res.append(df['pred'].corr(df['label']))
        if return_type != 'ic':
            res.append(df['pred'].corr(df['label'], method='spearman'))
        return res
    else:
        groups = df.groupby('datetime')
        res = []
        if return_type != 'ric':
            res.append(groups.apply(lambda df: df["pred"].corr(df["label"], method="pearson")))
        if return_type != 'ic':
            res.append(groups.apply(lambda df: df["pred"].corr(df["label"], method="spearman")))
        if reduction == 'sum':
            return [r.sum() for r in res] + [len(groups)]
        elif reduction == 'mean':
            return [r.mean() for r in res]
        else:
            return [r.to_numpy().tolist() for r in res]
