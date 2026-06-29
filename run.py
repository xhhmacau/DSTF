import argparse
import datetime
import gc
import os
import csv
import time
import platform
import random
import numpy as np
import pandas as pd
import torch
from pprint import pprint

import settings
from data_provider.online import data_loader
from data_provider.online.data_loader import Dataset_Recent
from exp.online_exp.exp_online import Exp_Online
from exp.online_exp.exp_solid import Exp_SOLID
from exp.offline_exp.exp_long_term_forecasting import Exp_Long_Term_Forecast as Offline_Exp
from exp.online_exp.exp_main import Exp_Main

cur_sec = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(cur_sec)

# Import online experiments
import exp.online_exp as online_exps
from exp.online_exp import *

# Import offline experiments
from exp.offline_exp.exp_long_term_forecasting import Exp_Long_Term_Forecast as Offline_Exp_Long_Term_Forecast

from settings import data_settings


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser(description='Unified TSF Benchmark - Online and Offline Learning', conflict_handler='resolve')

# Basic config
parser.add_argument('--learning_environment', type=str, default='online', 
                    choices=['online', 'offline'], 
                    help='Learning environment: online for continual learning, offline for traditional training')
parser.add_argument('--train_only', action='store_true', default=False,
                    help='perform training on full input dataset without validation and testing')
parser.add_argument('--wo_test', action='store_true', default=False, help='only valid, not test')
parser.add_argument('--wo_valid', action='store_true', default=False, help='only test')
parser.add_argument('--only_test', action='store_true', default=False)
parser.add_argument('--do_valid', action='store_true', default=False)
parser.add_argument('--model', type=str, required=True, default='PatchTST')
parser.add_argument('--override_hyper', type=str_to_bool, default=True,
                    help='Override hyperparameters with settings.py defaults')
parser.add_argument('--compile', action='store_true', default=False, help='Compile the model by Pytorch 2.0')
parser.add_argument('--reduce_bs', type=str_to_bool, default=False,
                    help='Override batch_size in hyperparams by setting.py')
parser.add_argument('--normalization', type=str, default=None)
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--tag', type=str, default='')

# SAN args
parser.add_argument('--period_len', type=int, default=24)
parser.add_argument('--station_lr', type=float, default=0.0001)
parser.add_argument('--station_type', type=str, default='adaptive')

# Time-LLM args
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM backbone: LLAMA, GPT2, BERT')
parser.add_argument('--llm_dim', type=int, default=4096, help='LLM hidden dim. LLAMA:4096, GPT2:768, BERT:768')
parser.add_argument('--llm_layers', type=int, default=32, help='Number of LLM layers to use')
parser.add_argument('--patch_len', type=int, default=16, help='Patch length for patching-based models')
parser.add_argument('--stride', type=int, default=8, help='Stride for patching')
parser.add_argument('--prompt_domain', type=int, default=0, help='Use domain-specific prompts')

# Aurora args
parser.add_argument('--aurora_pretrained_path', type=str, default=None, help='Path to pretrained Aurora checkpoint')
parser.add_argument('--inference_token_len', type=int, default=48, help='Aurora inference token length')
parser.add_argument('--aurora_num_samples', type=int, default=100, help='Number of probabilistic samples for Aurora')
parser.add_argument('--aurora_hidden_size', type=int, default=512)
parser.add_argument('--aurora_intermediate_size', type=int, default=1024)
parser.add_argument('--aurora_enc_layers', type=int, default=6)
parser.add_argument('--aurora_dec_layers', type=int, default=6)
parser.add_argument('--aurora_sampling_steps', type=int, default=50)

# Online
parser.add_argument('--online_method', type=str, default=None,
                    help='Online learning method: Online, Proceed, SOLID, OneNet, FSNet, DERpp, ER')
parser.add_argument('--skip', type=str, default=None)
parser.add_argument('--online_learning_rate', type=float, default=None)
parser.add_argument('--val_online_lr', action='store_true', default=True)
parser.add_argument('--diff_online_lr', action='store_true', default=False)
parser.add_argument('--save_opt', action='store_true', default=True)
parser.add_argument('--leakage', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--freeze', action='store_true', default=False)
parser.add_argument('--force_retrain', action='store_true', default=False)

# Proceed
parser.add_argument('--act', type=str, default='sigmoid', help='activation')
parser.add_argument('--tune_mode', type=str, default='down_up')
parser.add_argument('--ema', type=float, default=0, help='')
parser.add_argument('--concept_dim', type=int, default=200)
parser.add_argument('--bottleneck_dim', type=int, default=32, help='')
parser.add_argument('--individual_generator', action='store_true', default=False)
parser.add_argument('--share_encoder', action='store_true', default=False)
parser.add_argument('--use_mean', type=str_to_bool, default=True)
parser.add_argument('--joint_update_valid', action='store_true', default=False)
parser.add_argument('--comment', type=str, default='')
parser.add_argument('--wo_clip', action='store_true', default=False)

# OneNet
parser.add_argument('--learning_rate_w', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--learning_rate_bias', type=float, default=0.001, help='optimizer learning rate')

# Offline (from SRG-MoE)
parser.add_argument('--task_name', type=str, default='long_term_forecast',
                    help='task name, options:[long_term_forecast]')
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, default='', help='model id')

# MoE specific arguments
parser.add_argument('--expert_type', type=str, default='PatchTST', 
                   help='type of expert model, options: [PatchTST, DLinear]')
parser.add_argument('--alpha', type=float, default=0.3, help='alpha for EMA')
parser.add_argument('--entropy_weight', type=float, default=0.1, help='weight for entropy loss')
parser.add_argument('--mc_dropout', type=float, default=0.1, help='Monte Carlo dropout rate for gate network')

# Data loader
parser.add_argument('--border_type', type=str, default='online', help='set any other value for traditional data splits')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--dataset', type=str, default='ETTh1', help='data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--wrap_data_class', type=list, default=[])
parser.add_argument('--pin_gpu', type=str_to_bool, default=True)

# Forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# Model arguments
parser.add_argument('--individual', action='store_true', default=False,
                    help='DLinear: a linear layer for each variate(channel) individually')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

# PatchTST specific arguments
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--drop_last', action='store_true', default=False)
parser.add_argument('--pe_type', type=str, default='no', help='position encoding type for Leddam')
parser.add_argument('--n_layers', type=int, default=3, help='number of layers for models like Leddam')

# SRG-MoE offline specific arguments
parser.add_argument('--num_experts', type=int, default=8, help='number of experts in SREMC_MoE')
parser.add_argument('--activated_experts', type=int, default=2, help='number of activated experts')
parser.add_argument('--num_samples', type=int, default=5, help='number of MC dropout samples')
parser.add_argument('--mlp_hidden1', type=int, default=128, help='first MLP hidden dimension')
parser.add_argument('--seed', type=int, default=2023, help="Randomization seed")

# Additional model-specific parameters
parser.add_argument('--top_k', type=int, default=5, help='for TimesNet')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--seg_len', type=int, default=96, help='segment length for SegRNN')
parser.add_argument('--channel_independence', type=int, default=1, help='0: channel dependence 1: channel independence')
parser.add_argument('--decomp_method', type=str, default='moving_avg', help='method of series decomposition')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--down_sampling_method', type=str, default='max', help='TimeMixer down sampling method: max, avg, conv')
parser.add_argument('--down_sampling_window', type=int, default=2, help='window size for down sampling')
parser.add_argument('--down_sampling_layers', type=int, default=2, help='number of down sampling layers')

# Optimization
parser.add_argument('--begin_valid_epoch', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=7, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=str_to_bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type: cuda or mps')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

# Additional offline arguments (from SRG-MoE)
parser.add_argument('--use_dtw', type=bool, default=False, help='use dtw metric')
parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

# Additional arguments for compatibility
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--do_predict', action='store_true', default=False)
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--whole_model', action='store_true', default=False)

# Additional online-specific arguments (from OnlineTSF)
parser.add_argument('--test_train_num', type=int, default=500)
parser.add_argument('--selected_data_num', type=int, default=5)
parser.add_argument('--lambda_period', type=float, default=0.1)
parser.add_argument('--continual', action='store_true')

# BiMamba4TS specific arguments
parser.add_argument('--SRA', action='store_true', default=False, help='Automatic channel independence judgment')
parser.add_argument('--threshold', type=float, default=0.6, help='Threshold for SRA')
parser.add_argument('--d_conv', type=int, default=4, help='Conv kernel size for Mamba')
parser.add_argument('--e_fact', type=int, default=2, help='Expansion factor for Mamba')
parser.add_argument('--bi_dir', type=int, default=1, help='Bidirectional Mamba')
parser.add_argument('--residual', type=int, default=1, help='Residual connection')
parser.add_argument('--pos_learnable', type=str_to_bool, default=False, help='Learnable positional embedding')

# OLinear specific arguments
parser.add_argument('--q_mat_file', type=str, default='', help='Path to Q matrix file for OLinear')
parser.add_argument('--q_out_mat_file', type=str, default='', help='Path to Q output matrix file for OLinear')
parser.add_argument('--Q_MAT_file', type=str, default='', help='Path to Q matrix file for OLinear (Channel Independent)')
parser.add_argument('--Q_OUT_MAT_file', type=str, default='', help='Path to Q output matrix file for OLinear (Channel Independent)')
parser.add_argument('--embed_size', type=int, default=16, help='Embedding size for OLinear')
parser.add_argument('--CKA_flag', type=int, default=0, help='CKA flag for OLinear')
parser.add_argument('--temp_patch_len', type=int, default=16, help='Temporal patch length for OLinear')
parser.add_argument('--temp_stride', type=int, default=8, help='Temporal stride for OLinear')
parser.add_argument('--Q_chan_indep', type=int, default=0, help='Channel independence for OLinear Q matrix')

# Additional missing parameters
parser.add_argument('--use_time', type=str_to_bool, default=False, help='use time features')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='seasonal patterns')

# MICN-specific parameters
parser.add_argument('--conv_kernel', type=int, nargs='+', default=[17, 49], 
                    help='downsampling and upsampling convolution kernel_size for MICN')
parser.add_argument('--decomp_kernel', type=int, nargs='+', default=[17, 49], 
                    help='decomposition kernel_size for MICN')
parser.add_argument('--isometric_kernel', type=int, nargs='+', default=[17, 49], 
                    help='isometric convolution kernel_size for MICN')

# DUET specific arguments
parser.add_argument('--k', type=int, default=2, help='top-k experts to use in DUET')
parser.add_argument('--noisy_gating', type=str_to_bool, default=True, help='use noisy gating in DUET')
parser.add_argument('--hidden_size', type=int, default=128, help='hidden size for DUET router')
parser.add_argument('--CI', type=str_to_bool, default=True, help='Channel Independent mode in DUET')

# Transformer-based models arguments (Informer, Autoformer, Transformer)
parser.add_argument('--embed_type', type=int, default=0,
                    help='embedding type: 0=default, 1=value+temporal+positional, 2=value+temporal, 3=value+positional, 4=value only')

# Autoformer specific arguments
parser.add_argument('--output_enc', action='store_true', default=False, help='whether to output embedding from encoder')

# Crossformer specific arguments  
parser.add_argument('--win_size', type=int, default=2, help='window size for Crossformer')
parser.add_argument('--num_routers', type=int, default=4, help='number of routers for Crossformer')

# DeformableTST specific arguments
parser.add_argument('--stem_ratio', type=int, default=8, help='stem ratio for DeformableTST')
parser.add_argument('--down_ratio', type=int, default=2, help='down ratio for DeformableTST')
parser.add_argument('--fmap_size', type=int, default=96, help='feature series length')
parser.add_argument('--dims', nargs='+', type=int, default=[64, 128, 256, 512], help='dimensions for each stage in DeformableTST')
parser.add_argument('--depths', nargs='+', type=int, default=[1, 1, 3, 1], help='number of Transformer blocks for each stage')
parser.add_argument('--drop_path_rate', type=float, default=0.3, help='base drop path rate for DeformableTST')
parser.add_argument('--layer_scale_value', nargs='+', type=float, default=[1e-6, 1e-6, 1e-6, 1e-6], help='layer scale values for each stage in DeformableTST')
parser.add_argument('--use_pe', nargs='+', type=int, default=[1, 1, 1, 1], help='use positional encoding for each stage (1=True, 0=False)')
parser.add_argument('--use_lpu', nargs='+', type=int, default=[1,1,1,1], help='use Local Perception Unit; True 1 False 0')
parser.add_argument('--local_kernel_size', nargs='+', type=int, default=[3, 3, 3, 3], help='kernel size for LPU')
parser.add_argument('--use_dwc_mlp', nargs='+', type=int, default=[1,1,1,1], help='use FFN with a DWConv; True 1 False 0')
parser.add_argument('--heads', nargs='+', type=int, default=[2, 4, 8, 16], help='attention heads for DeformableTST')
parser.add_argument('--n_groups', nargs='+', type=int, default=[2, 4, 8, 16], help='number of offset groups')
parser.add_argument('--ksize', nargs='+', type=int, default=[9, 7, 5, 3], help='kernel size for offset sub-network')
parser.add_argument('--window_size', nargs='+', type=int, default=[3, 3, 3, 3], help='kernel size for window attention')
parser.add_argument('--stage_spec', nargs='+', type=list, default=[['D'], ['D'], ['D','D','D'], ['D']], help='type of blocks in each stage')
parser.add_argument('--nat_ksize', nargs='+', type=int, default=[3, 3, 3, 3], help='kernel size for neighborhood attention')
parser.add_argument('--n_vars', type=int, default=7, help='number of variables in the input series')
parser.add_argument('--offset_range_factor', nargs='+', type=float, default=[-1, -1, -1, -1], help='restrict the offset value in a small range')
parser.add_argument('--no_off', nargs='+', type=int, default=[0,0,0,0], help='not use offset; True 1 False 0')
parser.add_argument('--dwc_pe', nargs='+', type=int, default=[0,0,0,0], help='use DWC-pe; True 1 False 0')
parser.add_argument('--fixed_pe', nargs='+', type=int, default=[0,0,0,0], help='use fixed pe; True 1 False 0')
parser.add_argument('--log_cpb', nargs='+', type=int, default=[0,0,0,0], help='use pe of SWin-v2; True 1 False 0')
parser.add_argument('--use_head_norm', type=int, default=1, help='use final LN layer; True 1 False 0')

# TimeBridge specific arguments
parser.add_argument('--period', type=int, default=24, help='period for TimeBridge')
parser.add_argument('--num_p', type=int, default=None, help='number of periods for TimeBridge')
parser.add_argument('--pd_layers', type=int, default=0, help='period decomposition layers for TimeBridge')
parser.add_argument('--ia_layers', type=int, default=2, help='number of integrated attention layers')
parser.add_argument('--ca_layers', type=int, default=2, help='number of cointegrated attention layers')
parser.add_argument('--stable_len', type=int, default=8, help='stable length for period normalization')
parser.add_argument('--attn_dropout', type=float, default=0.1, help='attention dropout rate')

# S_Mamba specific arguments  
parser.add_argument('--class_strategy', type=str, default='projection', help='class strategy for S_Mamba')
parser.add_argument('--d_state', type=int, default=16, help='SSM state expansion factor for S_Mamba')

# DishTS specific arguments
parser.add_argument('--dish_backbone', type=str, default='DLinear',
                    help='Backbone model for DishTS: DLinear, Informer, etc.')
parser.add_argument('--dish_init', type=str, default='standard',
                    help='DishTS CONET initialization: standard, avg, uniform')
parser.add_argument('--dish_activate', type=str_to_bool, default=True,
                    help='DishTS: use GELU activation in CONET')

# Koopa specific arguments
parser.add_argument('--dynamic_dim', type=int, default=128, help='latent dimension of koopman embedding')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension of en/decoder')
parser.add_argument('--hidden_layers', type=int, default=2, help='number of hidden layers of en/decoder')
parser.add_argument('--num_blocks', type=int, default=3, help='number of Koopa blocks')
parser.add_argument('--alpha', type=float, default=0.2, help='spectrum filter ratio')
parser.add_argument('--multistep', action='store_true', help='whether to use approximation for multistep K', default=False)

# TimeLLM local model path
parser.add_argument('--llm_model_path', type=str, default='huggyllama/llama-7b', help='local path or Hugging Face repo ID of the LLM')

args = parser.parse_args()

# Convert string parameters to lists (for DeformableTST)
# if hasattr(args, 'dims') and isinstance(args.dims, str):
#     args.dims = eval(args.dims)
# if hasattr(args, 'depths') and isinstance(args.depths, str):
#     args.depths = eval(args.depths)
# if hasattr(args, 'heads') and isinstance(args.heads, str):
#     args.heads = eval(args.heads)
# if hasattr(args, 'n_groups') and isinstance(args.n_groups, str):
#     args.n_groups = eval(args.n_groups)
# if hasattr(args, 'ksize') and isinstance(args.ksize, str):
#     args.ksize = eval(args.ksize)
# if hasattr(args, 'stride') and isinstance(args.stride, str):
#     args.stride = eval(args.stride)

# Set device
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu and torch.cuda.is_available():
    args.device = torch.device('cuda:{}'.format(args.gpu))
    print(f'Using GPU: {args.device}')
else:
    args.device = torch.device('cpu')
    print('Using CPU')

# Process data settings - logic from OnlineTSF
if args.dataset in data_settings:
    args.enc_in, args.c_out = data_settings[args.dataset][args.features]
    args.data_path = data_settings[args.dataset]['data']
    args.dec_in = args.enc_in
    if 'T' in data_settings[args.dataset]:
        args.target = data_settings[args.dataset]['T']
    if 'freq' in data_settings[args.dataset]:
        args.freq = data_settings[args.dataset]['freq']

# Process model-related settings
if args.model.endswith('_leak'):
    args.model = args.model[:-len('_leak')]
    args.leakage = True
if args.online_method and args.online_method.endswith('_leak'):
    args.online_method = args.online_method[:-len('_leak')]
    args.leakage = True

if args.tag and args.tag[0] != '_':
    args.tag = '_' + args.tag

# Set args.data attribute
if args.dataset == 'stock':
    args.data = 'stock'
else:
    args.data = args.data_path[:5] if args.data_path.startswith('ETT') else 'custom'
# args.data = args.data_path[:5] if args.data_path.startswith('ETT') else 'custom'

# Adjust border type based on learning environment
if args.learning_environment == 'offline':
    args.border_type = 'traditional'  # Use traditional data splits for offline learning

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_results_to_csv(args, all_results, mode='online'):
    """
    Save experiment results to CSV file
    
    Args:
        args: experiment arguments
        all_results: dictionary containing all iteration results
        mode: 'online' or 'offline'
    """
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{mode}_{args.model}_{args.dataset}.csv')
    
    # Check if file exists to decide write mode
    file_exists = os.path.exists(output_file)
    
    # Compute statistics
    metrics_list = ['mse', 'mae', 'rmse', 'wape', 'msmape', 'mase'] 
    metrics_stats = {}
    
    for metric in metrics_list:
        metric_values = [
            value for value in all_results.get(metric, [])
            if value is not None and value != 'N/A'
        ]
        if metric_values:
            metric_array = np.array(metric_values, dtype=float)
            metrics_stats[metric] = {
                'mean': float(np.mean(metric_array)),
                'std': float(np.std(metric_array))
            }
    
    with open(output_file, 'a' if file_exists else 'w', newline='') as f:
        writer = csv.writer(f)
            
        # Write experiment info
        writer.writerow(['Model', args.model])
        writer.writerow(['Dataset', args.dataset])
        writer.writerow(['Pred_len', args.pred_len])
        writer.writerow(['Settings', str(vars(args))])
        writer.writerow([])
        
        # Write per-iteration results
        writer.writerow(['Iteration', 'MSE', 'MAE', 'RMSE', 'WAPE (%)', 'MSMAPE (%)', 'MASE'])
        for i, iter_result in enumerate(all_results['iterations']):
            row = [f'Iter_{i+1}']
            for metric in metrics_list:
                value = iter_result.get(metric, 'N/A')
                row.append('N/A' if value is None else value)
            writer.writerow(row)
        
        # Write statistics
        writer.writerow([])
        writer.writerow(['Metric', 'Mean', 'Std'])
        for metric in metrics_list:
            if metric in metrics_stats:
                writer.writerow([
                    metric.upper(),
                    f"{metrics_stats[metric]['mean']:.6f}",
                    f"{metrics_stats[metric]['std']:.6f}"
                ])
        
        # Write efficiency metrics
        writer.writerow([])
        writer.writerow(['Efficiency Metrics'])
        efficiency = all_results.get('efficiency') or {}
        writer.writerow(['Parameter Count', efficiency.get('parameter_count', all_results.get('params', 'N/A'))])
        train_speed = efficiency.get('training_speed_ms')
        infer_speed = efficiency.get('inference_speed_ms', efficiency.get('inference_time_ms'))
        writer.writerow(['Training Speed (ms/batch)', 'N/A' if train_speed is None else f"{train_speed:.2f}"])
        writer.writerow(['Inference Speed (ms/sample)', 'N/A' if infer_speed is None else f"{infer_speed:.2f}"])
        
        # Add blank lines between experiments for readability
        writer.writerow([])
        writer.writerow([])
    
    
    print(f"\nResults saved to: {output_file}")
    return output_file


def run_online_learning(args):
    """Run online learning experiments using OnlineTSF framework"""
    print("=" * 50)
    print("RUNNING ONLINE LEARNING EXPERIMENTS")
    print("=" * 50)
    
    # Add validation printing
    print(f"[Debug] Online Method: {'General (No specific method)' if not args.online_method else args.online_method}")
    
    # Configure online learning parameters
    if args.model in ['TimeMixer', 'TimesNet', 'MICN', 'DishTS']:
        args.timeenc = 1
    else:
        args.timeenc = 2

    if args.model == 'TimesNet':
        args.d_model = 32      # Reduce model dimension
        args.d_ff = 128        # Adjust feed-forward dimension
        print(f"TimesNet: Adjusted d_model to {args.d_model}, d_ff to {args.d_ff} to reduce memory usage")

    if args.model.endswith('_Ensemble') and 'TCN' not in args.model and 'FSNet' not in args.model:
        args.model = args.model[:-len('_Ensemble')]
        args.ensemble = True
    else:
        args.ensemble = False

    # Configure GPU settings
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    # Set model ID
    args.model_id = f'{args.dataset}_{args.seq_len}_{args.pred_len}_{args.model}'
    if args.normalization is not None:
        args.model_id += '_' + args.normalization

    # Handle border_type and patience
    if args.border_type == 'online':
        args.patience = min(args.patience, 3)

    # Default: use Exp_Online
    if not args.online_method:
        print("[Debug] Using General Online Learning (Exp_Online)")
        Exp = Exp_Online
    else:
        print(f"[Debug] Using specific online method: {args.online_method}")
        args.train_epochs = min(args.train_epochs, 25)
        args.save_opt = True
        
        if args.online_method == 'Online':
            if not args.force_retrain:
                args.pretrain = True
                args.only_test = True
            else:
                args.pretrain = False
                args.only_test = False
        
        Exp = getattr(online_exps, 'Exp_' + args.online_method)
        
        if args.online_method == 'SOLID':
            args.pretrain = True
            args.only_test = True
            args.online_method = 'Online'
            if not args.whole_model:
                args.freeze = True
    
    # Process hyperparameters
    if args.override_hyper and args.model in settings.hyperparams:
        for k, v in settings.get_hyperparams(args.dataset, args.model, args, args.reduce_bs).items():
            args.__setattr__(k, v)
    
    # Handle models that require x_mark
    if args.model in settings.need_x_mark:
        args.optim = 'AdamW'
        args.patience = 3

    # Set borders for online learning
    if hasattr(args, 'border_type'):
        settings.get_borders(args)
    
    train_data, train_loader, vali_data, vali_loader = None, None, None, None
    test_data, test_loader = None, None
    
    all_results = {
            'mse': [],  # MSE values per iteration
            'mae': [],  # MAE values per iteration
            'rmse': [], 'wape': [], 'msmape': [], 'mase': [],
            'iterations': [],  # Detailed results per iteration
            'params': None,  # Model parameter count
            'efficiency': None  # Efficiency metrics
        }    

    for ii in range(args.itr):
        # Use custom seed
        fix_seed = args.seed + ii if args.seed is not None else (2023 + ii)
        setup_seed(fix_seed)
        print(f'Iteration {ii+1}, Seed: {fix_seed}')
        
        # Create experiment setting string (OnlineTSF format)
        flag = args.online_method.lower() if args.online_method else args.border_type if args.border_type else args.data
        
        # Handle special flag settings
        if args.online_method:
            if not args.border_type:
                if args.online_method == 'Online':
                    flag = args.data
                    args.checkpoints = ""
                else:
                    flag = args.data + '_' + flag
            
            if flag == 'fsnet':
                flag = 'online'
            
            if 'proceed' in flag:
                if not args.freeze:
                    flag += "_fulltune"
                if not args.pretrain:
                    flag += "_new"
                flag += f"_{args.lradj}"
                flag += f'_{args.tune_mode}_btl{args.bottleneck_dim}_ema{args.ema}'
                if args.concept_dim:
                    flag += f'_mid{args.concept_dim}'
                if not args.individual_generator:
                    flag += '_share'
                if args.share_encoder:
                    flag += '_share_enc'
                if args.wo_clip:
                    flag += '_noclip'
        
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_lr{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            flag,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.learning_rate,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)
        
        # Initialize experiment
        exp = Exp(args)
        
        if train_data is None:
            train_data, train_loader = exp._get_data('train')
        if not hasattr(args, 'borders'):
            args.borders = train_data.borders
            if args.border_type != 'online' and args.model == 'PatchTST':
                settings.drop_last_PatchTST(args) # SOLID dropout the last when data split = 7:2:1
        exp.wrap_data_kwargs['borders'] = args.borders
        
        path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
        
        # Training phase
        if args.online_method not in ['Online', 'SOLID', 'ER', 'DERpp']:
            print('Checkpoints in', path)
            if (args.only_test or args.do_valid) and os.path.exists(path):
                print('Loading', path)
                exp.load_checkpoint(path)
                print('Learning rate of model_optim is', exp.model_optim.param_groups[0]['lr'])
            else:
                print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                _, train_data, train_loader, vali_data, vali_loader = exp.train(setting, train_data, train_loader,
                                                                                vali_data, vali_loader)
                torch.cuda.empty_cache()

        if args.online_learning_rate is not None and not isinstance(exp, Exp_SOLID):
            for j in range(len(exp.model_optim.param_groups)):
                exp.model_optim.param_groups[j]['lr'] = args.online_learning_rate
            print('Adjust learning rate of model_optim to', exp.model_optim.param_groups[0]['lr'])

        if args.do_valid and args.online_method and args.local_rank <= 0:
            assert isinstance(exp, Exp_Online)
            result = exp.online(online_data=vali_data if isinstance(vali_data, Dataset_Recent) else None,
                                  phase='val', show_progress=True)
            if isinstance(result, dict):
                mse, mae = result['mse'], result['mae']
            else:
                mse, mae = result[:2]
            print('Best Valid MSE:', mse)
            all_results['mse'].append(mse)
            all_results['mae'].append(mae)
            continue

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            setup_seed(fix_seed)
            result = exp.predict(path, setting, True)
            if isinstance(result, dict):
                mse, mae = result['mse'], result['mae']
            else:
                mse, mae = result[:2]
        elif not args.wo_test and not args.train_only and args.local_rank <= 0:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            setup_seed(fix_seed)
            if not args.wo_valid:
                vali_data = None
                torch.cuda.empty_cache()
                gc.collect()
                exp.update_valid()
            # Use online method for testing
            result = exp.online(test_data)
            
            # Process return values
            if isinstance(result, dict):
                mse = result['mse']
                mae = result['mae']
                rmse = result.get('rmse', None)
                wape = result.get('wape', None)
                msmape = result.get('msmape', None)
                mase = result.get('mase', None)
            else:
                mse, mae = result[:2]
                rmse, wape, msmape, mase = None, None, None, None
        
        if ii == 0:  # Only need to record once
            all_results['params'] = sum(p.numel() for p in exp.model.parameters())
            # Get efficiency metrics from experiment (including training speed)
            if hasattr(exp, 'efficiency_metrics') and exp.efficiency_metrics:
                all_results['efficiency'] = exp.efficiency_metrics.copy()
            else:
                # If experiment has no efficiency metrics, compute inference speed manually
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(100):  # Average over 100 runs
                        _ = exp.model(torch.randn(1, args.seq_len, args.enc_in).to(args.device))
                inference_time = (time.time() - start_time) / 100
                all_results['efficiency'] = {
                    'parameter_count': all_results['params'],
                    'inference_time_ms': inference_time * 1000,  # Convert to milliseconds
                }

        # Record per-iteration results
        iter_result = {
            'seed': fix_seed,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'wape': wape,
            'msmape': msmape,
            'mase': mase
        }
        all_results['iterations'].append(iter_result)
        all_results['mse'].append(mse)
        all_results['mae'].append(mae)
        all_results['rmse'].append(rmse)
        all_results['wape'].append(wape)
        all_results['msmape'].append(msmape)
        all_results['mase'].append(mase)

        torch.cuda.empty_cache()
    
    # Print final results - compute statistics for numeric data only
    for k in ['mse', 'mae']:
        if k in all_results and len(all_results[k]) > 0:
            arr = np.array(all_results[k])
            print(f'{k.upper()}: {arr.mean():.6f} ± {arr.std():.6f}')

     # Compute statistics
    mse_array = np.array(all_results['mse'])
    mae_array = np.array(all_results['mae'])
    
    # Format results output
    final_results = {
        'args': vars(args),
        'iterations': all_results['iterations'],
        'metrics': {
            'mse': {
                'values': all_results['mse'],
                'mean': float(np.mean(mse_array)),
                'std': float(np.std(mse_array))
            },
            'mae': {
                'values': all_results['mae'],
                'mean': float(np.mean(mae_array)),
                'std': float(np.std(mae_array))
            }
        },
        'efficiency': all_results['efficiency']
    }

    # Save results to CSV
    output_file = save_results_to_csv(args, all_results, mode='online')
    
    print("ONLINE LEARNING RESULTS:")
    print(f"MSE: {final_results['metrics']['mse']['mean']:.6f} ± {final_results['metrics']['mse']['std']:.6f}")
    print(f"MAE: {final_results['metrics']['mae']['mean']:.6f} ± {final_results['metrics']['mae']['std']:.6f}")
    
    return all_results


def run_offline_learning(args):
    """Run offline learning experiments using traditional training approach"""
    print("=" * 50)
    print("SWITCHING TO OFFLINE LEARNING MODE")  
    print("=" * 50)

    args.border_type = 'traditional'
    if not hasattr(args, 'use_time'):
        args.use_time = False
    if not hasattr(args, 'seasonal_patterns'):
        args.seasonal_patterns = 'Monthly'
    if not hasattr(args, 'task_name'):
        args.task_name = 'long_term_forecast'
    if not hasattr(args, 'is_training'):
        args.is_training = 1
    if not hasattr(args, 'model_id'):
        args.model_id = '' # Auto-set to empty string
    
    # Set data name
    # args.data = args.data_path[:5] if args.data_path.startswith('ETT') else 'custom'
    if args.dataset == 'stock':
        args.data = 'stock'
    else:
        args.data = args.data_path[:5] if args.data_path.startswith('ETT') else 'custom'

    # Set timeenc for offline mode as well
    if args.model in ['TimeMixer', 'TimesNet', 'MICN', 'DishTS']:
        args.timeenc = 1  # MICN/DishTS requires timeenc=1
    else:
        args.timeenc = 2  # Other models use timeenc=2

    # TimesNet: reduce memory usage (needed for both online and offline)
    if args.model == 'TimesNet':
        args.d_model = 32      # Reduce model dimension
        args.d_ff = 128        # Adjust feed-forward dimension
        print(f"TimesNet: Adjusted d_model to {args.d_model}, d_ff to {args.d_ff} to reduce memory usage")

    all_results = {
        'mse': [], 'mae': [], 'rmse': [], 'wape': [], 'msmape': [], 'mase': [],
        'params': None, 'efficiency': None, 'iterations': []
    }

    # Run offline training
    for ii in range(args.itr):
        # # Set random seed
        # if args.border_type:
        #     if args.model in ['PatchTST', 'iTransformer']:
        #         fix_seed = 2021 + ii
        #     else:
        #         fix_seed = 2023 + ii
        # else:
        #     fix_seed = 2023 + ii if args.model == 'iTransformer' else 2021 + ii
        # setup_seed(fix_seed)
        # print('Offline Seed:', fix_seed)

        # Use custom seed
        fix_seed = args.seed + ii if args.seed is not None else (2023 + ii)
        setup_seed(fix_seed)
        print('Offline Seed:', fix_seed)
        
        # Create experiment instance
        exp = Offline_Exp(args)
        
        # Generate setting string (following online format)
        model_name_with_norm = args.model
        norm = getattr(args, 'normalization', None)
        if norm and str(norm).lower() != 'none':
            model_name_with_norm = f"{args.model}_{norm}"
            
        setting = '{}_{}_{}_{}_offline_ft{}_sl{}_ll{}_pl{}_lr{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.dataset,  # Dataset name
            args.seq_len,  # Sequence length
            args.pred_len, # Prediction length
            model_name_with_norm,    # Model name
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.learning_rate,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)
        
        if args.is_training:
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # Get all metrics
            metrics = exp.test(setting)
            if isinstance(metrics, tuple):
                mse, mae = metrics
                # If only mse and mae returned, set others to N/A
                rmse, wape, msmape, mase = None, None, None, None
            else:
                # If dict returned, extract all metrics
                mse = metrics.get('mse', None)
                mae = metrics.get('mae', None)
                rmse = metrics.get('rmse', None)
                wape = metrics.get('wape', None)
                msmape = metrics.get('msmape', None)
                mase = metrics.get('mase', None)
        else:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            metrics = exp.test(setting, test=1)
            if isinstance(metrics, tuple):
                mse, mae = metrics
                rmse, wape, msmape, mase = None, None, None, None
            else:
                mse = metrics.get('mse', None)
                mae = metrics.get('mae', None)
                rmse = metrics.get('rmse', None)
                wape = metrics.get('wape', None)
                msmape = metrics.get('msmape', None)
                mase = metrics.get('mase', None)
        
        # Record per-iteration results
        iter_result = {
            'seed': fix_seed,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'wape': wape,
            'msmape': msmape,
            'mase': mase
        }
        all_results['iterations'].append(iter_result)
        all_results['mse'].append(mse)
        all_results['mae'].append(mae)
        all_results['rmse'].append(rmse)
        all_results['wape'].append(wape)
        all_results['msmape'].append(msmape)
        all_results['mase'].append(mase)
        
        # Record model params and efficiency metrics
        if ii == 0:
            all_results['params'] = sum(p.numel() for p in exp.model.parameters())
            # Get efficiency metrics from experiment (including training speed)
            if hasattr(exp, 'efficiency_metrics') and exp.efficiency_metrics:
                all_results['efficiency'] = exp.efficiency_metrics.copy()
            else:
                # If no efficiency metrics, compute inference speed manually
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(100):
                        _ = exp.model(torch.randn(1, args.seq_len, args.enc_in).to(args.device))
                inference_time = (time.time() - start_time) / 100
                all_results['efficiency'] = {
                    'parameter_count': all_results['params'],
                    'inference_time_ms': inference_time * 1000,
                }

        torch.cuda.empty_cache()

    # Compute statistics
    mse_array = np.array(all_results['mse'])
    mae_array = np.array(all_results['mae'])
    
    # Print final results
    print("OFFLINE LEARNING RESULTS:")
    for k in ['mse', 'mae']:
        if k in all_results and len(all_results[k]) > 0:
            arr = np.array(all_results[k])
            print(f'{k.upper()}: {arr.mean():.6f} ± {arr.std():.6f}')
    
    # Save results to CSV
    output_file = save_results_to_csv(args, all_results, mode='offline')


if __name__ == '__main__':
    if args.learning_environment == 'offline':
        run_offline_learning(args)
    else:
        print("=" * 50)
        print("SWITCHING TO ONLINE LEARNING MODE")
        print("=" * 50)
        run_online_learning(args)
