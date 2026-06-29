# Based on settings_pzy.py, integrating extensions from settings_lhy.py and settings_cq.py

need_x_y_mark = ['Autoformer', 'Transformer', 'Informer', 'TimeMixer', 'TimesNet', 'MICN', 'SCINet', 'TimeBridge', 'S_Mamba', 'BiMamba4TS', 'Koopa', 'DishTS', 'TimeLLM']
need_x_mark = ['TCN', 'FSNet', 'OneNet', 'iTransformer', 'TimeMixer']
need_x_mark += [name + '_Ensemble' for name in need_x_mark]
no_extra_param = ['Online', 'ER', 'DERpp']
peft_methods = ['lora', 'adapter', 'ssf', 'mam_adapter', 'basic_tuning']

data_settings = {
    'wind_N2': {'data': 'wind_N2.csv', 'T':'FR51', 'M':[254, 254], 'prefetch_batch_size': 16},
    'ECL':{'data':'electricity.csv','T':'OT','M':[321,321],'S':[1,1],'MS':[321,1], 'prefetch_batch_size': 10},
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7],'S':[1,1],'MS':[7,1], 'prefetch_batch_size': 128},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7],'S':[1,1],'MS':[7,1], 'prefetch_batch_size': 128},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7],'S':[1,1],'MS':[7,1], 'prefetch_batch_size': 128},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7],'S':[1,1],'MS':[7,1], 'prefetch_batch_size': 128},
    'Solar':{'data':'solar_AL.txt','T': 136,'M':[137,137],'S':[1,1],'MS':[137,1], 'prefetch_batch_size': 32},
    'Weather':{'data':'weather.csv','T':'OT','M':[21,21],'S':[1,1],'MS':[21,1], 'prefetch_batch_size': 64},
    'WTH':{'data':'WTH.csv','T':'OT','M':[12,12],'S':[1,1],'MS':[12,1], 'prefetch_batch_size': 64},
    'Traffic': {'data': 'traffic.csv', 'T':'OT', 'M':[862,862], 'prefetch_batch_size': 2},
    'METR_LA': {'data':'metr-la.csv','T': '773869','M':[207,207],'S':[1,1],'MS':[207,1], 'prefetch_batch_size': 16},
    'PEMS_BAY': {'data':'pems-bay.csv','T': 400001,'M':[325,325],'S':[1,1],'MS':[325,1], 'prefetch_batch_size': 10},
    'NYC_BIKE': {'data':'nyc-bike.h5','T': 0,'M':[500,500],'S':[1,1],'MS':[500,1], 'prefetch_batch_size': 4, 'feat_dim': 2},
    'PeMSD4': {'data':'PeMSD4/PeMSD4.npz','T': 0,'M':[921,921],'S':[1,1],'MS':[921,1], 'prefetch_batch_size': 2, 'feat_dim': 3},
    'PeMSD8': {'data':'PeMSD8/PeMSD8.npz','T': 0,'M':[510,510],'S':[1,1],'MS':[510,1], 'prefetch_batch_size': 6, 'feat_dim': 3},
    'Exchange': {'data': 'exchange_rate.csv', 'T':'OT', 'M':[8,8], 'prefetch_batch_size': 128},
    'exchange_rate': {'data': 'exchange_rate.csv', 'T':'OT', 'M':[8,8], 'prefetch_batch_size': 128},
    'Illness': {'data': 'illness.csv', 'T':'OT', 'M':[7,7], 'prefetch_batch_size': 128},
    # BeijingAQ datasets
    'BeijingAQ_Aoti': {'data': 'BeijingAQ/Aotizhongxin_2015_2025_cleaned.csv', 'T': 'AQI', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1], 'prefetch_batch_size': 128 },
    'BeijingAQ_Wanliu': {'data': 'BeijingAQ/Wanliu_2015_2025_cleaned.csv', 'T': 'AQI', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1], 'prefetch_batch_size': 128 },
    'BeijingAQ_Daxing': {'data': 'BeijingAQ/Daxing_2015_2025_cleaned.csv', 'T': 'AQI', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1], 'prefetch_batch_size': 128 },
    'BeijingAQ_Tiantan': {'data': 'BeijingAQ/Tiantan_2015_2025_cleaned.csv', 'T': 'AQI', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1], 'prefetch_batch_size': 128 },
    'BeijingAQ_Guanyuan': {'data': 'BeijingAQ/Guanyuan_2015_2025_cleaned.csv', 'T': 'AQI', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1], 'prefetch_batch_size': 128 },
    # Additional datasets
    'electricity': {'data': 'electricity.csv','T': 'Sub_metering_3','M': [7, 7],'S': [1, 1],'MS': [7, 1],'prefetch_batch_size': 128},
    'NYC_Taxi': {'data': 'NYC_Taxi.csv', 'T': 'Value', 'M': [1, 1], 'S': [1, 1], 'MS': [1, 1], 'prefetch_batch_size': 128},
    'stock': {'data': 'stock.csv', 'T': 'DailyPrice', 'M': [8, 8], 'S': [1, 1], 'MS': [8, 1], 'freq': 'd', 'prefetch_batch_size': 128},
    'wind': {'data': 'wind.csv', 'T': 'windspeed', 'M': [1, 1], 'S': [1, 1], 'MS': [1, 1], 'prefetch_batch_size': 128},
    'CHI_Crime': {'data': 'CHI_Crime_Cleaned_Hourly.csv', 'T': 'THEFT', 'M': [4, 4], 'prefetch_batch_size': 128},
    'saugeen_river': {'data': 'saugeen_river.csv','T': 'flow','M': [1, 1], 'S': [1, 1], 'MS': [1, 1],'prefetch_batch_size': 128},
    'Mooloolaba_Waves': {'data': 'mooloolaba_waves_complete_records.csv', 'T': 'Hs', 'M': [6, 6], 'S': [1, 1], 'MS': [6, 1], 'prefetch_batch_size': 128},
    'beijing':{'data':'beijing.csv', 'T':'shidu', 'M':[2,2],'prefetch_batch_size': 128},
    'guangzhou': {'data': 'guangzhou.csv', 'T': 'shidu', 'M': [2, 2], 'prefetch_batch_size': 128},
    'shenyang': {'data': 'shenyang.csv', 'T': 'shidu', 'M': [2, 2], 'prefetch_batch_size': 128},
}

def get_borders(args):
    if args.border_type == 'online':
        if args.data.startswith('ETTh'):
            border1s = [0, 4*30*24 - args.seq_len, 5*30*24 - args.seq_len]
            border2s = [4*30*24, 5*30*24, 20*30*24]
            args.borders = (border1s, border2s)
        elif args.data.startswith('ETTm'):
            border1s = [0, 4*30*24*4 - args.seq_len, 5*30*24*4 - args.seq_len]
            border2s = [4*30*24*4, 5*30*24*4, 20*30*24*4]
            args.borders = (border1s, border2s)
        else:
            args.ratio = (0.2, 0.75)

hyperparams = {
    'PatchTST': {'e_layers': 3},
    'MTGNN': {},
    'LightCTS': {},
    'Crossformer': {'lradj': 'Crossformer', 'e_layers': 3, 'seg_len': 24, 'd_ff': 512, 'd_model': 256, 'n_heads': 4, 'dropout': 0.2},
    'DLinear': {},
    'GPT4TS': {'e_layers': 3, 'd_model': 768, 'n_heads': 4, 'd_ff': 768, 'dropout': 0.3},
    'iTransformer': {'e_layers': 3, 'd_model': 512, 'd_ff': 512, 'activation': 'gelu', 'timeenc': 1, 'patience': 3, 'train_epochs': 10, },
    'Autoformer': {'train_epochs': 10, 'timeenc': 1},
    'Informer': {'train_epochs': 10, 'timeenc': 1},

    'TimeMixer': {
        'e_layers': 2,
        'd_model': 512,
        'd_ff': 512,
        'dropout': 0.05,
        'activation': 'gelu',
        'down_sampling_layers': 2,
        'down_sampling_window': 2,
        'decomp_method': 'moving_avg',
        'moving_avg': 25,
        'channel_independence': False
    },
    
    'TimeBridge': {
        'e_layers': 2,
        'd_model': 128,
        'n_heads': 4,
        'd_ff': 256,
        'dropout': 0.1,
        'attn_dropout': 0.05,
        'activation': 'gelu',
        'period': 24,
        'ia_layers': 1,
        'pd_layers': 1,
        'ca_layers': 0,
        'stable_len': 6,
        'revin': 1,
        'num_p': None,
    },

    'DeformableTST': {
        'revin': 1,
        'revin_affine': 0,
        'revin_subtract_last': 0,
        'stem_ratio': 8,
        'down_ratio': 2,
        'fmap_size': 768,
        'dims': [64, 128, 256, 512],
        'depths': [1, 1, 3, 1],
        'drop_path_rate': 0.3,
        'layer_scale_value': [-1, -1, -1, -1],
        'use_pe': [1, 1, 1, 1],
        'use_lpu': [1, 1, 1, 1],
        'local_kernel_size': [3, 3, 3, 3],
        'expansion': 4,
        'drop': 0.0,
        'use_dwc_mlp': [1, 1, 1, 1],
        'heads': [4, 8, 16, 32],
        'attn_drop': 0.0,
        'proj_drop': 0.0,
        'stage_spec': [['D'], ['D'], ['D', 'D', 'D'], ['D']],
        'window_size': [3, 3, 3, 3],
        'nat_ksize': [3, 3, 3, 3],
        'ksize': [9, 7, 5, 3],
        'stride': [8, 4, 2, 1],
        'n_groups': [2, 4, 8, 16],
        'offset_range_factor': [-1, -1, -1, -1],
        'no_off': [0, 0, 0, 0],
        'dwc_pe': [0, 0, 0, 0],
        'fixed_pe': [0, 0, 0, 0],
        'log_cpb': [0, 0, 0, 0],
        'head_dropout': 0.1,
        'head_type': 'Flatten',
        'use_head_norm': 1,
        'learning_rate': 0.0001,
        'train_epochs': 50,
        'warmup_epochs': 5,
        'batch_size': 512,
    },

    'Leddam': {
        'n_layers': 3,
        'pe_type': 'no',
        'd_model': 256,
        'dropout': 0.0,
        'revin': True,
        'kernel_size': 25,
        'learning_rate': 1e-4,
        'batch_size': 32,
        'train_epochs': 100,
    },

    'ModernTCN': {
        'stem_ratio': 6,
        'downsample_ratio': 2,
        'ffn_ratio': 2,
        'dims': [256,256,256,256],
        
        'num_blocks': [1,1,1,1],
        'large_size': [31,29,27,13],
        'small_size': [5,5,5,5],
        'dw_dims': [256,256,256,256],
        'small_kernel_merged': False,
        'use_multi_scale': True,
        
        'revin': 1,
        'affine': 0,
        'subtract_last': 0,
        'decomposition': 0,
        'kernel_size': 25,
        'individual': 0,
        'patch_size': 16,
        'patch_stride': 8,
        'learning_rate': 0.0001,
        'dropout': 0.05,
        'head_dropout': 0.0,
    },

    'duet': {
        'e_layers': 2,
        'd_model': 256,
        'n_heads': 8,
        'd_ff': 512,
        'dropout': 0.1,
        'fc_dropout': 0.1,
        'activation': 'gelu',
        'output_attention': False,
        'factor': 3,
        'moving_avg': 25,
        'num_experts': 4,
        'k': 2,
        'noisy_gating': True,
        'hidden_size': 128,
        'CI': True,
    },

    'DishTS': {
        'dish_init': 'standard',
        'dish_activate': True,
        'train_epochs': 100,
    },

    'S_Mamba': {
        'd_model': 512,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        'd_ff': 2048,
        'dropout': 0.1,
        'activation': 'gelu',
        'output_attention': False,
        'embed': 'timeF',
        'freq': 'h',
        'd_state': 32,
        'use_norm': 1,
        'class_strategy': 'projection',
        'learning_rate': 0.0001,
        'batch_size': 32,
        'train_epochs': 100,
    },

    'TimeLLM': {
        'd_model': 32,
        'd_ff': 128,
        'n_heads': 8,
        'dropout': 0.1,
        'llm_model': 'LLAMA',
        'llm_dim': 4096,
        'llm_layers': 32,
        'patch_len': 16,
        'stride': 8,
        'learning_rate': 0.01,
        'batch_size': 24,
        'train_epochs': 100,
        'task_name': 'long_term_forecast',
        'prompt_domain': 0,
    },

    'Aurora': {
        'n_heads': 8,
        'dropout': 0.2,
        'aurora_hidden_size': 512,
        'aurora_intermediate_size': 1024,
        'aurora_enc_layers': 6,
        'aurora_dec_layers': 6,
        'aurora_sampling_steps': 50,
        'inference_token_len': 48,
        'aurora_num_samples': 100,
        'learning_rate': 0.00002,
        'batch_size': 32,
        'train_epochs': 30,
    },
}

def get_hyperparams(data, model, args, reduce_bs=True):
    hyperparam: dict = hyperparams[model]
    if model == 'iTransformer':
        if data == 'Traffic':
            hyperparam['e_layers'] = 4
        elif 'ETT' in data:
            hyperparam['e_layers'] = 2
            if data == 'ETTh1':
                hyperparam['d_model'] = 256
                hyperparam['d_ff'] = 256
            else:
                hyperparam['d_model'] = 128
                hyperparam['d_ff'] = 128

    if model == 'PatchTST':
        if args.lradj != 'type3':
            if data in ['ETTh1', 'ETTh2', 'Weather', 'Exchange', 'wind']:
                hyperparam['lradj'] = 'type3'
            elif data in ['Illness']:
                hyperparam['lradj'] = 'constant'
            else:
                hyperparam['lradj'] = 'TST'
        if data in ['ETTh1', 'ETTh2', 'Illness']:
            hyperparam.update(**{'dropout': 0.3, 'fc_dropout': 0.3, 'n_heads': 4, 'd_model': 16, 'd_ff': 128})
        elif data in ['ETTm1', 'ETTm2', 'Weather', 'ECL', 'Traffic']:
            hyperparam.update(**{'dropout': 0.2, 'fc_dropout': 0.2, 'n_heads': 16, 'd_model': 128, 'd_ff': 256})
        else:
            hyperparam.update(**{'dropout': 0.2, 'fc_dropout': 0.2, 'n_heads': 16, 'd_model': 64, 'd_ff': 128})

    elif model == 'Crossformer':
        if data == 'ECL' or args.lradj == 'fixed':
            hyperparam['lradj'] = 'fixed'
        if reduce_bs:
            if data in ['PeMSD4']:
                hyperparam['batch_size'] = 4
            elif data in ['Traffic']:
                hyperparam['batch_size'] = 4
            elif data in ['NYC_BIKE', 'NYC_TAXI', 'PeMSD8']:
                hyperparam['batch_size'] = 8
        else:
            if data in ['Traffic', 'PeMSD4'] and args.pred_len >= 720:
                hyperparam['batch_size'] = 24
            if data in ['PeMSD8'] and args.pred_len >= 720:
                hyperparam['batch_size'] = 16

        if data in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Illness', 'wind', 'Exchange']:
            hyperparam['d_model'] = 256
            hyperparam['n_heads'] = 4
        else:
            hyperparam['d_model'] = 64
            hyperparam['n_heads'] = 2

        if data in ['Traffic', 'ECL']:
            hyperparam['d_ff'] = 128

        if data in ['Illness']:
            hyperparam['e_layers'] = 2

    elif model == 'GPT4TS':
        if data == 'ETTh1':
            hyperparam['lradj'] = 'typy4'
            hyperparam['tmax'] = 20
        elif data == 'ETTh2':
            hyperparam['dropout'] = 1
            hyperparam['tmax'] = 20
        elif data == 'Traffic':
            hyperparam['dropout'] = 0.3
        elif data == 'ECL':
            hyperparam['tmax'] = 10
        elif data == 'Illness':
            hyperparam['patch_size'] = 24
            hyperparam['batch_size'] = 16

        if data in ['ETTm1', 'ETTm2', 'ECL', 'Traffic', 'Weather', 'WTH']:
            hyperparam['seq_len'] = 512

        if data.startswith('ETTm'):
            hyperparam['stride'] = 16
        elif args.seq_len == 104:
            hyperparam['stride'] = 2

    elif model == 'TimeBridge':
        if data in ['ETTh1', 'ETTh2']:
            hyperparam.update({
                'period': 24,
                'd_model': 128,
                'n_heads': 4,
                'd_ff': 256,
            })
        elif data in ['ETTm1', 'ETTm2']:
            hyperparam.update({
                'period': 96,
                'd_model': 256,
                'n_heads': 8,
                'd_ff': 512,
            })
        elif data in ['Weather']:
            hyperparam.update({
                'period': 24,
                'd_model': 256,
                'n_heads': 8,
            })
        elif data in ['Traffic', 'ECL']:
            hyperparam.update({
                'period': 24,
                'd_model': 64,
                'n_heads': 4,
                'd_ff': 128,
            })
        elif data in ['BeijingAQ_Aoti', 'BeijingAQ_Wanliu', 'BeijingAQ_Daxing', 'BeijingAQ_Tiantan', 'BeijingAQ_Guanyuan']:
            hyperparam.update({
                'period': 24,
                'd_model': 128,
                'n_heads': 4,
                'd_ff': 256,
            })

    elif model == 'DeformableTST':
        hyperparam['fmap_size'] = args.seq_len
        hyperparam['n_vars'] = args.enc_in
        
        if data in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
            hyperparam.update({
                'dims': [32, 64, 128, 256],
                'heads': [2, 4, 8, 16],
                'n_groups': [2, 4, 8, 16],
                'batch_size': 256,
                'stem_ratio': 4,
                'ksize': [7, 5, 3, 3],
                'stride': [4, 2, 2, 1],
            })
        elif data in ['Weather', 'ECL']:
            hyperparam.update({
                'dims': [64, 128, 256, 512],
                'heads': [4, 8, 16, 32],
                'batch_size': 128,
            })
        elif data in ['Traffic']:
            hyperparam.update({
                'dims': [32, 64, 128, 256],
                'heads': [2, 4, 8, 16],
                'batch_size': 64,
            })
        elif data in ['BeijingAQ_Aoti', 'BeijingAQ_Wanliu', 'BeijingAQ_Daxing', 'BeijingAQ_Tiantan', 'BeijingAQ_Guanyuan']:
            hyperparam.update({
                'dims': [32, 64, 128, 256],
                'heads': [2, 4, 8, 16],
                'n_groups': [2, 4, 8, 16],
                'batch_size': 32,
                'stem_ratio': 4,
                'ksize': [7, 5, 3, 3],
                'stride': [4, 2, 2, 1],
            })
        elif data in ['wind', 'NYC_Taxi']:
            hyperparam.update({
                'dims': [16, 32, 64, 128],
                'heads': [1, 2, 4, 8],
                'n_groups': [1, 2, 4, 8],
                'batch_size': 128,
                'stem_ratio': 2,
                'ksize': [5, 3, 3, 3],
                'stride': [2, 2, 1, 1],
            })
        elif data in ['stock']:
            hyperparam.update({
                'dims': [32, 64, 128, 256],
                'heads': [2, 4, 8, 16],
                'n_groups': [2, 4, 8, 16],
                'batch_size': 64,
                'stem_ratio': 4,
                'ksize': [7, 5, 3, 3],
                'stride': [4, 2, 2, 1],
            })
        elif data in ['CHI_Crime']:
            hyperparam.update({
                'dims': [32, 64, 128, 256],
                'heads': [2, 4, 8, 16],
                'n_groups': [2, 4, 8, 16],
                'batch_size': 128,
                'stem_ratio': 4,
                'ksize': [7, 5, 3, 3],
                'stride': [4, 2, 2, 1],
            })
        else:
            # Default fallback for unmatched datasets (e.g., electricity -> 'custom')
            # stem_ratio=4 ensures seq_len=96 won't shrink to L=1 after downsampling
            hyperparam.update({
                'stem_ratio': 4,
                'ksize': [7, 5, 3, 3],
                'stride': [4, 2, 2, 1],
            })
        
        if args.pred_len >= 720:
            hyperparam['train_epochs'] = 30
        elif args.pred_len >= 336:
            hyperparam['train_epochs'] = 40

    elif model == 'Leddam':
        if data in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
            hyperparam.update({
                'd_model': 256,
                'n_layers': 3,
                'pe_type': 'no',
                'dropout': 0.0,
            })
        elif data in ['Weather', 'ECL', 'Traffic']:
            hyperparam.update({
                'd_model': 512,
                'n_layers': 3, 
                'pe_type': 'sincos',
                'dropout': 0.1,
            })
        elif data in ['BeijingAQ_Aoti', 'BeijingAQ_Wanliu', 'BeijingAQ_Daxing', 'BeijingAQ_Tiantan', 'BeijingAQ_Guanyuan']:
            hyperparam.update({
                'd_model': 256,
                'n_layers': 3,
                'pe_type': 'no',
                'dropout': 0.0,
            })

    return hyperparam


# Merged pretrain_lr_online_dict from lhy and cq
pretrain_lr_online_dict = {
    'TCN': {'ECL': 0.003, 'ETTh2': 0.003, 'ETTm1': 0.001, 'Weather': 0.001, 'Traffic': 0.003},
    'TCN_RevIN': {'ECL': 0.003, 'ETTh2': 0.001, 'ETTm1': 0.0001, 'Weather': 0.001, 'Traffic': 0.003},
    'TCN_Ensemble': {'ECL': 0.003, 'ETTh2': 0.003, 'ETTm1': 0.0003, 'Weather': 0.001, 'Traffic': 0.003},
    'FSNet_RevIN': {'ECL': 0.003, 'ETTh2': 0.003, 'ETTm1': 0.001, 'Weather': 0.003, 'Traffic': 0.003},
    'GPT4TS': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.001, 'Weather': 0.0001, 'ECL': 0.0001},
    'PatchTST': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001,
                 'stock': 0.0001, 'wind': 0.0001, 'NYC_Taxi': 0.0001, 'CHI_Crime': 0.0001, 
                 'saugeen_river': 0.0001, 'beijing': 0.0001, 'guangzhou': 0.0001, 'shenyang': 0.0001,
                 'BeijingAQ_Aoti': 0.0001, 'BeijingAQ_Wanliu': 0.0001, 'BeijingAQ_Daxing': 0.0001, 
                 'BeijingAQ_Tiantan': 0.0001, 'BeijingAQ_Guanyuan': 0.0001, 'Mooloolaba_Waves': 0.0001},
    'iTransformer': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.001, 'Weather': 0.00001, 'ECL': 0.0005},
    'NLinear': {'ETTh2': 0.05, 'ETTm1': 0.05, 'Traffic': 0.005, 'Weather': 0.01, 'ECL': 0.01},
    'Informer_RevIN': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001},
    'Autoformer_RevIN': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001},
    'Informer': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001,
                 'electricity': 0.0001, 'NYC_Taxi': 0.0001, 'stock': 0.0001, 'wind': 0.0001,
                 'CHI_Crime': 0.0001, 'saugeen_river': 0.0001, 'Mooloolaba_Waves': 0.0001,
                 'beijing': 0.0001, 'guangzhou': 0.0001, 'shenyang': 0.0001},
    'Autoformer': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001},
    'TimeBridge': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001,
                   'electricity': 0.0001, 'NYC_Taxi': 0.0001, 'stock': 0.0001, 'wind': 0.0001,
                   'CHI_Crime': 0.0001, 'saugeen_river': 0.0001, 'Mooloolaba_Waves': 0.0001,
                   'beijing': 0.0001, 'guangzhou': 0.0001, 'shenyang': 0.0001},
    'duet': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001,
             'stock': 0.0001, 'wind': 0.0001, 'NYC_Taxi': 0.0001, 'CHI_Crime': 0.0001, 
             'saugeen_river': 0.0001, 'beijing': 0.0001, 'guangzhou': 0.0001, 'shenyang': 0.0001,
             'BeijingAQ_Aoti': 0.0001, 'BeijingAQ_Wanliu': 0.0001, 'BeijingAQ_Daxing': 0.0001, 
             'BeijingAQ_Tiantan': 0.0001, 'BeijingAQ_Guanyuan': 0.0001, 'Mooloolaba_Waves': 0.0001},
    # Extended entries from lhy
    'DLinear': {'electricity': 0.0001, 'NYC_Taxi': 0.0001, 'stock': 0.0001, 'wind': 0.0001,
                'CHI_Crime': 0.0001, 'saugeen_river': 0.0001, 'Mooloolaba_Waves': 0.0001,
                'beijing': 0.0001, 'guangzhou': 0.0001, 'shenyang': 0.0001},
    'TimeMixer': {'electricity': 0.0001, 'NYC_Taxi': 0.0001, 'stock': 0.0001, 'wind': 0.0001,
                  'CHI_Crime': 0.0001, 'saugeen_river': 0.0001, 'Mooloolaba_Waves': 0.0001,
                  'beijing': 0.0001, 'guangzhou': 0.0001, 'shenyang': 0.0001},
    'TimesNet': {'electricity': 0.0001, 'NYC_Taxi': 0.0001, 'stock': 0.0001, 'wind': 0.0001,
                 'CHI_Crime': 0.0001, 'saugeen_river': 0.0001, 'Mooloolaba_Waves': 0.0001,
                 'beijing': 0.0001, 'guangzhou': 0.0001, 'shenyang': 0.0001},
    # Extended entries from cq
    'Leddam': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001,
               'stock': 0.0001, 'wind': 0.0001, 'NYC_Taxi': 0.0001, 'CHI_Crime': 0.0001, 
               'saugeen_river': 0.0001, 'beijing': 0.0001, 'guangzhou': 0.0001, 'shenyang': 0.0001,
               'BeijingAQ_Aoti': 0.0001, 'BeijingAQ_Wanliu': 0.0001, 'BeijingAQ_Daxing': 0.0001, 
               'BeijingAQ_Tiantan': 0.0001, 'BeijingAQ_Guanyuan': 0.0001, 'Mooloolaba_Waves': 0.0001},
    'MICN': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001,
             'stock': 0.0001, 'wind': 0.0001, 'NYC_Taxi': 0.0001, 'CHI_Crime': 0.0001, 
             'saugeen_river': 0.0001, 'beijing': 0.0001, 'guangzhou': 0.0001, 'shenyang': 0.0001,
             'BeijingAQ_Aoti': 0.0001, 'BeijingAQ_Wanliu': 0.0001, 'BeijingAQ_Daxing': 0.0001, 
             'BeijingAQ_Tiantan': 0.0001, 'BeijingAQ_Guanyuan': 0.0001, 'Mooloolaba_Waves': 0.0001},
    'SCINet': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001,
               'stock': 0.0001, 'wind': 0.0001, 'NYC_Taxi': 0.0001, 'CHI_Crime': 0.0001, 
               'saugeen_river': 0.0001, 'beijing': 0.0001, 'guangzhou': 0.0001, 'shenyang': 0.0001,
               'BeijingAQ_Aoti': 0.0001, 'BeijingAQ_Wanliu': 0.0001, 'BeijingAQ_Daxing': 0.0001, 
               'BeijingAQ_Tiantan': 0.0001, 'BeijingAQ_Guanyuan': 0.0001, 'Mooloolaba_Waves': 0.0001},
    'OLinear': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001,
                'stock': 0.0001, 'wind': 0.0001, 'NYC_Taxi': 0.0001, 'CHI_Crime': 0.0001, 
                'saugeen_river': 0.0001, 'beijing': 0.0001, 'guangzhou': 0.0001, 'shenyang': 0.0001,
                'BeijingAQ_Aoti': 0.0001, 'BeijingAQ_Wanliu': 0.0001, 'BeijingAQ_Daxing': 0.0001, 
                'BeijingAQ_Tiantan': 0.0001, 'BeijingAQ_Guanyuan': 0.0001, 'Mooloolaba_Waves': 0.0001},
    'BiMamba4TS': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001,
                   'stock': 0.0001, 'wind': 0.0001, 'NYC_Taxi': 0.0001, 'CHI_Crime': 0.0001, 
                   'saugeen_river': 0.0001, 'beijing': 0.0001, 'guangzhou': 0.0001, 'shenyang': 0.0001,
                   'BeijingAQ_Aoti': 0.0001, 'BeijingAQ_Wanliu': 0.0001, 'BeijingAQ_Daxing': 0.0001, 
                   'BeijingAQ_Tiantan': 0.0001, 'BeijingAQ_Guanyuan': 0.0001, 'Mooloolaba_Waves': 0.0001},
}

# Merged pretrain_lr_dict from lhy and cq
pretrain_lr_dict = {
    'PatchTST': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001,
                 'stock': 0.0001, 'wind': 0.0001, 'NYC_Taxi': 0.0001, 'CHI_Crime': 0.0001, 
                 'saugeen_river': 0.0001, 'beijing': 0.0001, 'guangzhou': 0.0001, 'shenyang': 0.0001,
                 'BeijingAQ_Aoti': 0.0001, 'BeijingAQ_Wanliu': 0.0001, 'BeijingAQ_Daxing': 0.0001, 
                 'BeijingAQ_Tiantan': 0.0001, 'BeijingAQ_Guanyuan': 0.0001, 'Mooloolaba_Waves': 0.0001},
    'iTransformer': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.001, 'Weather': 0.0001, 'ECL': 0.0005},
    'TimeBridge': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001},
    'duet': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001,
             'stock': 0.0001, 'wind': 0.0001, 'NYC_Taxi': 0.0001, 'CHI_Crime': 0.0001, 
             'saugeen_river': 0.0001, 'beijing': 0.0001, 'guangzhou': 0.0001, 'shenyang': 0.0001,
             'BeijingAQ_Aoti': 0.0001, 'BeijingAQ_Wanliu': 0.0001, 'BeijingAQ_Daxing': 0.0001, 
             'BeijingAQ_Tiantan': 0.0001, 'BeijingAQ_Guanyuan': 0.0001, 'Mooloolaba_Waves': 0.0001},
    'Leddam': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001,
               'stock': 0.0001, 'wind': 0.0001, 'NYC_Taxi': 0.0001, 'CHI_Crime': 0.0001, 
               'saugeen_river': 0.0001, 'beijing': 0.0001, 'guangzhou': 0.0001, 'shenyang': 0.0001,
               'BeijingAQ_Aoti': 0.0001, 'BeijingAQ_Wanliu': 0.0001, 'BeijingAQ_Daxing': 0.0001, 
               'BeijingAQ_Tiantan': 0.0001, 'BeijingAQ_Guanyuan': 0.0001, 'Mooloolaba_Waves': 0.0001},
    'MICN': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001,
             'stock': 0.0001, 'wind': 0.0001, 'NYC_Taxi': 0.0001, 'CHI_Crime': 0.0001, 
             'saugeen_river': 0.0001, 'beijing': 0.0001, 'guangzhou': 0.0001, 'shenyang': 0.0001,
             'BeijingAQ_Aoti': 0.0001, 'BeijingAQ_Wanliu': 0.0001, 'BeijingAQ_Daxing': 0.0001, 
             'BeijingAQ_Tiantan': 0.0001, 'BeijingAQ_Guanyuan': 0.0001, 'Mooloolaba_Waves': 0.0001},
    'SCINet': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001,
               'stock': 0.0001, 'wind': 0.0001, 'NYC_Taxi': 0.0001, 'CHI_Crime': 0.0001, 
               'saugeen_river': 0.0001, 'beijing': 0.0001, 'guangzhou': 0.0001, 'shenyang': 0.0001,
               'BeijingAQ_Aoti': 0.0001, 'BeijingAQ_Wanliu': 0.0001, 'BeijingAQ_Daxing': 0.0001, 
               'BeijingAQ_Tiantan': 0.0001, 'BeijingAQ_Guanyuan': 0.0001, 'Mooloolaba_Waves': 0.0001},
    'OLinear': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001,
                'stock': 0.0001, 'wind': 0.0001, 'NYC_Taxi': 0.0001, 'CHI_Crime': 0.0001, 
                'saugeen_river': 0.0001, 'beijing': 0.0001, 'guangzhou': 0.0001, 'shenyang': 0.0001,
                'BeijingAQ_Aoti': 0.0001, 'BeijingAQ_Wanliu': 0.0001, 'BeijingAQ_Daxing': 0.0001, 
                'BeijingAQ_Tiantan': 0.0001, 'BeijingAQ_Guanyuan': 0.0001, 'Mooloolaba_Waves': 0.0001},
    'BiMamba4TS': {'ETTh2': 0.0001, 'ETTm1': 0.0001, 'Traffic': 0.0001, 'Weather': 0.0001, 'ECL': 0.0001,
                   'stock': 0.0001, 'wind': 0.0001, 'NYC_Taxi': 0.0001, 'CHI_Crime': 0.0001, 
                   'saugeen_river': 0.0001, 'beijing': 0.0001, 'guangzhou': 0.0001, 'shenyang': 0.0001,
                   'BeijingAQ_Aoti': 0.0001, 'BeijingAQ_Wanliu': 0.0001, 'BeijingAQ_Daxing': 0.0001, 
                   'BeijingAQ_Tiantan': 0.0001, 'BeijingAQ_Guanyuan': 0.0001, 'Mooloolaba_Waves': 0.0001},
}


def drop_last_PatchTST(args):
    bs = 128 if args.dataset in ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', 'Weather'] else 32
    test_num = args.borders[1][2] - args.borders[0][2] - args.seq_len - args.pred_len + 1
    args.borders[1][2] -= test_num % bs
