# DSTF-Bench: A Shift-Aware Benchmark for Time Series Forecasting in Non-Stationary Environments

This repository provides the official PyTorch implementation of **DSTF-Bench**, a shift-aware benchmark for evaluating time series forecasting models under non-stationary environments.

DSTF-Bench is designed to diagnose not only whether forecasting performance changes under distribution shift, but also which type of shift is responsible. The benchmark follows a moment-centric taxonomy covering:

- **First-order intra-series shifts**: changes in local mean or baseline level.
- **Second-order intra-series shifts**: changes in variance, volatility, or local fluctuation scale.
- **Inter-series dependency shifts**: changes in cross-variable covariance and dependency structure.

The benchmark includes 15 public real-world datasets, unified offline and online evaluation protocols, representative forecasting backbones, shift-resilient frameworks, online adaptation methods, and controlled shift pressure tests.

## News

- **2026-06**: Initial public release of the DSTF-Bench codebase.
- Datasets are distributed separately at: https://github.com/xhhmacau/Datasets

## Requirements

Python 3.9 or 3.10 is recommended.

```bash
conda create -n dstf python=3.10 -y
conda activate dstf
pip install -r requirements.txt
```

GPU execution is recommended for all experiments. Select visible devices before launching Python:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py ...
```

If multiple GPUs are exposed, `--gpu` indexes the visible devices. For example, with `CUDA_VISIBLE_DEVICES=2,3`, `--gpu 0` uses physical GPU 2 and `--gpu 1` uses physical GPU 3.

## Data

DSTF-Bench datasets are released separately to keep this repository lightweight:

```text
https://github.com/xhhmacau/Datasets
```

Download the required files and place them under `./dataset/`. The default loader uses relative paths and expects dataset files to match the names configured in `settings.py`.

Example layout:

```text
dataset/
|-- stock.csv
|-- electricity.csv
|-- wind.csv
|-- NYC_Taxi.csv
|-- CHI_Crime_Cleaned_Hourly.csv
|-- mooloolaba_waves_complete_records.csv
|-- saugeen_river.csv
|-- BeijingAQ/
|   |-- Aotizhongxin_2015_2025_cleaned.csv
|   |-- Daxing_2015_2025_cleaned.csv
|   |-- Guanyuan_2015_2025_cleaned.csv
|   |-- Tiantan_2015_2025_cleaned.csv
|   `-- Wanliu_2015_2025_cleaned.csv
|-- beijing.csv
|-- guangzhou.csv
`-- shenyang.csv
```

Each CSV should contain a timestamp column named `date` followed by numerical variables. Dataset-specific target columns, dimensionalities, and frequencies are defined in `settings.py`.

## Benchmark Datasets

DSTF-Bench covers seven application domains and 15 datasets.

| Dataset key | Domain | Frequency | Variables | Target |
| --- | --- | --- | ---: | --- |
| `stock` | Economy | Daily | 8 | `DailyPrice` |
| `NYC_Taxi` | Traffic | 30 minutes | 1 | `Value` |
| `electricity` | Energy | 1 minute | 7 | `Sub_metering_3` |
| `wind` | Energy | 15 minutes | 1 | `windspeed` |
| `BeijingAQ_Aoti` | Air quality | Hourly | 7 | `AQI` |
| `BeijingAQ_Wanliu` | Air quality | Hourly | 7 | `AQI` |
| `BeijingAQ_Daxing` | Air quality | Hourly | 7 | `AQI` |
| `BeijingAQ_Tiantan` | Air quality | Hourly | 7 | `AQI` |
| `BeijingAQ_Guanyuan` | Air quality | Hourly | 7 | `AQI` |
| `beijing` | Agriculture/environment | Hourly | 2 | `shidu` |
| `guangzhou` | Agriculture/environment | Hourly | 2 | `shidu` |
| `shenyang` | Agriculture/environment | Hourly | 2 | `shidu` |
| `CHI_Crime` | Public safety | Hourly | 4 | `THEFT` |
| `Mooloolaba_Waves` | Ocean waves | 30 minutes | 6 | `Hs` |
| `saugeen_river` | Hydrology | Daily | 1 | `flow` |

## Benchmark Protocols

DSTF-Bench evaluates models under two complementary protocols.

| Protocol | Split | Look-back length | Prediction horizons | Evaluation behavior |
| --- | --- | ---: | --- | --- |
| Offline forecasting | 7:1:2 for most datasets; 4:2:4 for Stock | 96 | 96, 192, 336, 720 | Train once, keep parameters fixed during testing |
| Online forecasting | Chronological stream with train/validation/test periods | 96 | 24, 48, 96 | Process samples causally and update only after delayed labels become available |

The offline protocol measures robustness of a fixed trained model under future non-stationarity. The online protocol measures adaptation under temporally ordered streams without future-label leakage. Offline and online numbers should be compared within the same protocol, not directly across protocols, because they use different horizons and update opportunities.

## Supported Methods

### Main Forecasting Backbones

| Family | Models |
| --- | --- |
| Linear | `OLinear`, `DLinear`, `NLinear` |
| Transformer and attention | `Informer`, `Autoformer`, `PatchTST`, `Crossformer`, `DeformableTST` |
| Convolutional and hierarchical | `TCN`, `SCINet`, `MICN` |
| Decomposition, mixing, and non-stationarity-aware | `Leddam`, `duet`, `TimeBridge`, `TimeMixer`, `TimesNet` |
| State space | `S_Mamba` |

### Online Adaptation

| Method | Argument |
| --- | --- |
| Base online update | omit `--online_method` |
| SOLID | `--online_method SOLID` |
| PROCEED | `--online_method Proceed` |
| AdaptZ delayed-feedback ablation | `--online_method AdaptZ` |

The `AdaptZ` class in this repository implements a causal delayed-feedback ablation path in the unified online pipeline. It does not vendor the official ADAPT-Z z-adapter and feature-gradient predictor modules.

## Usage

### Offline Forecasting

Train and evaluate a fixed forecasting model:

```bash
CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --learning_environment offline \
  --model DLinear \
  --dataset stock \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --itr 1 \
  --train_epochs 10 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --gpu 0
```

### Online Forecasting

Run causal online updating:

```bash
CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --learning_environment online \
  --model NLinear \
  --dataset stock \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --itr 1 \
  --train_epochs 1 \
  --batch_size 32 \
  --online_learning_rate 0.0001 \
  --gpu 0
```

### PROCEED

```bash
CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --learning_environment online \
  --model NLinear \
  --dataset stock \
  --online_method Proceed \
  --online_learning_rate 0.0001 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --batch_size 32 \
  --gpu 0
```

### SOLID

```bash
CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --learning_environment online \
  --model NLinear \
  --dataset stock \
  --online_method SOLID \
  --online_learning_rate 0.0001 \
  --whole_model \
  --test_train_num 20 \
  --selected_data_num 5 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --batch_size 32 \
  --gpu 0
```

## Reproducibility

To reproduce the main benchmark, run all backbones over the offline and online horizons used in the paper.

Offline example:

```bash
for model in OLinear DLinear NLinear Informer Autoformer PatchTST Crossformer DeformableTST TCN SCINet MICN Leddam duet TimeBridge TimeMixer TimesNet S_Mamba; do
  for pred_len in 96 192 336 720; do
    CUDA_VISIBLE_DEVICES=0 python -u run.py \
      --learning_environment offline \
      --model "$model" \
      --dataset stock \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len "$pred_len" \
      --itr 3 \
      --gpu 0
  done
done
```

Online example:

```bash
for pred_len in 24 48 96; do
  CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --learning_environment online \
    --model NLinear \
    --dataset stock \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len "$pred_len" \
    --itr 3 \
    --online_learning_rate 0.0001 \
    --gpu 0
done
```

## Important Arguments

| Argument | Description |
| --- | --- |
| `--learning_environment` | `offline` for fixed-parameter forecasting, `online` for causal stream adaptation |
| `--model` | Forecasting backbone name |
| `--dataset` | Dataset key defined in `settings.py` |
| `--root_path` | Dataset root path, default `./dataset/` |
| `--features` | Forecasting task: `M`, `S`, or `MS` |
| `--target` | Target column for `S` or `MS` tasks |
| `--seq_len` | Look-back length |
| `--label_len` | Decoder warm-up length for encoder-decoder models |
| `--pred_len` | Forecasting horizon |
| `--online_method` | Online adaptation method |
| `--normalization` | Optional normalization wrapper, such as `revin` or `san` |
| `--itr` | Number of repeated runs |
| `--checkpoints` | Checkpoint directory, default `./checkpoints/` |
| `--gpu` | Index of the visible GPU to use |

## Outputs

Experiment outputs are written to:

```text
checkpoints/   # saved model checkpoints
results/       # CSV metric summaries
```

Both directories are ignored by git.

## Repository Structure

```text
DSTF-Bench/
|-- run.py
|-- settings.py
|-- requirements.txt
|-- dataset/
|-- data_provider/
|   |-- offline/
|   `-- online/
|-- exp/
|   |-- offline_exp/
|   `-- online_exp/
|-- models/
|-- layers/
|-- adapter/
`-- utils/
```

## Verification

The release code was smoke-tested on Stock with GPU execution. The following code paths were checked:

- Offline: `OLinear`, `DLinear`, `NLinear`, `Informer`, `Autoformer`, `PatchTST`, `Crossformer`, `DeformableTST`, `TCN`, `SCINet`, `MICN`, `Leddam`, `duet`, `TimeBridge`, `TimeMixer`, `TimesNet`, `S_Mamba`.
- Online: base `Exp_Online`, `SOLID`, `Proceed`, and `AdaptZ`.

Smoke tests use short one-epoch runs to verify executability. Full benchmark numbers require the paper-scale settings and repeated seeds.

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{xu2026dstfbench,
  title     = {DSTF-Bench: A Shift-Aware Benchmark for Time Series Forecasting in Non-Stationary Environments},
  author    = {Xu, Haihua},
  booktitle = {Proceedings of the ACM Conference},
  year      = {2026}
}
```

Please update the venue, DOI, and bibliographic metadata after publication.

## Acknowledgement

This repository builds on the open-source time series forecasting ecosystem. We thank the authors of the forecasting models and benchmark repositories that made reproducible TSF research possible.
