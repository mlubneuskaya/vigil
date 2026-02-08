# Auto-QA Pipeline: Quick Start Guide

## 1. Python Environment Setup

1.1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

1.2. Install required libraries:
```bash
pip install -r requirements.txt
```

## 2. Download Data from HuggingFace

2.1. Run the HuggingFace data download script:
```bash
python3 scripts/download_hf_data.py
```

- This will download datasets and place them in the `data/` directory.
- You may need to set up your HuggingFace token if prompted.

## 3. YAML Config Creation

Since the `configs/` and `jobs/` folders are git-ignored, you must create your own experiment config file. Here is a sample config:

Create a new file at `configs/example_experiment.yaml` with the following content:

```yaml
output_dir: outputs/example_run/

data_root: []
data_ids:
  "data/consumer_goods/outputs":
    - "0001"
    - "0002"
  
  "data/furniture/outputs":
    - "0001"
    - "0002"

qwen:
  model_name: Qwen/Qwen3-VL-8B-Instruct

segmentation:
  pairing_threshold: 0.3

background:
  device: cuda
  vlm_resolution: 1024
  batch_size: 16
  mask_margin_percent: 0.0
  absolute_difference_config:
    enabled: true

object_eval:
  resolution: 1024

```

- Adjust parameters as needed for your data and experiment.
- For full parameter descriptions, see [`src/configs/config_schema.py`](src/configs/config_schema.py).

## 4. Running the Pipeline

4.1. Run your experiment:

```bash
python3 main.py --config configs/example_experiment.yaml
```

## 5. Output and Logs

- Results are saved in the `outputs/` directory.
- Logs are in `outputs/logs/`.
