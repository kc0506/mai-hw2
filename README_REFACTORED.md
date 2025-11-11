# CLIP Fine-tuning: Modular Implementation

A clean, modular implementation of CLIP fine-tuning with Linear Probing and LoRA, using Hydra for configuration management.

## Project Structure

```
.
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main config
│   ├── dataset/               # Dataset configurations
│   │   ├── flowers102.yaml
│   │   └── cub200.yaml
│   └── model/                 # Model configurations
│       ├── linear_probe.yaml
│       └── lora.yaml
├── src/                       # Source code modules
│   ├── data/                  # Data loading and preprocessing
│   │   ├── datasets.py
│   │   └── transforms.py
│   ├── models/                # Model implementations
│   │   ├── linear_probe.py
│   │   └── lora_model.py
│   ├── training/              # Training utilities
│   │   └── trainer.py
│   ├── evaluation/            # Evaluation utilities
│   │   └── evaluator.py
│   └── utils/                 # Visualization and helpers
│       └── visualization.py
├── train.py                   # Training CLI
├── evaluate.py                # Evaluation CLI
├── analyze.py                 # Analysis/visualization CLI
└── outputs/                   # Output directory (auto-created)
```


## Quick Start

### Using Just (Recommended)

[Just](https://github.com/casey/just) is a command runner that simplifies running the pipeline. Install it with: `cargo install just` or your package manager.

```bash
# Run full pipeline for linear probe on Flowers102
just pipeline-linear-flowers

# Run full pipeline for LoRA on CUB-200
just pipeline-lora-cub

# Run all experiments on Flowers102 (both models + comparison)
just flowers-all

# Run EVERYTHING
just all

# See all available commands
just --list
```

### Manual Commands (Without Just)

#### 1. Train a Linear Probe Model

```bash
# Train on Flowers102 with default settings
python train.py

# Train on CUB-200-2011
python train.py dataset=cub200

# Train on specific GPU
python train.py device=cuda:0
```

#### 2. Train a LoRA Model

```bash
# Train LoRA on Flowers102
python train.py model=lora

# Train LoRA on CUB-200-2011 with custom hyperparameters
python train.py model=lora dataset=cub200 model.lora_r=16
```

#### 3. Evaluate a Trained Model

```bash
# Evaluate linear probe
python evaluate.py

# Evaluate LoRA model
python evaluate.py model=lora
```

#### 4. Analyze and Visualize Results

```bash
# Analyze single model (auto-detects output dir from config)
python analyze.py model=linear_probe dataset=flowers102

# Analyze LoRA model
python analyze.py model=lora dataset=flowers102

# Compare models using analysis config
python analyze.py --config-name analysis
```

## Configuration

### Hydra Configuration System

This project uses Hydra for flexible configuration management. All three CLI scripts (train, evaluate, analyze) use Hydra configs.

**Config Files:**
- `configs/config.yaml` - Main configuration
- `configs/dataset/*.yaml` - Dataset-specific configs
- `configs/model/*.yaml` - Model-specific configs
- `configs/analysis.yaml` - Analysis/comparison config

### Key Configuration Options

**Training Settings** (`configs/config.yaml`):
```yaml
training:
  num_epochs: 1
  batch_size: 128
  learning_rate: 1e-3
  gradient_accumulation_steps: 1
```

**Model Settings** (`configs/model/`):
- `linear_probe.yaml`: Learning rate for linear probe
- `lora.yaml`: LoRA rank, alpha, dropout, learning rate

**Dataset Settings** (`configs/dataset/`):
- `flowers102.yaml`: Root directory, download option
- `cub200.yaml`: Validation split ratio

**Analysis Settings** (`configs/analysis.yaml`):
- Configure model comparisons
- Specify output directories and display names

## Output Structure

After training and evaluation, outputs are organized as:

```
outputs/
└── {model_name}/
    └── {dataset_name}/
        ├── config.yaml                  # Training config
        ├── training_metrics.json        # Training history
        ├── training_curves.png          # Loss/accuracy plots
        ├── model_checkpoint.pt          # Model weights
        ├── lora_adapters/               # LoRA adapters (LoRA only)
        └── evaluation/
            ├── evaluation_results.json  # Test metrics
            ├── predictions.npz          # Predictions array
            └── confusion_matrix.png     # Confusion matrix plot
```

## Advanced Usage

### Using Weights & Biases

Enable wandb logging:

```bash
# Edit configs/config.yaml
use_wandb: true
wandb:
  project: "my-clip-project"
  entity: "my-username"

# Or override from CLI
python train.py use_wandb=true wandb.project=my-clip-project
```

### Custom Hyperparameter Sweeps

```bash
# Try different learning rates
python train.py model.learning_rate=1e-4
python train.py model.learning_rate=5e-4
python train.py model.learning_rate=1e-3

# Try different LoRA ranks
python train.py model=lora model.lora_r=4
python train.py model=lora model.lora_r=8
python train.py model=lora model.lora_r=16
```

### Multi-GPU Training

```bash
# Specify GPU
python train.py device=cuda:0

# Or set in environment
CUDA_VISIBLE_DEVICES=2 python train.py
```

## Example Workflows

### Using Justfile (Recommended)

```bash
# Run complete pipeline for specific dataset/model combination
just pipeline-linear-flowers    # Linear probe on Flowers102
just pipeline-lora-cub          # LoRA on CUB-200

# Run all experiments on one dataset
just flowers-all                # All models on Flowers102
just cub-all                    # All models on CUB-200

# Run all experiments for one method
just linear-all                 # Linear probe on both datasets
just lora-all                   # LoRA on both datasets

# Run everything and compare
just all

# Quick test run
just test

# Check environment
just check

# View results summary
just results
```

### Manual Workflow (Without Just)

Complete workflow for both models on both datasets:

```bash
# 1. Train all combinations
python train.py model=linear_probe dataset=flowers102
python train.py model=linear_probe dataset=cub200
python train.py model=lora dataset=flowers102
python train.py model=lora dataset=cub200

# 2. Evaluate all models
python evaluate.py model=linear_probe dataset=flowers102
python evaluate.py model=linear_probe dataset=cub200
python evaluate.py model=lora dataset=flowers102
python evaluate.py model=lora dataset=cub200

# 3. Analyze individual models
python analyze.py model=linear_probe dataset=flowers102
python analyze.py model=lora dataset=flowers102

# 4. Compare models (edit configs/analysis.yaml first)
python analyze.py --config-name analysis
```

### Available Just Commands

```bash
just --list                     # Show all available commands

# Individual pipelines
just pipeline-linear-flowers    # Train -> Eval -> Analyze
just pipeline-linear-cub
just pipeline-lora-flowers
just pipeline-lora-cub

# Batch operations
just train-all                  # Train all models
just eval-all                   # Evaluate all models
just analyze-all                # Analyze all results
just compare-all                # Generate comparison plots

# By dataset
just flowers-all                # All experiments on Flowers102
just cub-all                    # All experiments on CUB-200

# By method
just linear-all                 # All linear probe experiments
just lora-all                   # All LoRA experiments

# Utilities
just clean                      # Remove outputs
just fresh                      # Clean + run all
just check                      # Check environment setup
just results                    # Show results summary
just test                       # Quick test run
```

## Extending the Framework

### Adding a New Dataset

1. Create config file `configs/dataset/my_dataset.yaml`
2. Add loader function in `src/data/datasets.py`
3. Update `get_dataset()` factory function

### Adding a New Model

1. Create model class in `src/models/my_model.py`
2. Create config file `configs/model/my_model.yaml`
3. Update imports in `src/models/__init__.py`
4. Add model instantiation in `train.py` and `evaluate.py`

## Troubleshooting

**CUDA out of memory**: Reduce `training.batch_size` or increase `gradient_accumulation_steps`

```bash
python train.py training.batch_size=64 training.gradient_accumulation_steps=2
```

**LoRA error with input_ids**: Fixed - removed `task_type` from LoRA config for vision models

**Import errors**: Make sure to run from project root directory

## License

MIT
