# Usage Examples

## Basic Workflow

### 1. Train Linear Probe on Flowers102
```bash
python train.py
# Uses defaults: model=linear_probe, dataset=flowers102
# Output: outputs/linear_probe/flowers102/
```

### 2. Evaluate the Model
```bash
python evaluate.py
# Loads checkpoint from outputs/linear_probe/flowers102/model_checkpoint.pt
# Saves results to outputs/linear_probe/flowers102/evaluation/
```

### 3. Visualize Results
```bash
python analyze.py
# Generates plots in outputs/linear_probe/flowers102/
# - training_curves.png
# - evaluation/confusion_matrix.png
```

## Advanced Examples

### Train LoRA with Custom Settings
```bash
# Train with higher LoRA rank
python train.py model=lora model.lora_r=16 model.lora_alpha=32

# Train for more epochs
python train.py model=lora training.num_epochs=10

# Use smaller batch size with gradient accumulation
python train.py model=lora training.batch_size=32 training.gradient_accumulation_steps=4
```

### CUB-200-2011 Dataset
```bash
# Train both models on CUB
python train.py dataset=cub200
python train.py model=lora dataset=cub200

# Evaluate
python evaluate.py dataset=cub200
python evaluate.py model=lora dataset=cub200

# Analyze
python analyze.py dataset=cub200
python analyze.py model=lora dataset=cub200
```

### Compare Multiple Models

**Option 1: Edit `configs/analysis.yaml`**
```yaml
compare:
  - output_dir: outputs/linear_probe/flowers102
    name: "Linear Probe"
  - output_dir: outputs/lora/flowers102
    name: "LoRA"
  - output_dir: outputs/linear_probe/cub200
    name: "Linear Probe (CUB)"
  - output_dir: outputs/lora/cub200
    name: "LoRA (CUB)"

comparison_output_dir: outputs/comparison/all_models
```

Then run:
```bash
python analyze.py --config-name analysis
```

**Option 2: Override from command line**
```bash
python analyze.py --config-name analysis \
  'compare=[{output_dir: outputs/linear_probe/flowers102, name: "LP"}, {output_dir: outputs/lora/flowers102, name: "LoRA"}]'
```

### Multi-GPU Training
```bash
# Use specific GPU
python train.py device=cuda:0
python train.py device=cuda:1 model=lora

# Or set environment variable
CUDA_VISIBLE_DEVICES=2 python train.py
```

### Experiment Tracking with Weights & Biases
```bash
# Enable wandb
python train.py use_wandb=true wandb.project=my-clip-experiments

# With custom run name
python train.py use_wandb=true wandb.project=my-clip-experiments \
  wandb.name="lora-r16-flowers102" model=lora model.lora_r=16
```

### Save Different Experiment Runs
```bash
# Custom output directories
python train.py output_dir=outputs/experiments/exp1
python train.py output_dir=outputs/experiments/exp2 model.learning_rate=5e-4

# Then analyze
python analyze.py output_dir=outputs/experiments/exp1
```

## Hyperparameter Sweeps

### Learning Rate Sweep
```bash
for lr in 1e-4 5e-4 1e-3 5e-3; do
  python train.py model.learning_rate=$lr \
    output_dir=outputs/lr_sweep/lr_${lr}
done

# Analyze all
for lr in 1e-4 5e-4 1e-3 5e-3; do
  python analyze.py output_dir=outputs/lr_sweep/lr_${lr}
done
```

### LoRA Rank Sweep
```bash
for rank in 4 8 16 32; do
  python train.py model=lora model.lora_r=$rank \
    output_dir=outputs/lora_sweep/rank_${rank}
done
```

### Batch Size Sweep
```bash
for bs in 32 64 128 256; do
  python train.py training.batch_size=$bs \
    output_dir=outputs/batch_sweep/bs_${bs}
done
```

## Quick Testing

### Single Epoch Quick Test
```bash
# Fast sanity check
python train.py training.num_epochs=1 training.batch_size=256
python evaluate.py
python analyze.py
```

### Debug Mode (Small Subset)
Create `configs/debug.yaml`:
```yaml
defaults:
  - config

training:
  num_epochs: 1
  batch_size: 8

# Note: You'd need to add dataset subsetting logic for true debug mode
```

Then:
```bash
python train.py --config-name debug
```

## Production Run

### Full Training Pipeline
```bash
#!/bin/bash
# train_all.sh

DATASETS="flowers102 cub200"
MODELS="linear_probe lora"

for dataset in $DATASETS; do
  for model in $MODELS; do
    echo "Training $model on $dataset..."
    python train.py model=$model dataset=$dataset

    echo "Evaluating $model on $dataset..."
    python evaluate.py model=$model dataset=$dataset

    echo "Analyzing $model on $dataset..."
    python analyze.py model=$model dataset=$dataset
  done
done

echo "Generating comparison plots..."
python analyze.py --config-name analysis
```

Run:
```bash
chmod +x train_all.sh
./train_all.sh
```

## Resuming / Loading Checkpoints

### Evaluate Pre-trained Model
```bash
# Train
python train.py model=lora dataset=flowers102

# Later, evaluate the saved checkpoint
python evaluate.py model=lora dataset=flowers102
# Automatically loads from outputs/lora/flowers102/model_checkpoint.pt
```

### Use Checkpoint in Custom Script
```python
from src.models import LoRAModel
import torch

# Load model
model = LoRAModel(
    model_id="openai/clip-vit-large-patch14",
    num_classes=102,
    device="cuda"
)
model.load_checkpoint("outputs/lora/flowers102/model_checkpoint.pt")
model.eval()

# Use for inference
# ... your code here ...
```

## Tips & Tricks

### View Config Before Running
```bash
# See what config will be used
python train.py --cfg job
```

### Override Multiple Settings
```bash
python train.py \
  model=lora \
  dataset=cub200 \
  training.num_epochs=5 \
  training.batch_size=64 \
  model.lora_r=16 \
  device=cuda:1 \
  output_dir=outputs/custom_exp
```

### Check Output Structure
```bash
tree outputs/linear_probe/flowers102/
# outputs/linear_probe/flowers102/
# ├── config.yaml
# ├── training_metrics.json
# ├── training_curves.png
# ├── model_checkpoint.pt
# └── evaluation/
#     ├── evaluation_results.json
#     ├── predictions.npz
#     └── confusion_matrix.png
```
