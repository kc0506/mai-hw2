# Justfile for CLIP Fine-tuning Pipeline
# Usage: just <recipe_name>

# Default recipe - show help
default:
    @just --list

# ==================== Individual Commands ====================

# Train linear probe on Flowers102
train-linear-flowers:
    python train.py model=linear_probe dataset=flowers102 training.batch_size=256 device=cuda:1

# Train linear probe on Flowers102 (robust mode with auto device selection)
train-linear-flowers-robust:
    python train_robust.py model=linear_probe dataset=flowers102

# Train linear probe on CUB-200
train-linear-cub:
    python train.py model=linear_probe dataset=cub200

# Train LoRA on Flowers102
# train-lora-flowers:
#     python train.py model=lora dataset=flowers102 training.batch_size=32
train-lora-flowers *ARGS:                                                                                                                                     
    python train.py model=lora dataset=flowers102 {{ ARGS }}

# Train LoRA on Flowers102 (robust mode)
train-lora-flowers-robust:
    python train_robust.py model=lora dataset=flowers102

# Train LoRA on CUB-200
train-lora-cub:
    python train.py model=lora dataset=cub200

# Evaluate linear probe on Flowers102
eval-linear-flowers:
    python evaluate.py model=linear_probe dataset=flowers102

# Evaluate linear probe on CUB-200
eval-linear-cub:
    python evaluate.py model=linear_probe dataset=cub200

# Evaluate LoRA on Flowers102
eval-lora-flowers:
    python evaluate.py model=lora dataset=flowers102

# Evaluate LoRA on CUB-200
eval-lora-cub:
    python evaluate.py model=lora dataset=cub200

# Analyze linear probe on Flowers102
analyze-linear-flowers:
    python analyze.py model=linear_probe dataset=flowers102

# Analyze linear probe on CUB-200
analyze-linear-cub:
    python analyze.py model=linear_probe dataset=cub200

# Analyze LoRA on Flowers102
analyze-lora-flowers:
    python analyze.py model=lora dataset=flowers102

# Analyze LoRA on CUB-200
analyze-lora-cub:
    python analyze.py model=lora dataset=cub200

# ==================== Full Pipelines ====================

# Run full pipeline (train -> eval -> analyze) for linear probe on Flowers102
pipeline-linear-flowers: train-linear-flowers eval-linear-flowers analyze-linear-flowers
    @echo "Completed linear probe pipeline on Flowers102"

# Run full pipeline for linear probe on CUB-200
pipeline-linear-cub: train-linear-cub eval-linear-cub analyze-linear-cub
    @echo "Completed linear probe pipeline on CUB-200"

# Run full pipeline for LoRA on Flowers102
pipeline-lora-flowers: train-lora-flowers eval-lora-flowers analyze-lora-flowers
    @echo "Completed LoRA pipeline on Flowers102"

# Run full pipeline for LoRA on CUB-200
pipeline-lora-cub: train-lora-cub eval-lora-cub analyze-lora-cub
    @echo "Completed LoRA pipeline on CUB-200"

# ==================== Batch Operations ====================

# Train all models on all datasets
train-all: train-linear-flowers train-linear-cub train-lora-flowers train-lora-cub
    @echo "Completed training all models"

# Evaluate all models on all datasets
eval-all: eval-linear-flowers eval-linear-cub eval-lora-flowers eval-lora-cub
    @echo "Completed evaluating all models"

# Analyze all models on all datasets
analyze-all: analyze-linear-flowers analyze-linear-cub analyze-lora-flowers analyze-lora-cub
    @echo "Completed analyzing all models"

# Compare models on Flowers102
compare-flowers:
    python analyze.py --config-name analysis

# Compare all models
compare-all: compare-flowers
    @echo "Completed all comparisons"

# ==================== Complete Workflows ====================

# Run all pipelines for Flowers102 (both models)
flowers-all: pipeline-linear-flowers pipeline-lora-flowers compare-flowers
    @echo "Completed all experiments on Flowers102"

# Run all pipelines for CUB-200 (both models)
cub-all: pipeline-linear-cub pipeline-lora-cub
    @echo "Completed all experiments on CUB-200"

# Run all linear probe experiments (both datasets)
linear-all: pipeline-linear-flowers pipeline-linear-cub
    @echo "Completed all linear probe experiments"

# Run all LoRA experiments (both datasets)
lora-all: pipeline-lora-flowers pipeline-lora-cub
    @echo "Completed all LoRA experiments"

# Run EVERYTHING (train, eval, analyze, compare)
all: train-all eval-all analyze-all compare-all
    @echo "Completed full experimental pipeline!"

# ==================== Utilities ====================

# Clean output directories
clean:
    rm -rf outputs/
    rm -rf logs/
    @echo "Cleaned output directories"

# Clean and run everything
fresh: clean all
    @echo "Fresh run completed!"

# Check if environment is set up correctly
check:
    @echo "Checking Python environment..."
    @python --version
    @echo "\nChecking CUDA availability..."
    @python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"
    @echo "\nChecking required packages..."
    @python -c "import transformers, peft, datasets, hydra; print('All required packages installed')"

# Show current results summary
results:
    @echo "=== Training Results Summary ==="
    @echo ""
    @for dir in outputs/*/; do \
        if [ -f "$$dir/evaluation/evaluation_results.json" ]; then \
            echo "$$dir:"; \
            python -c "import json; data=json.load(open('$$dir/evaluation/evaluation_results.json')); print(f\"  Test Accuracy: {data['test_accuracy']*100:.2f}%\")"; \
        fi \
    done

# Open tensorboard (if using wandb locally)
tensorboard:
    tensorboard --logdir=outputs/

# ==================== Development ====================

# Quick test run (1 epoch, small batch)
test:
    python train.py training.num_epochs=1 training.batch_size=32 output_dir=outputs/test
    python evaluate.py training.batch_size=32 output_dir=outputs/test
    @echo "Test run completed"

# Debug mode with verbose output
debug:
    python train.py training.num_epochs=1 training.batch_size=8 hydra.verbose=true

# Format code
format:
    black src/ train.py evaluate.py analyze.py
    isort src/ train.py evaluate.py analyze.py

# Lint code
lint:
    flake8 src/ train.py evaluate.py analyze.py
    mypy src/
