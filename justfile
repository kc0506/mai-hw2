# Justfile for CLIP Fine-tuning Pipeline
# Usage: just <recipe_name> [args...]
# Example: just train-linear-flowers device=cuda:1 training.num_epochs=5

# Default recipe - show help
default:
    @just --list

# ==================== Flexible Commands ====================

# Train with any configuration
train *ARGS:
    python train.py {{ ARGS }}

# Train with robust mode
train-robust *ARGS:
    python train_robust.py {{ ARGS }}

# Evaluate with any configuration
eval *ARGS:
    python evaluate.py {{ ARGS }}

# Analyze with any configuration
analyze *ARGS:
    python analyze.py {{ ARGS }}

# ==================== Individual Commands ====================

# Train linear probe on Flowers102
train-linear-flowers *ARGS:
    python train.py model=linear_probe dataset=flowers102 {{ ARGS }}

# Train linear probe on Flowers102 (robust mode)
train-linear-flowers-robust *ARGS:
    python train_robust.py model=linear_probe dataset=flowers102 {{ ARGS }}

# Train linear probe on CUB-200
train-linear-cub *ARGS:
    python train.py model=linear_probe dataset=cub200 {{ ARGS }}

# Train linear probe on CUB-200 (robust mode)
train-linear-cub-robust *ARGS:
    python train_robust.py model=linear_probe dataset=cub200 {{ ARGS }}

# Train LoRA on Flowers102
train-lora-flowers *ARGS:
    python train.py model=lora dataset=flowers102 {{ ARGS }} training.batch_size=32 training.gradient_accumulation_steps=8

# Train LoRA on Flowers102 (robust mode)
train-lora-flowers-robust *ARGS:
    python train_robust.py model=lora dataset=flowers102 {{ ARGS }}

# Train LoRA on CUB-200
train-lora-cub *ARGS:
    python train.py model=lora dataset=cub200 {{ ARGS }} training.batch_size=32 training.gradient_accumulation_steps=8

# Train LoRA on CUB-200 (robust mode)
train-lora-cub-robust *ARGS:
    python train_robust.py model=lora dataset=cub200 {{ ARGS }}

# Evaluate linear probe on Flowers102
eval-linear-flowers *ARGS:
    python evaluate.py model=linear_probe dataset=flowers102 {{ ARGS }}

# Evaluate linear probe on CUB-200
eval-linear-cub *ARGS:
    python evaluate.py model=linear_probe dataset=cub200 {{ ARGS }}

# Evaluate LoRA on Flowers102
eval-lora-flowers *ARGS:
    python evaluate.py model=lora dataset=flowers102 {{ ARGS }}

# Evaluate LoRA on CUB-200
eval-lora-cub *ARGS:
    python evaluate.py model=lora dataset=cub200 {{ ARGS }}

# Analyze linear probe on Flowers102
analyze-linear-flowers *ARGS:
    python analyze.py model=linear_probe dataset=flowers102 {{ ARGS }}

# Analyze linear probe on CUB-200
analyze-linear-cub *ARGS:
    python analyze.py model=linear_probe dataset=cub200 {{ ARGS }}

# Analyze LoRA on Flowers102
analyze-lora-flowers *ARGS:
    python analyze.py model=lora dataset=flowers102 {{ ARGS }}

# Analyze LoRA on CUB-200
analyze-lora-cub *ARGS:
    python analyze.py model=lora dataset=cub200 {{ ARGS }}

# ==================== Full Pipelines ====================

# Run full pipeline for linear probe on Flowers102
pipeline-linear-flowers *ARGS: (train-linear-flowers ARGS) (eval-linear-flowers ARGS) (analyze-linear-flowers ARGS)
    @echo "✓ Completed linear probe pipeline on Flowers102"

# Run full pipeline for linear probe on CUB-200
pipeline-linear-cub *ARGS: (train-linear-cub ARGS) (eval-linear-cub ARGS) (analyze-linear-cub ARGS)
    @echo "✓ Completed linear probe pipeline on CUB-200"

# Run full pipeline for LoRA on Flowers102
pipeline-lora-flowers *ARGS: (train-lora-flowers ARGS) (eval-lora-flowers ARGS) (analyze-lora-flowers ARGS)
    @echo "✓ Completed LoRA pipeline on Flowers102"

# Run full pipeline for LoRA on CUB-200
pipeline-lora-cub *ARGS: (train-lora-cub ARGS) (eval-lora-cub ARGS) (analyze-lora-cub ARGS)
    @echo "✓ Completed LoRA pipeline on CUB-200"

# Run full pipeline with robust mode for linear probe on Flowers102
pipeline-linear-flowers-robust *ARGS: (train-linear-flowers-robust ARGS) (eval-linear-flowers ARGS) (analyze-linear-flowers ARGS)
    @echo "✓ Completed robust linear probe pipeline on Flowers102"

# Run full pipeline with robust mode for LoRA on Flowers102
pipeline-lora-flowers-robust *ARGS: (train-lora-flowers-robust ARGS) (eval-lora-flowers ARGS) (analyze-lora-flowers ARGS)
    @echo "✓ Completed robust LoRA pipeline on Flowers102"

# ==================== Batch Operations ====================

# Train all models on all datasets
train-all *ARGS:
    just train-linear-flowers {{ ARGS }}
    just train-linear-cub {{ ARGS }}
    just train-lora-flowers {{ ARGS }}
    just train-lora-cub {{ ARGS }}
    @echo "✓ Completed training all models"

# Train all with robust mode
train-all-robust *ARGS:
    just train-linear-flowers-robust {{ ARGS }}
    just train-linear-cub-robust {{ ARGS }}
    just train-lora-flowers-robust {{ ARGS }}
    just train-lora-cub-robust {{ ARGS }}
    @echo "✓ Completed training all models (robust mode)"

# Evaluate all models
eval-all *ARGS:
    just eval-linear-flowers {{ ARGS }}
    just eval-linear-cub {{ ARGS }}
    just eval-lora-flowers {{ ARGS }}
    just eval-lora-cub {{ ARGS }}
    @echo "✓ Completed evaluating all models"

# Analyze all models
analyze-all *ARGS:
    just analyze-linear-flowers {{ ARGS }}
    just analyze-linear-cub {{ ARGS }}
    just analyze-lora-flowers {{ ARGS }}
    just analyze-lora-cub {{ ARGS }}
    @echo "✓ Completed analyzing all models"

# Compare models on Flowers102
compare-flowers *ARGS:
    python analyze.py --config-name analysis {{ ARGS }}

# Compare all models
compare-all *ARGS: (compare-flowers ARGS)
    @echo "✓ Completed all comparisons"

# ==================== Complete Workflows ====================

# Run all pipelines for Flowers102 (both models)
flowers-all *ARGS: (pipeline-linear-flowers ARGS) (pipeline-lora-flowers ARGS) (compare-flowers ARGS)
    @echo "✓ Completed all experiments on Flowers102"

# Run all pipelines for CUB-200 (both models)
cub-all *ARGS: (pipeline-linear-cub ARGS) (pipeline-lora-cub ARGS)
    @echo "✓ Completed all experiments on CUB-200"

# Run all linear probe experiments (both datasets)
linear-all *ARGS: (pipeline-linear-flowers ARGS) (pipeline-linear-cub ARGS)
    @echo "✓ Completed all linear probe experiments"

# Run all LoRA experiments (both datasets)
lora-all *ARGS: (pipeline-lora-flowers ARGS) (pipeline-lora-cub ARGS)
    @echo "✓ Completed all LoRA experiments"

# Run EVERYTHING (train, eval, analyze, compare)
all *ARGS: (train-all ARGS) (eval-all ARGS) (analyze-all ARGS) (compare-all ARGS)
    @echo "✓ Completed full experimental pipeline!"

# Run everything with robust mode
all-robust *ARGS: (train-all-robust ARGS) (eval-all ARGS) (analyze-all ARGS) (compare-all ARGS)
    @echo "✓ Completed full experimental pipeline (robust mode)!"

# ==================== Utilities ====================

# Clean output directories
clean:
    rm -rf outputs/
    rm -rf logs/
    @echo "✓ Cleaned output directories"

# Clean and run everything
fresh *ARGS: clean (all ARGS)
    @echo "✓ Fresh run completed!"

# Check if environment is set up correctly
check:
    @echo "Checking Python environment..."
    @python --version
    @echo "\nChecking CUDA availability..."
    @python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"
    @echo "\nChecking required packages..."
    @python -c "import transformers, peft, datasets, hydra; print('✓ All required packages installed')"
    @echo "\nGPU Memory Status:"
    @nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader,nounits || echo "nvidia-smi not available"

# Show current results summary
results:
    @echo "=== Training Results Summary ==="
    @echo ""
    @for dir in outputs/*/*/; do \
        if [ -f "$$dir/evaluation/evaluation_results.json" ]; then \
            echo "$$dir:"; \
            python -c "import json; data=json.load(open('$$dir/evaluation/evaluation_results.json')); print(f\"  Test Accuracy: {data['test_accuracy']*100:.2f}%\")"; \
        fi \
    done

# Open tensorboard
tensorboard:
    tensorboard --logdir=outputs/

# ==================== Development ====================

# Quick test run (1 epoch, small batch)
test *ARGS:
    python train.py training.num_epochs=1 training.batch_size=32 output_dir=outputs/test {{ ARGS }}
    python evaluate.py training.batch_size=32 output_dir=outputs/test {{ ARGS }}
    @echo "✓ Test run completed"

# Quick test with robust mode
test-robust *ARGS:
    python train_robust.py training.num_epochs=1 training.batch_size=32 output_dir=outputs/test {{ ARGS }}
    python evaluate.py training.batch_size=32 output_dir=outputs/test {{ ARGS }}
    @echo "✓ Test run (robust) completed"

# Debug mode with verbose output
debug *ARGS:
    python train.py training.num_epochs=1 training.batch_size=8 hydra.verbose=true {{ ARGS }}

# Format code
format:
    black src/ train.py evaluate.py analyze.py train_robust.py
    isort src/ train.py evaluate.py analyze.py train_robust.py

# Lint code
lint:
    flake8 src/ train.py evaluate.py analyze.py train_robust.py
    mypy src/
