# Justfile Command Reference

Quick reference for all justfile commands. Run `just --list` to see all available commands.

## Installation

```bash
# macOS
brew install just

# Linux (cargo)
cargo install just

# Ubuntu/Debian
wget -qO - 'https://proget.makedeb.org/debian-feeds/prebuilt-mpr.pub' | gpg --dearmor | sudo tee /usr/share/keyrings/prebuilt-mpr-archive-keyring.gpg 1> /dev/null
echo "deb [arch=all,$(dpkg --print-architecture) signed-by=/usr/share/keyrings/prebuilt-mpr-archive-keyring.gpg] https://proget.makedeb.org prebuilt-mpr $(lsb_release -cs)" | sudo tee /etc/apt/sources.list.d/prebuilt-mpr.list
sudo apt update
sudo apt install just
```

## Quick Reference

### 🚀 Complete Pipelines (Train → Eval → Analyze)

| Command | Description |
|---------|-------------|
| `just pipeline-linear-flowers` | Linear probe on Flowers102 |
| `just pipeline-linear-cub` | Linear probe on CUB-200 |
| `just pipeline-lora-flowers` | LoRA on Flowers102 |
| `just pipeline-lora-cub` | LoRA on CUB-200 |

### 📊 By Dataset

| Command | Description |
|---------|-------------|
| `just flowers-all` | Both models on Flowers102 + comparison |
| `just cub-all` | Both models on CUB-200 |

### 🔬 By Method

| Command | Description |
|---------|-------------|
| `just linear-all` | Linear probe on both datasets |
| `just lora-all` | LoRA on both datasets |

### 🎯 Individual Operations

**Training:**
| Command | Description |
|---------|-------------|
| `just train-linear-flowers` | Train linear probe on Flowers102 |
| `just train-linear-cub` | Train linear probe on CUB-200 |
| `just train-lora-flowers` | Train LoRA on Flowers102 |
| `just train-lora-cub` | Train LoRA on CUB-200 |
| `just train-all` | Train all combinations |

**Evaluation:**
| Command | Description |
|---------|-------------|
| `just eval-linear-flowers` | Evaluate linear probe on Flowers102 |
| `just eval-linear-cub` | Evaluate linear probe on CUB-200 |
| `just eval-lora-flowers` | Evaluate LoRA on Flowers102 |
| `just eval-lora-cub` | Evaluate LoRA on CUB-200 |
| `just eval-all` | Evaluate all models |

**Analysis:**
| Command | Description |
|---------|-------------|
| `just analyze-linear-flowers` | Analyze linear probe on Flowers102 |
| `just analyze-linear-cub` | Analyze linear probe on CUB-200 |
| `just analyze-lora-flowers` | Analyze LoRA on Flowers102 |
| `just analyze-lora-cub` | Analyze LoRA on CUB-200 |
| `just analyze-all` | Analyze all results |
| `just compare-flowers` | Compare models on Flowers102 |
| `just compare-all` | Generate all comparisons |

### 🛠️ Utilities

| Command | Description |
|---------|-------------|
| `just all` | Run EVERYTHING (full pipeline) |
| `just clean` | Remove output directories |
| `just fresh` | Clean + run all |
| `just check` | Check environment setup |
| `just results` | Show results summary |
| `just test` | Quick test run (1 epoch) |
| `just debug` | Debug mode with verbose output |
| `just tensorboard` | Launch tensorboard |

### 🧹 Development

| Command | Description |
|---------|-------------|
| `just format` | Format code with black/isort |
| `just lint` | Lint code with flake8/mypy |

## Common Workflows

### 1. Run Everything
```bash
just all
```
This runs: train-all → eval-all → analyze-all → compare-all

### 2. Quick Test
```bash
just test
```
Fast sanity check (1 epoch, small batch)

### 3. Single Experiment
```bash
just pipeline-lora-flowers
```
Train → Evaluate → Analyze LoRA on Flowers102

### 4. Compare Methods
```bash
# First run experiments
just pipeline-linear-flowers
just pipeline-lora-flowers

# Then compare
just compare-flowers
```

### 5. Fresh Start
```bash
just fresh
```
Cleans outputs and runs everything from scratch

### 6. Check Setup
```bash
just check
```
Verifies Python, CUDA, and package installation

### 7. View Results
```bash
just results
```
Shows test accuracy for all completed experiments

## Tips

1. **Parallel Execution**: Just runs dependencies in order, but you can run independent commands in parallel:
   ```bash
   just train-linear-flowers &
   just train-lora-cub &
   wait
   ```

2. **Dry Run**: See what would be executed:
   ```bash
   just --dry-run all
   ```

3. **Custom Commands**: Edit `justfile` to add your own recipes

4. **Override Config**: Recipes use default configs, but you can still manually override:
   ```bash
   # Instead of: just train-linear-flowers
   python train.py model=linear_probe dataset=flowers102 training.num_epochs=10
   ```

## Troubleshooting

**Command not found:**
```bash
# Make sure just is installed
just --version

# If not, install it (see Installation section)
```

**Recipe fails:**
```bash
# Check which step failed in the pipeline
just train-linear-flowers  # Test each step individually
just eval-linear-flowers
just analyze-linear-flowers
```

**Need to rerun just one step:**
```bash
# Just run that specific command
just eval-linear-flowers  # Re-evaluate without retraining
```

## Recipe Composition

Justfile recipes can depend on other recipes:

```
pipeline-linear-flowers: train-linear-flowers eval-linear-flowers analyze-linear-flowers
    @echo "✓ Completed pipeline"
```

This means:
1. Runs `train-linear-flowers` first
2. Then runs `eval-linear-flowers`
3. Then runs `analyze-linear-flowers`
4. Finally prints completion message

If any step fails, the pipeline stops.
