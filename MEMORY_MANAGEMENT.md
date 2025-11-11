# Memory Management & OOM Handling

Guide for handling insufficient GPU memory and CUDA Out of Memory (OOM) errors.

## Quick Start

### Option 1: Use Robust Training Script (Recommended)

```bash
# Auto-select best available GPU and handle OOM errors
python train_robust.py model=lora dataset=flowers102

# Wait for GPU if none available
python train_robust.py model=lora dataset=flowers102 wait_for_gpu=true

# Use robust config preset
python train_robust.py --config-name robust model=lora dataset=flowers102
```

### Option 2: Manual Configuration

```bash
# Reduce batch size
python train.py training.batch_size=32

# Use gradient accumulation (same effective batch size, less memory)
python train.py training.batch_size=32 training.gradient_accumulation_steps=4

# Try different GPU
python train.py device=cuda:1
```

## Robust Training Features

The `train_robust.py` script provides:

1. **Automatic Device Selection**: Finds GPU with most free memory
2. **OOM Error Handling**: Automatically retries with smaller batch size
3. **Device Fallback**: Tries different GPUs, falls back to CPU if needed
4. **GPU Wait Mode**: Waits for GPU to become available
5. **Memory Monitoring**: Tracks and reports memory usage

## Configuration

### Robust Config (`configs/robust.yaml`)

```yaml
# Auto device selection
auto_device: true               # Automatically find best GPU
wait_for_gpu: true              # Wait if no GPU available
min_gpu_memory_mb: 4000         # Minimum GPU memory required
gpu_wait_timeout_minutes: 60    # Max wait time
gpu_check_interval_seconds: 30  # Check interval

# OOM handling
min_batch_size: 4               # Minimum batch size before giving up
max_retries: 3                  # Max retries on OOM
```

### Override in Command Line

```bash
python train_robust.py \
  auto_device=true \
  min_gpu_memory_mb=6000 \
  min_batch_size=8 \
  max_retries=5
```

## Strategies

### 1. Automatic Device Selection

The system checks all available GPUs and selects the one with most free memory:

```bash
# Let system choose best GPU
python train_robust.py auto_device=true
```

**How it works:**
1. Queries `nvidia-smi` for GPU memory
2. Filters GPUs with enough free memory
3. Selects GPU with most available memory
4. Falls back to CPU if no suitable GPU found

### 2. OOM Error Handling

When CUDA OOM occurs, the system automatically:

1. **Reduces batch size by half**
2. **Clears GPU cache**
3. **Retries training**
4. If still fails → **tries different GPU**
5. Last resort → **falls back to CPU**

```bash
# Training with OOM handling
python train_robust.py model=lora max_retries=5 min_batch_size=4
```

### 3. Wait for GPU

If all GPUs are busy, wait for one to become available:

```bash
# Wait up to 60 minutes for GPU
python train_robust.py \
  wait_for_gpu=true \
  gpu_wait_timeout_minutes=60 \
  gpu_check_interval_seconds=30
```

**Use case:** Queue multiple jobs that will run when resources available

### 4. Manual Batch Size Reduction

If you know your memory constraints:

```bash
# Start with smaller batch size
python train.py training.batch_size=16

# Increase gradient accumulation to maintain effective batch size
# Effective batch size = batch_size * gradient_accumulation_steps
python train.py training.batch_size=16 training.gradient_accumulation_steps=8
```

## Justfile Integration

Update your justfile to use robust training:

```just
# Robust training recipes
train-linear-flowers-robust:
    python train_robust.py model=linear_probe dataset=flowers102

train-lora-flowers-robust:
    python train_robust.py model=lora dataset=flowers102

# Auto-select device for all trainings
train-all-robust:
    python train_robust.py model=linear_probe dataset=flowers102 &
    python train_robust.py model=linear_probe dataset=cub200 &
    python train_robust.py model=lora dataset=flowers102 &
    python train_robust.py model=lora dataset=cub200 &
    wait
```

## Memory Optimization Tips

### 1. Reduce Model Size
```bash
# Use smaller LoRA rank
python train.py model=lora model.lora_r=4  # Instead of 8

# Reduce number of workers
python train.py training.num_workers=2
```

### 2. Gradient Accumulation
```yaml
# configs/config.yaml
training:
  batch_size: 32                    # Actual batch size (uses less memory)
  gradient_accumulation_steps: 4    # Effective batch = 32 * 4 = 128
```

### 3. Mixed Precision (Future Enhancement)
```python
# Add to trainer.py
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 4. Gradient Checkpointing (For Very Large Models)
```python
# Add to model
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    return checkpoint(self._forward, x)
```

## Monitoring Memory

### Check GPU Status
```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or use justfile
just check
```

### Memory Summary During Training
```bash
# Robust script automatically prints memory summary
python train_robust.py model=lora
```

Output:
```
GPU Memory Summary (cuda:0):
  Allocated: 5.23 GB
  Reserved: 6.45 GB
  Max Allocated: 7.12 GB
```

## Common Scenarios

### Scenario 1: All GPUs Busy
```bash
# Wait for any GPU to become available
python train_robust.py \
  wait_for_gpu=true \
  gpu_wait_timeout_minutes=120
```

### Scenario 2: Intermittent OOM
```bash
# Aggressive OOM handling
python train_robust.py \
  max_retries=5 \
  min_batch_size=4 \
  auto_device=true
```

### Scenario 3: Known Memory Constraints
```bash
# Manually optimize for your hardware
python train.py \
  training.batch_size=16 \
  training.gradient_accumulation_steps=8 \
  device=cuda:2
```

### Scenario 4: CPU Fallback
```bash
# Allow CPU fallback but prefer GPU
python train_robust.py auto_device=true min_gpu_memory_mb=2000
```

## Troubleshooting

### OOM Even with Batch Size 1
```bash
# Try:
1. Use smaller model (reduce lora_r)
2. Reduce number of workers
3. Use CPU (slow but works)

python train.py device=cpu training.num_workers=0
```

### Script Exits Without Retry
Check logs for:
- Non-OOM CUDA error (e.g., driver issue)
- Other exceptions not related to memory

### GPU Not Detected
```bash
# Check CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Check nvidia-smi works
nvidia-smi
```

## API Usage

### In Your Own Scripts

```python
from src.utils import find_best_device, OOMHandler, retry_on_oom

# Find best device
device = find_best_device(min_memory_mb=4000)

# OOM handler
oom_handler = OOMHandler(initial_batch_size=128, min_batch_size=4)

# Retry decorator
@retry_on_oom(max_retries=3)
def train_model():
    # Your training code
    pass

# Adaptive batch training
from src.utils import adaptive_batch_training

def train_fn(batch_size):
    # Training code that uses batch_size
    return results

results = adaptive_batch_training(train_fn, initial_batch_size=128)
```

## Best Practices

1. **Start with robust script** for unknown hardware
2. **Monitor memory usage** in first run
3. **Tune batch size** based on observations
4. **Use gradient accumulation** to maintain effective batch size
5. **Consider mixed precision** for very large models
6. **Queue jobs** with wait_for_gpu when resources are shared

## Summary

| Problem | Solution | Command |
|---------|----------|---------|
| OOM error | Use robust script | `python train_robust.py` |
| All GPUs busy | Wait mode | `wait_for_gpu=true` |
| Unknown hardware | Auto device | `auto_device=true` |
| Known constraints | Manual config | `training.batch_size=32` |
| Large models | Gradient accum | `gradient_accumulation_steps=4` |
