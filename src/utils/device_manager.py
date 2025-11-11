"""Smart device management with automatic fallback and OOM handling."""

import time
import torch
import subprocess
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def get_gpu_memory_info():
    """Get free memory for all GPUs.

    Returns:
        List of tuples (gpu_id, free_memory_mb, total_memory_mb)
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )

        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            gpu_id, free_mem, total_mem = map(int, line.split(','))
            gpu_info.append((gpu_id, free_mem, total_mem))

        return gpu_info
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def find_best_device(min_memory_mb=4000, exclude_devices=None):
    """Find GPU with most free memory.

    Args:
        min_memory_mb: Minimum free memory required (MB)
        exclude_devices: List of device IDs to exclude

    Returns:
        Device string (e.g., "cuda:0") or "cpu"
    """
    exclude_devices = exclude_devices or []
    gpu_info = get_gpu_memory_info()

    if not gpu_info:
        logger.warning("No GPUs available, using CPU")
        return "cpu"

    # Filter available GPUs
    available_gpus = [
        (gpu_id, free_mem)
        for gpu_id, free_mem, _ in gpu_info
        if free_mem >= min_memory_mb and gpu_id not in exclude_devices
    ]

    if not available_gpus:
        logger.warning(f"No GPU with {min_memory_mb}MB free memory, using CPU")
        return "cpu"

    # Pick GPU with most free memory
    best_gpu_id = max(available_gpus, key=lambda x: x[1])[0]
    logger.info(f"Selected cuda:{best_gpu_id} with {available_gpus[best_gpu_id][1]}MB free")

    return f"cuda:{best_gpu_id}"


def wait_for_device(min_memory_mb=4000, timeout_minutes=60, check_interval_seconds=30):
    """Wait until a GPU becomes available.

    Args:
        min_memory_mb: Minimum free memory required
        timeout_minutes: Maximum wait time
        check_interval_seconds: How often to check

    Returns:
        Device string or None if timeout
    """
    timeout_seconds = timeout_minutes * 60
    elapsed = 0

    logger.info(f"Waiting for GPU with {min_memory_mb}MB free memory...")

    while elapsed < timeout_seconds:
        device = find_best_device(min_memory_mb=min_memory_mb)

        if device != "cpu":
            logger.info(f"Found available device: {device}")
            return device

        logger.info(f"No GPU available. Waiting {check_interval_seconds}s... ({elapsed}/{timeout_seconds}s)")
        time.sleep(check_interval_seconds)
        elapsed += check_interval_seconds

    logger.warning("Timeout waiting for GPU")
    return None


class OOMHandler:
    """Handles CUDA Out of Memory errors with retry logic."""

    def __init__(self, initial_batch_size, min_batch_size=4, max_retries=3):
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_retries = max_retries
        self.retry_count = 0

    def reduce_batch_size(self):
        """Reduce batch size by half."""
        self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
        logger.info(f"Reduced batch size to {self.current_batch_size}")
        return self.current_batch_size

    def should_retry(self):
        """Check if we should retry."""
        return self.retry_count < self.max_retries and self.current_batch_size >= self.min_batch_size

    def handle_oom(self, device):
        """Handle OOM error.

        Args:
            device: Current device

        Returns:
            tuple: (should_retry, new_batch_size, new_device)
        """
        self.retry_count += 1

        # Clear CUDA cache
        if "cuda" in device:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if not self.should_retry():
            logger.error(f"Max retries ({self.max_retries}) reached or batch size too small")
            return False, None, None

        # Try reducing batch size first
        if self.current_batch_size > self.min_batch_size:
            new_batch_size = self.reduce_batch_size()
            logger.info(f"Retry {self.retry_count}/{self.max_retries} with batch_size={new_batch_size}")
            return True, new_batch_size, device

        # If batch size already at minimum, try different device
        logger.info("Batch size at minimum, trying different device...")
        gpu_info = get_gpu_memory_info()

        # Get current GPU ID
        current_gpu = int(device.split(":")[-1]) if "cuda:" in device else None

        # Try next GPU
        if current_gpu is not None and len(gpu_info) > 1:
            next_gpu = (current_gpu + 1) % len(gpu_info)
            new_device = f"cuda:{next_gpu}"
            logger.info(f"Switching to {new_device}")

            # Reset batch size when switching device
            self.current_batch_size = self.initial_batch_size
            return True, self.current_batch_size, new_device

        # Fall back to CPU as last resort
        logger.warning("Falling back to CPU")
        self.current_batch_size = self.initial_batch_size
        return True, self.current_batch_size, "cpu"


def clear_gpu_memory(device):
    """Clear GPU memory cache.

    Args:
        device: Device string
    """
    if "cuda" in device:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info(f"Cleared GPU memory cache for {device}")


def print_memory_summary(device):
    """Print memory usage summary.

    Args:
        device: Device string
    """
    if "cuda" in device:
        gpu_id = int(device.split(":")[-1])
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(gpu_id) / 1024**3

        logger.info(f"\nGPU Memory Summary ({device}):")
        logger.info(f"  Allocated: {allocated:.2f} GB")
        logger.info(f"  Reserved: {reserved:.2f} GB")
        logger.info(f"  Max Allocated: {max_allocated:.2f} GB")
