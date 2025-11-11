#!/usr/bin/env python3
"""Robust training script with automatic device selection and OOM handling."""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from transformers import CLIPProcessor
from pathlib import Path
import logging

from src.data import get_dataset, get_dataloaders
from src.models import LinearProbeModel, LoRAModel
from src.training import Trainer
from src.utils.device_manager import (
    find_best_device,
    wait_for_device,
    OOMHandler,
    clear_gpu_memory,
    print_memory_summary,
)
from src.utils.retry_decorator import retry_on_oom

logger = logging.getLogger(__name__)


def train_with_oom_handling(cfg, model, train_loader, val_loader, criterion, optimizer, device):
    """Train model with OOM error handling.

    Args:
        cfg: Configuration
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss criterion
        optimizer: Optimizer
        device: Device to use

    Returns:
        Training metrics
    """
    oom_handler = OOMHandler(
        initial_batch_size=cfg.training.batch_size,
        min_batch_size=cfg.get("min_batch_size", 4),
        max_retries=cfg.get("max_retries", 3)
    )

    current_device = device

    while True:
        try:
            logger.info(f"Starting training on {current_device}")

            # Move model to current device
            if hasattr(model, 'vision_model'):
                model.vision_model.to(current_device)
            if hasattr(model, 'visual_projection'):
                model.visual_projection.to(current_device)
            if hasattr(model, 'vision_model_lora'):
                model.vision_model_lora.to(current_device)
            if hasattr(model, 'head'):
                model.head.to(current_device)

            # Create trainer
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=current_device,
                num_epochs=cfg.training.num_epochs,
                gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
                output_dir=cfg.output_dir,
                use_wandb=cfg.use_wandb,
            )

            # Train
            metrics = trainer.train()

            # Success!
            logger.info("Training completed successfully")
            print_memory_summary(current_device)

            return metrics

        except RuntimeError as e:
            error_msg = str(e).lower()

            if "out of memory" in error_msg or "cuda" in error_msg:
                logger.warning(f"OOM Error: {e}")

                # Handle OOM
                should_retry, new_batch_size, new_device = oom_handler.handle_oom(current_device)

                if not should_retry:
                    logger.error("Cannot recover from OOM. Exiting.")
                    raise

                # Update configuration
                if new_batch_size != cfg.training.batch_size:
                    logger.info(f"Updating batch_size: {cfg.training.batch_size} -> {new_batch_size}")
                    cfg.training.batch_size = new_batch_size

                    # Recreate data loaders with new batch size
                    train_loader, val_loader, _ = get_dataloaders(
                        train_loader.dataset,
                        val_loader.dataset,
                        train_loader.dataset,  # dummy
                        batch_size=new_batch_size,
                        num_workers=cfg.training.num_workers
                    )

                if new_device != current_device:
                    logger.info(f"Switching device: {current_device} -> {new_device}")
                    current_device = new_device
                    clear_gpu_memory(device)

                logger.info("Retrying training...")

            else:
                # Not an OOM error, re-raise
                logger.error(f"Non-OOM error: {e}")
                raise


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function with robust error handling.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print("CLIP Fine-tuning Training (Robust Mode)")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Set random seed
    torch.manual_seed(cfg.seed)

    # Setup device with auto-selection
    if cfg.get("auto_device", False):
        logger.info("Auto-selecting best available device...")
        device = find_best_device(
            min_memory_mb=cfg.get("min_gpu_memory_mb", 4000)
        )

        if device == "cpu" and cfg.get("wait_for_gpu", False):
            logger.info("No GPU available, waiting...")
            device = wait_for_device(
                min_memory_mb=cfg.get("min_gpu_memory_mb", 4000),
                timeout_minutes=cfg.get("gpu_wait_timeout_minutes", 60),
                check_interval_seconds=cfg.get("gpu_check_interval_seconds", 30)
            )

            if device is None:
                logger.warning("Timeout waiting for GPU, using CPU")
                device = "cpu"
    else:
        device = cfg.device if torch.cuda.is_available() else "cpu"

    print(f"\nUsing device: {device}")

    if "cuda" in device:
        print_memory_summary(device)

    # Load CLIP processor
    print(f"Loading CLIP processor: {cfg.clip_model_id}")
    processor = CLIPProcessor.from_pretrained(cfg.clip_model_id)

    # Load dataset
    print(f"\nLoading dataset: {cfg.dataset.name}")
    train_dts, val_dts, test_dts, class_names, num_classes = get_dataset(
        dataset_name=cfg.dataset.name,
        clip_processor=processor,
        root_dir=cfg.dataset.get("root_dir", "../data"),
        download=cfg.dataset.get("download", True),
        val_split=cfg.dataset.get("val_split", 0.1),
    )

    print(f"Train samples: {len(train_dts)}")
    print(f"Val samples: {len(val_dts)}")
    print(f"Test samples: {len(test_dts)}")
    print(f"Number of classes: {num_classes}")

    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dts, val_dts, test_dts,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers
    )

    # Create model
    print(f"\nCreating model: {cfg.model.type}")
    if cfg.model.type == "linear_probe":
        model = LinearProbeModel(
            model_id=cfg.clip_model_id,
            num_classes=num_classes,
            device=device
        )
    elif cfg.model.type == "lora":
        model = LoRAModel(
            model_id=cfg.clip_model_id,
            num_classes=num_classes,
            lora_r=cfg.model.lora_r,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
            device=device
        )
        print("\nLoRA Trainable Parameters:")
        model.print_trainable_parameters()
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.get_trainable_parameters(),
        lr=cfg.model.learning_rate
    )

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)
    print(f"\nConfig saved to: {config_path}")

    # Initialize wandb if enabled
    if cfg.use_wandb:
        import wandb
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"{cfg.model.name}_{cfg.dataset.name}",
        )

    # Train with OOM handling
    print("\n" + "=" * 80)
    try:
        metrics = train_with_oom_handling(
            cfg, model, train_loader, val_loader, criterion, optimizer, device
        )

        # Save model checkpoint
        checkpoint_path = output_dir / "model_checkpoint.pt"
        model.save_checkpoint(checkpoint_path)
        print(f"\nModel checkpoint saved to: {checkpoint_path}")

        # Save LoRA adapters if applicable
        if cfg.model.type == "lora":
            adapters_path = output_dir / "lora_adapters"
            model.save_lora_adapters(adapters_path)
            print(f"LoRA adapters saved to: {adapters_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    finally:
        if cfg.use_wandb:
            import wandb
            wandb.finish()

        # Clean up
        clear_gpu_memory(device)

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
