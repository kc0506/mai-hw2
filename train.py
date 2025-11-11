#!/usr/bin/env python3
"""Training script for CLIP fine-tuning with Hydra configuration."""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from transformers import CLIPProcessor
from pathlib import Path

from src.data import get_dataset, get_dataloaders
from src.models import LinearProbeModel, LoRAModel
from src.training import Trainer


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print("CLIP Fine-tuning Training")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Set random seed
    torch.manual_seed(cfg.seed)

    # Setup device
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

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

    print(sum(p.numel() for p in optimizer.param_groups[0]['params']))


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

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=cfg.training.num_epochs,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        output_dir=output_dir,
        use_wandb=cfg.use_wandb,
    )

    # Train
    print("\n" + "=" * 80)
    metrics = trainer.train()

    # Save model checkpoint
    checkpoint_path = output_dir / "model_checkpoint.pt"
    model.save_checkpoint(checkpoint_path)
    print(f"\nModel checkpoint saved to: {checkpoint_path}")

    # Save LoRA adapters if applicable
    if cfg.model.type == "lora":
        adapters_path = output_dir / "lora_adapters"
        model.save_lora_adapters(adapters_path)
        print(f"LoRA adapters saved to: {adapters_path}")

    if cfg.use_wandb:
        import wandb
        wandb.finish()

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
