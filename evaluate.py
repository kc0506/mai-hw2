#!/usr/bin/env python3
"""Evaluation script for trained CLIP models."""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from transformers import CLIPProcessor
from pathlib import Path

from src.data import get_dataset, get_dataloaders
from src.models import LinearProbeModel, LoRAModel
from src.evaluation import Evaluator


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main evaluation function.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print("CLIP Fine-tuning Evaluation")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

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

    print(f"Test samples: {len(test_dts)}")
    print(f"Number of classes: {num_classes}")

    # Create test dataloader
    _, _, test_loader = get_dataloaders(
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
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    # Load checkpoint
    checkpoint_path = Path(cfg.output_dir) / "model_checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\nLoading checkpoint from: {checkpoint_path}")
    model.load_checkpoint(checkpoint_path)

    # Setup evaluator
    criterion = nn.CrossEntropyLoss()
    output_dir = Path(cfg.output_dir) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        output_dir=output_dir
    )

    # Evaluate
    print("\n" + "=" * 80)
    results = evaluator.evaluate(class_names=class_names)

    print("\n" + "=" * 80)
    print("Evaluation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
