"""Training utilities for CLIP fine-tuning."""

import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm


class Trainer:
    """Generic trainer for CLIP fine-tuning."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs=1,
        gradient_accumulation_steps=1,
        output_dir="outputs",
        use_wandb=False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_wandb = use_wandb
        if use_wandb:
            import wandb
            self.wandb = wandb

        # Metrics storage
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.epoch_times = []

    def train(self):
        """Run full training loop."""
        print(f"Starting training for {self.num_epochs} epochs...")

        for epoch in range(self.num_epochs):
            epoch_start = time.time()

            # Train and validate
            train_loss = self._train_epoch(epoch)
            val_loss, val_acc = self._validate_epoch(epoch)

            # Record metrics
            epoch_time = time.time() - epoch_start
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.epoch_times.append(epoch_time)

            # Log
            print(
                f"Epoch {epoch + 1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc * 100:.2f}% | "
                f"Time: {epoch_time:.2f}s"
            )

            if self.use_wandb:
                self.wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "epoch_time": epoch_time,
                })

        # Save results
        self._save_metrics()
        print(f"\nTraining complete! Metrics saved to {self.output_dir}")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "epoch_times": self.epoch_times,
        }

    def _train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        self.optimizer.zero_grad()

        for batch_idx, (images, labels) in enumerate(
            tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs} [Train]")
        ):
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # Backward pass with gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            running_loss += loss.item() * self.gradient_accumulation_steps

        # Handle remaining gradients
        if (batch_idx + 1) % self.gradient_accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return running_loss / len(self.train_loader)

    def _validate_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(
                self.val_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs} [Val]"
            ):
                images, labels = images.to(self.device), labels.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return val_loss / len(self.val_loader), correct / total

    def _save_metrics(self):
        """Save training metrics to JSON."""
        metrics = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "epoch_times": self.epoch_times,
        }

        metrics_path = self.output_dir / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
