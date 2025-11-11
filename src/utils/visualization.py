"""Visualization utilities."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_training_curves(metrics, output_path="training_curves.png", title_prefix=""):
    """Plot training and validation curves.

    Args:
        metrics: Dict with train_losses, val_losses, val_accuracies
        output_path: Path to save figure
        title_prefix: Prefix for plot titles (e.g., "Linear Probe" or "LoRA")
    """
    train_losses = metrics["train_losses"]
    val_losses = metrics["val_losses"]
    val_accuracies = metrics["val_accuracies"]
    num_epochs = len(train_losses)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    ax1.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", marker="o")
    ax1.plot(range(1, num_epochs + 1), val_losses, label="Val Loss", marker="s")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title_prefix} Loss Curves")
    ax1.legend()
    ax1.grid(True)

    # Accuracy curve
    ax2.plot(
        range(1, num_epochs + 1),
        [acc * 100 for acc in val_accuracies],
        label="Val Accuracy",
        marker="o",
        color="green",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{title_prefix} Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {output_path}")


def plot_confusion_matrix(
    cm, class_names=None, output_path="confusion_matrix.png", title="Confusion Matrix", cmap="Blues"
):
    """Plot confusion matrix.

    Args:
        cm: Confusion matrix array
        class_names: List of class names (optional)
        output_path: Path to save figure
        title: Plot title
        cmap: Colormap
    """
    plt.figure(figsize=(12, 10))

    if class_names and len(class_names) <= 20:
        # Show labels if not too many classes
        sns.heatmap(
            cm, cmap=cmap, fmt="d", cbar=True,
            xticklabels=class_names, yticklabels=class_names
        )
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
    else:
        # Don't show labels for many classes
        sns.heatmap(cm, cmap=cmap, fmt="d", cbar=True)

    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def compare_models(results_dict, output_dir="outputs"):
    """Compare multiple model results.

    Args:
        results_dict: Dict mapping model_name -> results
        output_dir: Directory to save comparison plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compare accuracies
    model_names = list(results_dict.keys())
    accuracies = [results_dict[name]["test_accuracy"] * 100 for name in model_names]

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, accuracies, color=["skyblue", "lightcoral"])
    plt.ylabel("Test Accuracy (%)")
    plt.title("Model Comparison")
    plt.ylim([0, 100])
    for i, v in enumerate(accuracies):
        plt.text(i, v + 1, f"{v:.2f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=150)
    plt.close()
    print(f"Model comparison saved to {output_dir / 'model_comparison.png'}")
