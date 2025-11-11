#!/usr/bin/env python3
"""Analysis script for visualizing training and evaluation results."""

import hydra
from omegaconf import DictConfig, OmegaConf
import json
from pathlib import Path
import numpy as np

from src.utils import plot_training_curves, plot_confusion_matrix, compare_models


def load_results(output_dir):
    """Load training and evaluation results from output directory.

    Args:
        output_dir: Path to output directory

    Returns:
        dict: Results dictionary
    """
    output_dir = Path(output_dir)

    results = {}

    # Load training metrics
    metrics_path = output_dir / "training_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            results["training"] = json.load(f)

    # Load evaluation results
    eval_dir = output_dir / "evaluation"
    eval_path = eval_dir / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path, "r") as f:
            results["evaluation"] = json.load(f)

    return results


def analyze_single_model(output_dir, model_name="Model"):
    """Analyze results for a single model.

    Args:
        output_dir: Path to model output directory
        model_name: Name for plot titles
    """
    output_dir = Path(output_dir)
    print(f"Analyzing results from: {output_dir}")

    results = load_results(output_dir)

    if not results:
        print(f"No results found in {output_dir}")
        return

    # Plot training curves
    if "training" in results:
        print("\nPlotting training curves...")
        plot_training_curves(
            metrics=results["training"],
            output_path=output_dir / "training_curves.png",
            title_prefix=model_name
        )

    # Plot confusion matrix
    if "evaluation" in results:
        print("Plotting confusion matrix...")
        cm = np.array(results["evaluation"]["confusion_matrix"])
        plot_confusion_matrix(
            cm=cm,
            output_path=output_dir / "evaluation" / "confusion_matrix.png",
            title=f"{model_name} - Confusion Matrix"
        )

        # Print summary statistics
        print(f"\n{model_name} Test Results:")
        print(f"Test Loss: {results['evaluation']['test_loss']:.4f}")
        print(f"Test Accuracy: {results['evaluation']['test_accuracy'] * 100:.2f}%")

    print(f"\nAnalysis complete! Visualizations saved to {output_dir}")


def compare_multiple_models(output_dirs, model_names, comparison_output_dir="outputs/comparison"):
    """Compare results from multiple models.

    Args:
        output_dirs: List of output directories
        model_names: List of model names
        comparison_output_dir: Directory to save comparison plots
    """
    comparison_output_dir = Path(comparison_output_dir)
    comparison_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Comparing {len(output_dirs)} models...")

    results_dict = {}
    for output_dir, name in zip(output_dirs, model_names):
        results = load_results(output_dir)
        if "evaluation" in results:
            results_dict[name] = results["evaluation"]

    if len(results_dict) < 2:
        print("Need at least 2 models with evaluation results for comparison")
        return

    # Generate comparison plots
    compare_models(results_dict, output_dir=comparison_output_dir)

    print(f"\nComparison complete! Results saved to {comparison_output_dir}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main analysis function.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print("CLIP Fine-tuning Analysis")
    print("=" * 80)

    # Determine output directory from config
    output_dir = Path(cfg.output_dir)
    model_name = cfg.model.name

    # Check if we should compare models
    if cfg.get("compare", None):
        # Compare mode: analyze multiple model/dataset combinations
        print("\nComparison Mode")
        print("=" * 80)

        output_dirs = []
        model_names = []

        for comparison in cfg.compare:
            comp_output_dir = Path(comparison.output_dir)
            comp_model_name = comparison.get("name", comparison.output_dir)
            output_dirs.append(comp_output_dir)
            model_names.append(comp_model_name)

        print(f"\nComparing {len(output_dirs)} models:")
        for name, path in zip(model_names, output_dirs):
            print(f"  - {name}: {path}")

        compare_multiple_models(
            output_dirs,
            model_names,
            comparison_output_dir=cfg.get("comparison_output_dir", "outputs/comparison")
        )
    else:
        # Single model analysis
        print(f"\nAnalyzing single model: {model_name}")
        print(f"Dataset: {cfg.dataset.name}")
        print(f"Output directory: {output_dir}")
        print("=" * 80)

        analyze_single_model(output_dir, model_name=f"{model_name.replace('_', ' ').title()}")

    print("\n" + "=" * 80)
    print("Analysis completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
