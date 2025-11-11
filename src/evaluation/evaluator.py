"""Evaluation utilities for CLIP fine-tuning."""

import json
from pathlib import Path

import torch
import numpy as np
from tqdm.autonotebook import tqdm
from sklearn.metrics import classification_report, confusion_matrix


class Evaluator:
    """Evaluator for CLIP fine-tuned models."""

    def __init__(self, model, test_loader, criterion, device, output_dir="outputs"):
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, class_names=None):
        """Run evaluation on test set.

        Args:
            class_names: List of class names for classification report

        Returns:
            dict: Evaluation results
        """
        print("Starting evaluation...")
        self.model.eval()

        test_preds = []
        test_labels = []
        test_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)
                test_loss += loss.item()

                _, predicted = torch.max(logits, 1)
                test_preds.extend(predicted.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        test_preds = np.array(test_preds)
        test_labels = np.array(test_labels)
        test_accuracy = (test_preds == test_labels).mean()
        test_loss = test_loss / len(self.test_loader)

        print(f"\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        # Classification report
        if class_names:
            report_dict = classification_report(
                test_labels, test_preds,
                target_names=class_names,
                zero_division=0,
                output_dict=True
            )
            report_str = classification_report(
                test_labels, test_preds,
                target_names=class_names,
                zero_division=0
            )
            print("\nClassification Report:")
            print(report_str)
        else:
            report_dict = None
            report_str = None

        # Confusion matrix
        cm = confusion_matrix(test_labels, test_preds)

        # Save results
        results = {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "predictions": test_preds.tolist(),
            "labels": test_labels.tolist(),
            "confusion_matrix": cm.tolist(),
            "classification_report": report_dict,
        }

        self._save_results(results)
        print(f"\nEvaluation results saved to {self.output_dir}")

        return results

    def _save_results(self, results):
        """Save evaluation results to JSON."""
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save predictions separately for easier loading
        preds_path = self.output_dir / "predictions.npz"
        np.savez(
            preds_path,
            predictions=results["predictions"],
            labels=results["labels"],
            confusion_matrix=results["confusion_matrix"],
        )
