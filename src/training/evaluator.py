"""
Evaluator module for CV Classifier.
Handles final evaluation on the held-out test set.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


class PipelineEvaluator:
    """
    Evaluates a trained pipeline on the test set.

    IMPORTANT: This should only be called ONCE at the very end,
    after all model selection and tuning is complete!
    """

    def __init__(
        self,
        pipeline: Pipeline,
        label_encoder: LabelEncoder
    ):
        """
        Initialize with a trained pipeline.

        Args:
            pipeline: Trained sklearn Pipeline
            label_encoder: Fitted LabelEncoder
        """
        self.pipeline = pipeline
        self.label_encoder = label_encoder
        self.results: Optional[Dict[str, Any]] = None

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate the pipeline on test data.

        Args:
            X_test: Test texts (raw, uncleaned)
            y_test: True labels (encoded or strings)
            return_predictions: Whether to include predictions in results

        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\nEvaluating on test set ({len(X_test)} samples)...")

        # Handle label encoding
        if isinstance(y_test[0], str):
            y_true = self.label_encoder.transform(y_test)
        else:
            y_true = y_test

        # Get predictions
        y_pred = self.pipeline.predict(X_test)

        # Calculate metrics
        results = {
            'n_samples': len(X_test),
            'n_classes': len(self.label_encoder.classes_),
            'evaluated_at': datetime.now().isoformat(),
            'metrics': {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
                'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
                'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
                'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            }
        }

        # Per-class metrics
        class_names = list(self.label_encoder.classes_)
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        results['per_class'] = {}
        for i, class_name in enumerate(class_names):
            results['per_class'][class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i])
            }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = {
            'matrix': cm.tolist(),
            'labels': class_names
        }

        # Classification report (as text)
        results['classification_report'] = classification_report(
            y_true, y_pred,
            target_names=class_names,
            zero_division=0
        )

        # Optionally include predictions
        if return_predictions:
            results['predictions'] = {
                'y_true': y_true.tolist() if hasattr(y_true, 'tolist') else list(y_true),
                'y_pred': y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred),
                'y_true_labels': [class_names[i] for i in y_true],
                'y_pred_labels': [class_names[i] for i in y_pred]
            }

        self.results = results

        # Print summary
        print("\n" + "=" * 60)
        print(" TEST SET EVALUATION RESULTS")
        print("=" * 60)
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {results['metrics']['accuracy']:.4f}")
        print(f"  F1 Macro:  {results['metrics']['f1_macro']:.4f}")
        print(f"  Precision: {results['metrics']['precision_macro']:.4f}")
        print(f"  Recall:    {results['metrics']['recall_macro']:.4f}")
        print("\nClassification Report:")
        print(results['classification_report'])

        return results

    def compare_with_cv(
        self,
        cv_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare test results with cross-validation results.

        This helps detect overfitting: if CV scores are much higher
        than test scores, there may be data leakage or overfitting.

        Args:
            cv_results: Results from cross-validation

        Returns:
            Comparison dictionary
        """
        if self.results is None:
            raise ValueError("Run evaluate() first")

        comparison = {
            'cv_accuracy': cv_results['scores']['accuracy']['cv_mean'],
            'test_accuracy': self.results['metrics']['accuracy'],
            'accuracy_diff': cv_results['scores']['accuracy']['cv_mean'] - self.results['metrics']['accuracy'],
            'cv_f1': cv_results['scores']['f1_macro']['cv_mean'],
            'test_f1': self.results['metrics']['f1_macro'],
            'f1_diff': cv_results['scores']['f1_macro']['cv_mean'] - self.results['metrics']['f1_macro'],
        }

        print("\n" + "=" * 60)
        print(" CV vs TEST COMPARISON")
        print("=" * 60)
        print(f"\nAccuracy:")
        print(f"  CV:   {comparison['cv_accuracy']:.4f}")
        print(f"  Test: {comparison['test_accuracy']:.4f}")
        print(f"  Diff: {comparison['accuracy_diff']:+.4f}")

        print(f"\nF1 Macro:")
        print(f"  CV:   {comparison['cv_f1']:.4f}")
        print(f"  Test: {comparison['test_f1']:.4f}")
        print(f"  Diff: {comparison['f1_diff']:+.4f}")

        # Warning if large discrepancy
        if abs(comparison['accuracy_diff']) > 0.05:
            print("\n WARNING: Large discrepancy between CV and test scores!")
            print("   This might indicate overfitting or data leakage.")
        else:
            print("\n CV and test scores are consistent - good sign!")

        return comparison

    def save_report(
        self,
        output_dir: Path,
        comparison: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save evaluation report to file.

        Args:
            output_dir: Directory to save report
            comparison: Optional CV comparison results

        Returns:
            Path to saved report
        """
        if self.results is None:
            raise ValueError("Run evaluate() first")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare full report
        report = {
            'evaluation': self.results,
            'comparison': comparison
        }

        # Save JSON report
        json_path = output_dir / 'test_evaluation.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nEvaluation report saved to: {json_path}")

        # Save text report
        txt_path = output_dir / 'test_evaluation_report.txt'
        with open(txt_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(" CV CLASSIFIER - TEST SET EVALUATION\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Evaluated at: {self.results['evaluated_at']}\n")
            f.write(f"Test samples: {self.results['n_samples']}\n")
            f.write(f"Number of classes: {self.results['n_classes']}\n\n")

            f.write("OVERALL METRICS\n")
            f.write("-" * 40 + "\n")
            for metric, value in self.results['metrics'].items():
                f.write(f"  {metric}: {value:.4f}\n")

            f.write("\n\nCLASSIFICATION REPORT\n")
            f.write("-" * 40 + "\n")
            f.write(self.results['classification_report'])

            if comparison:
                f.write("\n\nCV vs TEST COMPARISON\n")
                f.write("-" * 40 + "\n")
                f.write(f"  CV Accuracy:   {comparison['cv_accuracy']:.4f}\n")
                f.write(f"  Test Accuracy: {comparison['test_accuracy']:.4f}\n")
                f.write(f"  Difference:    {comparison['accuracy_diff']:+.4f}\n")

        print(f"Text report saved to: {txt_path}")

        return json_path

    def get_misclassified(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_samples: int = 10
    ) -> pd.DataFrame:
        """
        Get examples of misclassified samples for error analysis.

        Args:
            X_test: Test texts
            y_test: True labels
            n_samples: Number of misclassified samples to return

        Returns:
            DataFrame with misclassified examples
        """
        # Handle label encoding
        if isinstance(y_test[0], str):
            y_true = self.label_encoder.transform(y_test)
            y_true_labels = y_test
        else:
            y_true = y_test
            y_true_labels = self.label_encoder.inverse_transform(y_test)

        y_pred = self.pipeline.predict(X_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)

        # Find misclassified indices
        misclassified_mask = y_true != y_pred
        misclassified_indices = np.where(misclassified_mask)[0]

        # Sample if too many
        if len(misclassified_indices) > n_samples:
            misclassified_indices = np.random.choice(
                misclassified_indices, n_samples, replace=False
            )

        # Create DataFrame
        df = pd.DataFrame({
            'text': [X_test[i][:200] + '...' if len(X_test[i]) > 200 else X_test[i]
                     for i in misclassified_indices],
            'true_label': [y_true_labels[i] for i in misclassified_indices],
            'predicted_label': [y_pred_labels[i] for i in misclassified_indices]
        })

        return df
