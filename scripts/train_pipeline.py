#!/usr/bin/env python3
"""
Main training script for CV Classifier with proper data leakage prevention.

This script:
1. Loads raw data
2. Splits into train/test BEFORE any preprocessing
3. Runs cross-validation on training data only
4. Trains final model on all training data
5. Evaluates once on held-out test set
6. Saves the pipeline and all artifacts

Usage:
    python scripts/train_pipeline.py
    python scripts/train_pipeline.py --classifier random_forest
    python scripts/train_pipeline.py --skip-cv  # Skip cross-validation
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.training import (
    DataSplitter,
    CVClassifierPipelineBuilder,
    CVClassifierTrainer,
    PipelineEvaluator
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train CV Classifier with proper data handling'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/raw/resume_dataset.csv',
        help='Path to raw dataset CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--split-dir',
        type=str,
        default='data/splits',
        help='Directory to save/load data split indices'
    )
    parser.add_argument(
        '--classifier',
        type=str,
        default='random_forest',
        choices=CVClassifierPipelineBuilder.list_available_classifiers(),
        help='Classifier to use'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for test set'
    )
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    parser.add_argument(
        '--skip-cv',
        action='store_true',
        help='Skip cross-validation (faster but less informative)'
    )
    parser.add_argument(
        '--force-new-split',
        action='store_true',
        help='Force creating a new train/test split even if one exists'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    print("=" * 70)
    print(" CV CLASSIFIER TRAINING - ANTI DATA LEAKAGE PIPELINE")
    print("=" * 70)

    # Resolve paths
    data_path = PROJECT_ROOT / args.data_path
    output_dir = PROJECT_ROOT / args.output_dir
    split_dir = PROJECT_ROOT / args.split_dir

    print(f"\nConfiguration:")
    print(f"  Data path:   {data_path}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Split dir:   {split_dir}")
    print(f"  Classifier:  {args.classifier}")
    print(f"  Test size:   {args.test_size}")
    print(f"  CV folds:    {args.n_folds}")
    print(f"  Random state: {args.random_state}")

    # =========================================================================
    # STEP 1: Load raw data
    # =========================================================================
    print("\n" + "=" * 70)
    print(" STEP 1: Loading raw data")
    print("=" * 70)

    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")

    # Identify text and target columns
    text_column = 'Resume'
    target_column = 'Category'

    if text_column not in df.columns:
        print(f"ERROR: Text column '{text_column}' not found")
        sys.exit(1)
    if target_column not in df.columns:
        print(f"ERROR: Target column '{target_column}' not found")
        sys.exit(1)

    print(f"\nTarget distribution:")
    print(df[target_column].value_counts())

    # =========================================================================
    # STEP 2: Split RAW data (before any preprocessing!)
    # =========================================================================
    print("\n" + "=" * 70)
    print(" STEP 2: Splitting raw data (BEFORE preprocessing)")
    print("=" * 70)

    splitter = DataSplitter(
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=True
    )

    if splitter.split_exists(split_dir) and not args.force_new_split:
        print("Using existing split...")
        train_df, test_df = splitter.load_split(df, split_dir)
    else:
        print("Creating new split...")
        train_df, test_df = splitter.split_and_save(df, target_column, split_dir)

    # Extract features and labels
    X_train = train_df[text_column].values
    y_train = train_df[target_column].values
    X_test = test_df[text_column].values
    y_test = test_df[target_column].values

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")

    # =========================================================================
    # STEP 3: Encode labels
    # =========================================================================
    print("\n" + "=" * 70)
    print(" STEP 3: Encoding labels")
    print("=" * 70)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {list(label_encoder.classes_)}")

    # =========================================================================
    # STEP 4: Cross-validation (on training data ONLY)
    # =========================================================================
    print("\n" + "=" * 70)
    print(" STEP 4: Cross-validation (training data only)")
    print("=" * 70)

    trainer = CVClassifierTrainer(
        classifier_name=args.classifier,
        n_folds=args.n_folds,
        random_state=args.random_state
    )

    if not args.skip_cv:
        cv_results = trainer.cross_validate(X_train, y_train_encoded)
    else:
        print("Cross-validation skipped (--skip-cv flag)")
        cv_results = None

    # =========================================================================
    # STEP 5: Train final model on all training data
    # =========================================================================
    print("\n" + "=" * 70)
    print(" STEP 5: Training final model")
    print("=" * 70)

    pipeline = trainer.train(X_train, y_train_encoded, label_encoder)

    # =========================================================================
    # STEP 6: Evaluate on held-out test set (ONCE!)
    # =========================================================================
    print("\n" + "=" * 70)
    print(" STEP 6: Final evaluation on test set")
    print("=" * 70)

    evaluator = PipelineEvaluator(pipeline, label_encoder)
    test_results = evaluator.evaluate(X_test, y_test_encoded)

    # Compare with CV if available
    comparison = None
    if cv_results is not None:
        comparison = evaluator.compare_with_cv(cv_results)

    # =========================================================================
    # STEP 7: Save everything
    # =========================================================================
    print("\n" + "=" * 70)
    print(" STEP 7: Saving models and reports")
    print("=" * 70)

    # Save pipeline and training artifacts
    trainer.label_encoder = label_encoder
    saved_files = trainer.save(output_dir, save_cv_results=(cv_results is not None))

    # Save evaluation report
    evaluator.save_report(output_dir, comparison)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE")
    print("=" * 70)

    print(f"\nFinal Test Metrics:")
    print(f"  Accuracy:  {test_results['metrics']['accuracy']:.4f}")
    print(f"  F1 Macro:  {test_results['metrics']['f1_macro']:.4f}")
    print(f"  Precision: {test_results['metrics']['precision_macro']:.4f}")
    print(f"  Recall:    {test_results['metrics']['recall_macro']:.4f}")

    if cv_results:
        print(f"\nCV vs Test Comparison:")
        print(f"  CV Accuracy:   {cv_results['scores']['accuracy']['cv_mean']:.4f}")
        print(f"  Test Accuracy: {test_results['metrics']['accuracy']:.4f}")

    print(f"\nSaved files:")
    for name, path in saved_files.items():
        print(f"  - {name}: {path}")

    print("\n Pipeline ready for production use!")
    print(f"   Load with: joblib.load('{output_dir}/cv_classifier_pipeline.pkl')")

    return 0


if __name__ == '__main__':
    sys.exit(main())
