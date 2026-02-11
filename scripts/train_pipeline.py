#!/usr/bin/env python3
"""
Main training script for CV Classifier with anti data-leakage.

Features:
- Duplicate handling (split on unique texts, reinject safe duplicates)
- NLP Data Augmentation (synonym replacement, deletion, swap, insertion)
- Cross-validation on training data only
- Final evaluation on held-out test set

Usage:
    python scripts/train_pipeline.py
    python scripts/train_pipeline.py --classifier random_forest
    python scripts/train_pipeline.py --no-augmentation
    python scripts/train_pipeline.py --optimize
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import pandas as pd

from src.training.trainer import CVClassifierTrainer
from src.training.evaluator import PipelineEvaluator
from src.training.pipeline_builder import CVClassifierPipelineBuilder


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train the CV Classifier pipeline'
    )

    parser.add_argument(
        '--data', type=str,
        default='data/raw/resume_dataset.csv',
        help='Path to the raw dataset CSV'
    )

    parser.add_argument(
        '--classifier', type=str,
        default='random_forest',
        choices=CVClassifierPipelineBuilder.list_available_classifiers(),
        help='Classifier to use'
    )

    parser.add_argument(
        '--no-augmentation', action='store_true',
        help='Disable data augmentation'
    )

    parser.add_argument(
        '--no-duplicates', action='store_true',
        help='Disable duplicate handling (not recommended)'
    )

    parser.add_argument(
        '--optimize', action='store_true',
        help='Run hyperparameter optimization with GridSearchCV'
    )

    parser.add_argument(
        '--folds', type=int, default=5,
        help='Number of cross-validation folds'
    )

    parser.add_argument(
        '--output', type=str,
        default='models',
        help='Output directory for models'
    )

    parser.add_argument(
        '--splits', type=str,
        default='data/splits',
        help='Directory for train/test split indices'
    )

    parser.add_argument(
        '--fresh-split', action='store_true',
        help='Force a new train/test split even if one exists'
    )

    parser.add_argument(
        '--skip-cv', action='store_true',
        help='Skip cross-validation (faster)'
    )

    parser.add_argument(
        '--random-state', type=int, default=42,
        help='Random seed for reproducibility'
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Resolve paths
    data_path = PROJECT_ROOT / args.data
    output_dir = PROJECT_ROOT / args.output
    splits_dir = PROJECT_ROOT / args.splits

    print("=" * 70)
    print(" CV CLASSIFIER - TRAINING PIPELINE")
    print("=" * 70)
    print(f"\n Configuration:")
    print(f"   Data: {data_path}")
    print(f"   Classifier: {args.classifier}")
    print(f"   Augmentation: {not args.no_augmentation}")
    print(f"   Duplicate handling: {not args.no_duplicates}")
    print(f"   Optimization: {args.optimize}")
    print(f"   CV Folds: {args.folds}")
    print(f"   Output: {output_dir}")

    # =========================================================================
    # STEP 1: Load data
    # =========================================================================
    print("\n" + "=" * 70)
    print(" STEP 1: LOADING DATA")
    print("=" * 70)

    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"\n   Loaded {len(df)} samples")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Categories: {df['Category'].nunique()}")

    # =========================================================================
    # STEP 2: Initialize trainer
    # =========================================================================
    trainer = CVClassifierTrainer(
        classifier_name=args.classifier,
        n_folds=args.folds,
        random_state=args.random_state,
        use_augmentation=not args.no_augmentation,
        augmentation_strategy='balance'
    )

    # =========================================================================
    # STEP 3: Handle fresh split
    # =========================================================================
    if args.fresh_split and splits_dir.exists():
        import shutil
        shutil.rmtree(splits_dir)
        print(f"\n   Removed existing split: {splits_dir}")

    # =========================================================================
    # STEP 4: Prepare data (split + augmentation)
    # =========================================================================
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        df=df,
        text_column='Resume',
        target_column='Category',
        split_dir=splits_dir,
        handle_duplicates=not args.no_duplicates
    )

    # =========================================================================
    # STEP 5: Cross-validation
    # =========================================================================
    if not args.skip_cv:
        cv_results = trainer.cross_validate(X_train, y_train)
    else:
        print("\n   Cross-validation skipped (--skip-cv)")
        cv_results = None

    # =========================================================================
    # STEP 6: Training (with or without optimization)
    # =========================================================================
    if args.optimize:
        pipeline, best_params = trainer.optimize(X_train, y_train)
    else:
        pipeline = trainer.train(X_train, y_train)

    # =========================================================================
    # STEP 7: Evaluation on test set
    # =========================================================================
    evaluator = PipelineEvaluator(trainer.pipeline, trainer.label_encoder)
    test_results = evaluator.evaluate(X_test, y_test)

    # Compare CV vs Test
    comparison = None
    if cv_results is not None:
        comparison = evaluator.compare_with_cv(cv_results)

    # =========================================================================
    # STEP 8: Save everything
    # =========================================================================
    print("\n" + "=" * 70)
    print(" SAVING ARTIFACTS")
    print("=" * 70)

    saved_files = trainer.save(output_dir)

    # Save evaluation report
    reports_dir = output_dir / 'reports'
    evaluator.save_report(reports_dir, comparison)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE")
    print("=" * 70)

    print(f"\n Summary:")
    print(f"   Classifier: {args.classifier}")
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    if cv_results:
        print(f"   CV F1-Score: {cv_results['scores']['f1_macro']['cv_mean']:.4f}")

    print(f"   Test F1-Score: {test_results['metrics']['f1_macro']:.4f}")
    print(f"   Test Accuracy: {test_results['metrics']['accuracy']:.4f}")

    # Stats from trainer
    if trainer.full_stats.get('split'):
        split_stats = trainer.full_stats['split']
        if 'split' in split_stats:
            s = split_stats['split']
            if s.get('duplicates_handled'):
                print(f"\n   Duplicate handling:")
                print(f"     Safe duplicates added to train: {s.get('safe_duplicates_added', 0)}")
                print(f"     Unsafe duplicates discarded: {s.get('unsafe_duplicates_discarded', 0)}")

    if trainer.full_stats.get('augmentation'):
        aug_stats = trainer.full_stats['augmentation']
        print(f"\n   Augmentation:")
        print(f"     Original samples: {aug_stats.get('original_samples', 0)}")
        print(f"     Augmented samples: {aug_stats.get('augmented_samples', 0)}")
        print(f"     Final samples: {aug_stats.get('final_samples', 0)}")

    print(f"\n Saved files:")
    for name, path in saved_files.items():
        print(f"   - {name}: {path}")

    print("\n Pipeline ready for production use!")
    print(f"   Load with: joblib.load('{output_dir}/cv_classifier_pipeline.pkl')")

    print("\n" + "=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
