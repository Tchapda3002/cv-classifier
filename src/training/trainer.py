"""
Trainer module for CV Classifier.
Handles data splitting, augmentation, cross-validation, and final model training.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import joblib

from .pipeline_builder import CVClassifierPipelineBuilder
from .data_splitter import DataSplitter
from .augmentation import TextAugmenter


class CVClassifierTrainer:
    """
    Complete trainer class that handles:
    - Data loading and splitting (with duplicate handling)
    - Data augmentation
    - Cross-validation on training data
    - Final model training
    - Model and artifact saving
    """

    def __init__(
        self,
        classifier_name: str = 'random_forest',
        n_folds: int = 5,
        random_state: int = 42,
        tfidf_params: Optional[Dict[str, Any]] = None,
        cleaner_params: Optional[Dict[str, Any]] = None,
        classifier_params: Optional[Dict[str, Any]] = None,
        use_augmentation: bool = True,
        augmentation_strategy: str = 'balance'
    ):
        """
        Initialize the trainer.

        Args:
            classifier_name: Name of the classifier to use
            n_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            tfidf_params: Override TF-IDF parameters
            cleaner_params: Override text cleaner parameters
            classifier_params: Override classifier parameters
            use_augmentation: Whether to apply data augmentation
            augmentation_strategy: 'balance' or 'multiply'
        """
        self.classifier_name = classifier_name
        self.n_folds = n_folds
        self.random_state = random_state
        self.tfidf_params = tfidf_params
        self.cleaner_params = cleaner_params
        self.classifier_params = classifier_params
        self.use_augmentation = use_augmentation
        self.augmentation_strategy = augmentation_strategy

        self.pipeline: Optional[Pipeline] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.cv_results: Optional[Dict[str, Any]] = None
        self.training_time: float = 0.0
        self.full_stats: Dict[str, Any] = {}

    def _build_pipeline(self) -> Pipeline:
        """Build a fresh pipeline instance."""
        builder = CVClassifierPipelineBuilder(
            classifier_name=self.classifier_name,
            tfidf_params=self.tfidf_params,
            cleaner_params=self.cleaner_params,
            classifier_params=self.classifier_params
        )
        return builder.build()

    def prepare_data(
        self,
        df: pd.DataFrame,
        text_column: str = 'Resume',
        target_column: str = 'Category',
        split_dir: Optional[Path] = None,
        test_size: float = 0.2,
        handle_duplicates: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data with splitting and optional augmentation.

        Args:
            df: Raw dataframe
            text_column: Name of text column
            target_column: Name of target column
            split_dir: Directory to save/load split indices
            test_size: Proportion for test set
            handle_duplicates: Whether to handle duplicates

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Initialize splitter
        splitter = DataSplitter(
            test_size=test_size,
            random_state=self.random_state,
            stratify=True,
            handle_duplicates=handle_duplicates
        )

        # Split data
        if split_dir is not None and splitter.split_exists(split_dir):
            X_train, X_test, y_train, y_test = splitter.load_split(
                df, text_column, target_column, split_dir
            )
        else:
            if split_dir is None:
                split_dir = Path('data/splits')
            X_train, X_test, y_train, y_test = splitter.split_and_save(
                df, text_column, target_column, split_dir
            )

        self.full_stats['split'] = splitter.stats

        # Apply augmentation if enabled
        if self.use_augmentation:
            augmenter = TextAugmenter(random_state=self.random_state)
            X_train, y_train = augmenter.augment_dataset(
                X_train, y_train,
                strategy=self.augmentation_strategy
            )
            self.full_stats['augmentation'] = augmenter.stats

        return X_train, X_test, y_train, y_test

    def cross_validate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        scoring: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on the training data.

        IMPORTANT: This should only be called with training data,
        never with test data!

        Args:
            X_train: Training texts (raw, uncleaned)
            y_train: Training labels (encoded or strings)
            scoring: List of scoring metrics

        Returns:
            Dictionary of CV results
        """
        if scoring is None:
            scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

        # Encode labels if needed
        if isinstance(y_train[0], str):
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y_train)
            else:
                y_encoded = self.label_encoder.transform(y_train)
        else:
            y_encoded = y_train

        print(f"\n{'=' * 60}")
        print(f" CROSS-VALIDATION ({self.n_folds} folds)")
        print(f"{'=' * 60}")
        print(f"\n   Classifier: {self.classifier_name}")
        print(f"   Training samples: {len(X_train)}")

        # Create CV splitter
        cv = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )

        # Build pipeline for CV
        pipeline = self._build_pipeline()

        # Run cross-validation
        start_time = time.time()
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_encoded,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        cv_time = time.time() - start_time

        # Summarize results
        results = {
            'classifier': self.classifier_name,
            'n_folds': self.n_folds,
            'cv_time_seconds': cv_time,
            'n_samples': len(X_train),
            'scores': {}
        }

        for metric in scoring:
            test_key = f'test_{metric}'
            train_key = f'train_{metric}'

            results['scores'][metric] = {
                'cv_mean': float(np.mean(cv_scores[test_key])),
                'cv_std': float(np.std(cv_scores[test_key])),
                'cv_scores': cv_scores[test_key].tolist(),
                'train_mean': float(np.mean(cv_scores[train_key])),
                'train_std': float(np.std(cv_scores[train_key]))
            }

        self.cv_results = results

        # Print summary
        print(f"\n   Completed in {cv_time:.1f}s")
        print(f"\n   CV Results:")
        for metric, scores in results['scores'].items():
            print(f"     {metric}:")
            print(f"       CV:    {scores['cv_mean']:.4f} (+/- {scores['cv_std']:.4f})")
            print(f"       Train: {scores['train_mean']:.4f} (+/- {scores['train_std']:.4f})")

        return results

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        label_encoder: Optional[LabelEncoder] = None
    ) -> Pipeline:
        """
        Train the final model on all training data.

        Args:
            X_train: Training texts (raw, uncleaned)
            y_train: Training labels (can be strings or encoded)
            label_encoder: Optional pre-fitted label encoder

        Returns:
            Trained pipeline
        """
        print(f"\n{'=' * 60}")
        print(f" FINAL MODEL TRAINING")
        print(f"{'=' * 60}")
        print(f"\n   Classifier: {self.classifier_name}")
        print(f"   Training samples: {len(X_train)}")

        # Handle label encoding
        if label_encoder is not None:
            self.label_encoder = label_encoder
            y_encoded = y_train if not isinstance(y_train[0], str) else label_encoder.transform(y_train)
        elif isinstance(y_train[0], str):
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y_train)
            else:
                y_encoded = self.label_encoder.transform(y_train)
        else:
            y_encoded = y_train

        # Build and train pipeline
        self.pipeline = self._build_pipeline()

        start_time = time.time()
        self.pipeline.fit(X_train, y_encoded)
        self.training_time = time.time() - start_time

        print(f"\n   Training completed in {self.training_time:.1f}s")
        print(f"   Classes: {len(self.label_encoder.classes_)}")

        return self.pipeline

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Optional[Dict[str, list]] = None,
        n_jobs: int = -1
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        """
        Optimize hyperparameters using GridSearchCV.

        Args:
            X_train: Training texts
            y_train: Training labels
            param_grid: Parameter grid (uses default if None)
            n_jobs: Number of parallel jobs

        Returns:
            Tuple of (best_pipeline, best_params)
        """
        print(f"\n{'=' * 60}")
        print(f" HYPERPARAMETER OPTIMIZATION")
        print(f"{'=' * 60}")

        # Encode labels if needed
        if isinstance(y_train[0], str):
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y_train)
            else:
                y_encoded = self.label_encoder.transform(y_train)
        else:
            y_encoded = y_train

        # Get default param grid if not provided
        if param_grid is None:
            param_grid = CVClassifierPipelineBuilder.get_param_grid(self.classifier_name)

        print(f"\n   Parameter grid: {param_grid}")

        # Build base pipeline
        pipeline = self._build_pipeline()

        # GridSearchCV
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=self.n_folds,
            scoring='f1_macro',
            n_jobs=n_jobs,
            verbose=1
        )

        start_time = time.time()
        grid_search.fit(X_train, y_encoded)
        optimization_time = time.time() - start_time

        print(f"\n   Optimization completed in {optimization_time:.1f}s")
        print(f"   Best score: {grid_search.best_score_:.4f}")
        print(f"   Best params: {grid_search.best_params_}")

        self.pipeline = grid_search.best_estimator_
        self.training_time = optimization_time

        return self.pipeline, grid_search.best_params_

    def save(
        self,
        output_dir: Path,
        save_cv_results: bool = True
    ) -> Dict[str, Path]:
        """
        Save the trained pipeline and artifacts.

        Args:
            output_dir: Directory to save models
            save_cv_results: Whether to save CV results

        Returns:
            Dictionary of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Save pipeline
        if self.pipeline is not None:
            pipeline_path = output_dir / 'cv_classifier_pipeline.pkl'
            joblib.dump(self.pipeline, pipeline_path)
            saved_files['pipeline'] = pipeline_path
            print(f"Pipeline saved to: {pipeline_path}")

        # Save label encoder
        if self.label_encoder is not None:
            encoder_path = output_dir / 'label_encoder.pkl'
            joblib.dump(self.label_encoder, encoder_path)
            saved_files['label_encoder'] = encoder_path
            print(f"Label encoder saved to: {encoder_path}")

        # Save CV results
        if save_cv_results and self.cv_results is not None:
            cv_path = output_dir / 'cv_results.json'
            with open(cv_path, 'w') as f:
                json.dump(self.cv_results, f, indent=2)
            saved_files['cv_results'] = cv_path
            print(f"CV results saved to: {cv_path}")

        # Save training metadata
        metadata = {
            'classifier': self.classifier_name,
            'training_time_seconds': self.training_time,
            'trained_at': datetime.now().isoformat(),
            'n_classes': len(self.label_encoder.classes_) if self.label_encoder else None,
            'classes': list(self.label_encoder.classes_) if self.label_encoder else None,
            'pipeline_steps': [name for name, _ in self.pipeline.steps] if self.pipeline else None,
            'use_augmentation': self.use_augmentation,
            'augmentation_strategy': self.augmentation_strategy,
            'full_stats': self.full_stats
        }

        metadata_path = output_dir / 'training_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        saved_files['metadata'] = metadata_path
        print(f"Metadata saved to: {metadata_path}")

        return saved_files

    @classmethod
    def load(cls, model_dir: Path) -> Tuple[Pipeline, LabelEncoder]:
        """
        Load a trained pipeline and label encoder.

        Args:
            model_dir: Directory containing saved models

        Returns:
            Tuple of (pipeline, label_encoder)
        """
        model_dir = Path(model_dir)

        pipeline_path = model_dir / 'cv_classifier_pipeline.pkl'
        encoder_path = model_dir / 'label_encoder.pkl'

        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline not found at: {pipeline_path}")
        if not encoder_path.exists():
            raise FileNotFoundError(f"Label encoder not found at: {encoder_path}")

        pipeline = joblib.load(pipeline_path)
        label_encoder = joblib.load(encoder_path)

        print(f"Loaded pipeline from: {pipeline_path}")
        print(f"Loaded label encoder from: {encoder_path}")

        return pipeline, label_encoder
