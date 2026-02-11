"""
Data splitting utilities with duplicate handling for anti data-leakage.

Strategy:
1. Detect and remove duplicates
2. Split on UNIQUE data only
3. Reinject "safe" duplicates (those whose original is in train, not test)
4. Save indices for reproducibility
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitter:
    """
    Handles the initial train/test split with duplicate management.
    Ensures no data leakage by:
    - Splitting on unique CV texts only
    - Only adding duplicates back to train if their original is in train
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        handle_duplicates: bool = True
    ):
        """
        Args:
            test_size: Proportion of data for test set
            random_state: Random seed for reproducibility
            stratify: Whether to stratify by target variable
            handle_duplicates: Whether to detect and handle duplicates
        """
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        self.handle_duplicates = handle_duplicates

        # Statistics
        self.stats: Dict[str, Any] = {}

    def _analyze_duplicates(
        self,
        df: pd.DataFrame,
        text_column: str
    ) -> Dict[str, Any]:
        """
        Analyze duplicates in the dataset.

        Args:
            df: DataFrame with text data
            text_column: Name of column containing text

        Returns:
            Dictionary with duplicate analysis
        """
        # Detect duplicates
        duplicates_mask = df.duplicated(subset=[text_column], keep=False)
        n_duplicates_total = duplicates_mask.sum()
        n_unique_duplicated = df[duplicates_mask][text_column].nunique()

        # Count copies per text
        text_counts = df[text_column].value_counts()
        duplicated_texts = text_counts[text_counts > 1]

        # Distribution of copies
        copy_distribution = {}
        if len(duplicated_texts) > 0:
            copy_distribution = duplicated_texts.value_counts().to_dict()

        analysis = {
            'total_rows': len(df),
            'unique_texts': df[text_column].nunique(),
            'rows_with_duplicates': int(n_duplicates_total),
            'unique_texts_that_have_duplicates': int(n_unique_duplicated),
            'copy_distribution': {str(k): int(v) for k, v in copy_distribution.items()}
        }

        return analysis

    def split_and_save(
        self,
        df: pd.DataFrame,
        text_column: str,
        target_column: str,
        output_dir: Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data with duplicate handling and save indices.

        IMPORTANT: This should be called on RAW data before any preprocessing!

        Args:
            df: Raw dataframe
            text_column: Name of text column (e.g., 'Resume')
            target_column: Name of target column (e.g., 'Category')
            output_dir: Directory to save split indices

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(" DATA SPLITTING WITH DUPLICATE HANDLING")
        print("=" * 60)

        # Step 1: Analyze duplicates
        print("\n1. Analyzing duplicates...")
        dup_analysis = self._analyze_duplicates(df, text_column)
        self.stats['duplicate_analysis'] = dup_analysis

        print(f"   Total rows: {dup_analysis['total_rows']}")
        print(f"   Unique texts: {dup_analysis['unique_texts']}")
        print(f"   Rows with duplicates: {dup_analysis['rows_with_duplicates']}")

        if self.handle_duplicates and dup_analysis['rows_with_duplicates'] > 0:
            return self._split_with_duplicate_handling(
                df, text_column, target_column, output_dir
            )
        else:
            return self._split_simple(
                df, text_column, target_column, output_dir
            )

    def _split_simple(
        self,
        df: pd.DataFrame,
        text_column: str,
        target_column: str,
        output_dir: Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simple split without duplicate handling."""
        print("\n2. Performing simple split (no duplicates to handle)...")

        X = df[text_column].values
        y = df[target_column].values

        stratify_col = y if self.stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_col
        )

        # Save indices (as positions)
        train_indices = list(range(len(X_train)))
        test_indices = list(range(len(X_test)))

        self._save_indices(
            output_dir, train_indices, test_indices,
            df, text_column, target_column
        )

        self.stats['split'] = {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'duplicates_handled': False
        }

        print(f"\n   Train: {len(X_train)}")
        print(f"   Test: {len(X_test)}")

        return X_train, X_test, y_train, y_test

    def _split_with_duplicate_handling(
        self,
        df: pd.DataFrame,
        text_column: str,
        target_column: str,
        output_dir: Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split with duplicate handling:
        1. Get unique texts
        2. Split unique texts
        3. Add safe duplicates back to train
        """
        print("\n2. Splitting on UNIQUE texts only...")

        # Get unique texts (keep first occurrence)
        df_unique = df.drop_duplicates(subset=[text_column], keep='first').reset_index(drop=True)

        # Get duplicates (all copies except first)
        df_duplicates = df[df.duplicated(subset=[text_column], keep='first')].reset_index(drop=True)

        print(f"   Unique texts: {len(df_unique)}")
        print(f"   Duplicates removed: {len(df_duplicates)}")

        # Split unique texts
        X_unique = df_unique[text_column].values
        y_unique = df_unique[target_column].values

        stratify_col = y_unique if self.stratify else None

        X_train_unique, X_test, y_train_unique, y_test = train_test_split(
            X_unique, y_unique,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_col
        )

        print(f"\n3. Initial split:")
        print(f"   Train (unique): {len(X_train_unique)}")
        print(f"   Test (unique): {len(X_test)}")

        # Create test set for fast lookup
        test_set = set(X_test)

        # Find safe duplicates (not in test set)
        print("\n4. Reinjecting safe duplicates into train...")
        safe_duplicates_X = []
        safe_duplicates_y = []
        unsafe_count = 0

        for idx, row in df_duplicates.iterrows():
            text = row[text_column]
            label = row[target_column]

            if text not in test_set:
                safe_duplicates_X.append(text)
                safe_duplicates_y.append(label)
            else:
                unsafe_count += 1

        print(f"   Safe duplicates (added to train): {len(safe_duplicates_X)}")
        print(f"   Unsafe duplicates (discarded): {unsafe_count}")

        # Combine train + safe duplicates
        if len(safe_duplicates_X) > 0:
            X_train = np.concatenate([X_train_unique, np.array(safe_duplicates_X)])
            y_train = np.concatenate([y_train_unique, np.array(safe_duplicates_y)])
        else:
            X_train = X_train_unique
            y_train = y_train_unique

        # Verify no leakage
        print("\n5. Verifying anti data-leakage...")
        train_set = set(X_train)
        overlap = train_set.intersection(test_set)

        if len(overlap) > 0:
            raise ValueError(f"DATA LEAKAGE DETECTED: {len(overlap)} texts in both train and test!")

        print("   SUCCESS: No texts in common between train and test!")

        # Save indices and metadata
        self._save_split_data(
            output_dir, X_train, X_test, y_train, y_test,
            df, text_column, target_column,
            safe_duplicates=len(safe_duplicates_X),
            unsafe_duplicates=unsafe_count
        )

        self.stats['split'] = {
            'train_unique': len(X_train_unique),
            'train_with_duplicates': len(X_train),
            'safe_duplicates_added': len(safe_duplicates_X),
            'unsafe_duplicates_discarded': unsafe_count,
            'test_size': len(X_test),
            'duplicates_handled': True
        }

        print(f"\n   Final Train: {len(X_train)}")
        print(f"   Final Test: {len(X_test)}")

        return X_train, X_test, y_train, y_test

    def _save_split_data(
        self,
        output_dir: Path,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        df: pd.DataFrame,
        text_column: str,
        target_column: str,
        safe_duplicates: int = 0,
        unsafe_duplicates: int = 0
    ):
        """Save split data and metadata."""
        output_dir = Path(output_dir)

        # Save test indices (based on original unique df for reproducibility)
        df_unique = df.drop_duplicates(subset=[text_column], keep='first')
        test_texts_set = set(X_test)
        test_indices = [i for i, text in enumerate(df_unique[text_column]) if text in test_texts_set]

        with open(output_dir / 'test_indices.json', 'w') as f:
            json.dump(test_indices, f)

        # Save metadata
        metadata = {
            'test_size': self.test_size,
            'random_state': self.random_state,
            'stratify': self.stratify,
            'handle_duplicates': self.handle_duplicates,
            'text_column': text_column,
            'target_column': target_column,
            'original_total_samples': len(df),
            'unique_samples': len(df_unique),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'safe_duplicates_added': safe_duplicates,
            'unsafe_duplicates_discarded': unsafe_duplicates,
            'created_at': datetime.now().isoformat(),
            'class_distribution_train': pd.Series(y_train).value_counts().to_dict(),
            'class_distribution_test': pd.Series(y_test).value_counts().to_dict()
        }

        with open(output_dir / 'split_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"\n   Metadata saved to: {output_dir}")

    def _save_indices(
        self,
        output_dir: Path,
        train_indices: List[int],
        test_indices: List[int],
        df: pd.DataFrame,
        text_column: str,
        target_column: str
    ):
        """Save simple indices."""
        output_dir = Path(output_dir)

        with open(output_dir / 'train_indices.json', 'w') as f:
            json.dump(train_indices, f)

        with open(output_dir / 'test_indices.json', 'w') as f:
            json.dump(test_indices, f)

        metadata = {
            'test_size': self.test_size,
            'random_state': self.random_state,
            'stratify': self.stratify,
            'handle_duplicates': self.handle_duplicates,
            'total_samples': len(df),
            'train_samples': len(train_indices),
            'test_samples': len(test_indices),
            'created_at': datetime.now().isoformat()
        }

        with open(output_dir / 'split_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_split(
        self,
        df: pd.DataFrame,
        text_column: str,
        target_column: str,
        split_dir: Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load a previously saved split.

        Args:
            df: Raw dataframe (must match original)
            text_column: Name of text column
            target_column: Name of target column
            split_dir: Directory containing split indices

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        split_dir = Path(split_dir)

        # Load metadata
        with open(split_dir / 'split_metadata.json', 'r') as f:
            metadata = json.load(f)

        # Load test indices
        with open(split_dir / 'test_indices.json', 'r') as f:
            test_indices = json.load(f)

        print(f"Loading existing split from: {split_dir}")

        if metadata.get('handle_duplicates', False):
            # Reconstruct split with duplicate handling
            df_unique = df.drop_duplicates(subset=[text_column], keep='first').reset_index(drop=True)
            df_duplicates = df[df.duplicated(subset=[text_column], keep='first')].reset_index(drop=True)

            # Get test data
            X_test = df_unique.iloc[test_indices][text_column].values
            y_test = df_unique.iloc[test_indices][target_column].values
            test_set = set(X_test)

            # Get train data (all unique not in test + safe duplicates)
            train_mask = ~df_unique.index.isin(test_indices)
            X_train_unique = df_unique[train_mask][text_column].values
            y_train_unique = df_unique[train_mask][target_column].values

            # Add safe duplicates
            safe_X = []
            safe_y = []
            for _, row in df_duplicates.iterrows():
                if row[text_column] not in test_set:
                    safe_X.append(row[text_column])
                    safe_y.append(row[target_column])

            if len(safe_X) > 0:
                X_train = np.concatenate([X_train_unique, np.array(safe_X)])
                y_train = np.concatenate([y_train_unique, np.array(safe_y)])
            else:
                X_train = X_train_unique
                y_train = y_train_unique
        else:
            # Simple load
            with open(split_dir / 'train_indices.json', 'r') as f:
                train_indices = json.load(f)

            X_train = df.iloc[train_indices][text_column].values
            y_train = df.iloc[train_indices][target_column].values
            X_test = df.iloc[test_indices][text_column].values
            y_test = df.iloc[test_indices][target_column].values

        print(f"  Train samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")

        return X_train, X_test, y_train, y_test

    def get_split_metadata(self, split_dir: Path) -> Dict[str, Any]:
        """Load and return split metadata."""
        split_dir = Path(split_dir)
        with open(split_dir / 'split_metadata.json', 'r') as f:
            return json.load(f)

    def split_exists(self, split_dir: Path) -> bool:
        """Check if a split already exists."""
        split_dir = Path(split_dir)
        return (
            (split_dir / 'test_indices.json').exists() and
            (split_dir / 'split_metadata.json').exists()
        )
