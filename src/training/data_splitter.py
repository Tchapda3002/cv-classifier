"""
Data splitting utilities that ensure raw data is split BEFORE any preprocessing.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitter:
    """
    Handles the initial train/test split of raw data.
    Saves indices for reproducibility.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ):
        """
        Args:
            test_size: Proportion of data for test set
            random_state: Random seed for reproducibility
            stratify: Whether to stratify by target variable
        """
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

    def split_and_save(
        self,
        df: pd.DataFrame,
        target_column: str,
        output_dir: Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data and save indices for reproducibility.

        IMPORTANT: This should be called on RAW data before any preprocessing!

        Args:
            df: Raw dataframe
            target_column: Name of target column
            output_dir: Directory to save split indices

        Returns:
            Tuple of (train_df, test_df)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get indices
        indices = np.arange(len(df))

        # Stratify if requested
        stratify_col = df[target_column] if self.stratify else None

        # Split indices
        train_indices, test_indices = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_col
        )

        # Save indices
        with open(output_dir / 'train_indices.json', 'w') as f:
            json.dump(train_indices.tolist(), f)

        with open(output_dir / 'test_indices.json', 'w') as f:
            json.dump(test_indices.tolist(), f)

        # Save metadata
        metadata = {
            'test_size': self.test_size,
            'random_state': self.random_state,
            'stratify': self.stratify,
            'target_column': target_column,
            'total_samples': len(df),
            'train_samples': len(train_indices),
            'test_samples': len(test_indices),
            'created_at': datetime.now().isoformat(),
            'class_distribution_train': df.iloc[train_indices][target_column].value_counts().to_dict(),
            'class_distribution_test': df.iloc[test_indices][target_column].value_counts().to_dict()
        }

        with open(output_dir / 'split_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Data split completed:")
        print(f"  - Train samples: {len(train_indices)}")
        print(f"  - Test samples: {len(test_indices)}")
        print(f"  - Indices saved to: {output_dir}")

        return df.iloc[train_indices].copy(), df.iloc[test_indices].copy()

    def load_split(
        self,
        df: pd.DataFrame,
        split_dir: Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load a previously saved split.

        Args:
            df: Raw dataframe (must match original)
            split_dir: Directory containing split indices

        Returns:
            Tuple of (train_df, test_df)
        """
        split_dir = Path(split_dir)

        with open(split_dir / 'train_indices.json', 'r') as f:
            train_indices = json.load(f)

        with open(split_dir / 'test_indices.json', 'r') as f:
            test_indices = json.load(f)

        # Verify
        metadata = self.get_split_metadata(split_dir)
        if len(df) != metadata['total_samples']:
            raise ValueError(
                f"DataFrame size ({len(df)}) doesn't match saved split ({metadata['total_samples']})"
            )

        print(f"Loaded existing split from: {split_dir}")
        print(f"  - Train samples: {len(train_indices)}")
        print(f"  - Test samples: {len(test_indices)}")

        return df.iloc[train_indices].copy(), df.iloc[test_indices].copy()

    def get_split_metadata(self, split_dir: Path) -> Dict[str, Any]:
        """Load and return split metadata."""
        split_dir = Path(split_dir)
        with open(split_dir / 'split_metadata.json', 'r') as f:
            return json.load(f)

    def split_exists(self, split_dir: Path) -> bool:
        """Check if a split already exists."""
        split_dir = Path(split_dir)
        return (
            (split_dir / 'train_indices.json').exists() and
            (split_dir / 'test_indices.json').exists() and
            (split_dir / 'split_metadata.json').exists()
        )
