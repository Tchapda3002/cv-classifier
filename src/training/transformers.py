"""
Custom sklearn transformers for the CV Classifier pipeline.
These transformers wrap existing preprocessing logic in sklearn-compatible classes.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Import the existing TextCleaner
import sys
from pathlib import Path

# Add src to path if needed
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from preprocessing.text_cleaner import TextCleaner


class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible wrapper around TextCleaner.

    This is a STATELESS transformer - fit() does nothing,
    and transform() applies the same cleaning to any data.
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_phone_numbers: bool = True,
        remove_numbers: bool = False,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        stem: bool = False
    ):
        """
        Initialize with the same parameters as TextCleaner.

        Args:
            lowercase: Convert to lowercase
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            remove_phone_numbers: Remove phone numbers
            remove_numbers: Remove numbers (default False to keep years of experience)
            remove_punctuation: Remove punctuation
            remove_stopwords: Remove English stopwords
            lemmatize: Apply lemmatization
            stem: Apply stemming (if True, overrides lemmatize)
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_phone_numbers = remove_phone_numbers
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stem = stem

        # Create the underlying cleaner
        self._cleaner = None

    def _get_cleaner(self) -> TextCleaner:
        """Lazily create the TextCleaner instance."""
        if self._cleaner is None:
            self._cleaner = TextCleaner(
                lowercase=self.lowercase,
                remove_urls=self.remove_urls,
                remove_emails=self.remove_emails,
                remove_phone_numbers=self.remove_phone_numbers,
                remove_numbers=self.remove_numbers,
                remove_punctuation=self.remove_punctuation,
                remove_stopwords=self.remove_stopwords,
                lemmatize=self.lemmatize,
                stem=self.stem
            )
        return self._cleaner

    def fit(self, X, y=None):
        """
        Fit method - does nothing as this is a stateless transformer.

        Args:
            X: Input data (ignored for fitting)
            y: Target labels (ignored)

        Returns:
            self
        """
        # Stateless - nothing to learn
        return self

    def transform(self, X) -> np.ndarray:
        """
        Apply text cleaning to all input texts.

        Args:
            X: Array-like of strings (raw CV texts)

        Returns:
            Array of cleaned strings
        """
        cleaner = self._get_cleaner()

        # Handle different input types
        if isinstance(X, pd.DataFrame):
            # Assume first column or 'Resume' column
            if 'Resume' in X.columns:
                texts = X['Resume'].values
            else:
                texts = X.iloc[:, 0].values
        elif isinstance(X, pd.Series):
            texts = X.values
        else:
            texts = np.asarray(X).ravel()

        # Apply cleaning to each text
        cleaned = [cleaner.clean_text(str(text)) for text in texts]

        return np.array(cleaned)

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Simple transformer to select a specific column from a DataFrame.
    Useful as the first step in a pipeline that receives DataFrames.
    """

    def __init__(self, column: str = 'Resume'):
        """
        Args:
            column: Name of the column to select
        """
        self.column = column

    def fit(self, X, y=None):
        """Stateless - nothing to fit."""
        return self

    def transform(self, X) -> np.ndarray:
        """
        Select the specified column.

        Args:
            X: DataFrame or array-like

        Returns:
            Array of values from the selected column
        """
        if isinstance(X, pd.DataFrame):
            if self.column in X.columns:
                return X[self.column].values
            else:
                raise ValueError(f"Column '{self.column}' not found in DataFrame")
        elif isinstance(X, pd.Series):
            return X.values
        else:
            # Already an array, return as-is
            return np.asarray(X).ravel()


class DebugTransformer(BaseEstimator, TransformerMixin):
    """
    Debug transformer that prints information about data passing through.
    Useful for pipeline debugging - remove in production.
    """

    def __init__(self, name: str = "debug", verbose: bool = True):
        self.name = name
        self.verbose = verbose

    def fit(self, X, y=None):
        if self.verbose:
            print(f"[{self.name}] fit() called")
            print(f"  - X type: {type(X)}")
            print(f"  - X shape: {getattr(X, 'shape', len(X))}")
            if y is not None:
                print(f"  - y shape: {getattr(y, 'shape', len(y))}")
        return self

    def transform(self, X):
        if self.verbose:
            print(f"[{self.name}] transform() called")
            print(f"  - X type: {type(X)}")
            print(f"  - X shape: {getattr(X, 'shape', len(X))}")
            if hasattr(X, 'dtype'):
                print(f"  - X dtype: {X.dtype}")
        return X
