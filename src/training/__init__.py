"""
Training module for CV Classifier with anti data-leakage and augmentation.

This module provides:
- DataSplitter: Split raw data with duplicate handling
- TextAugmenter: NLP-based data augmentation
- TextCleanerTransformer: sklearn-compatible text cleaning
- CVClassifierPipelineBuilder: Build complete sklearn Pipelines
- CVClassifierTrainer: Cross-validation and training
- PipelineEvaluator: Final evaluation on test set
"""

from .data_splitter import DataSplitter
from .augmentation import TextAugmenter
from .transformers import TextCleanerTransformer, ColumnSelector
from .pipeline_builder import CVClassifierPipelineBuilder
from .trainer import CVClassifierTrainer
from .evaluator import PipelineEvaluator

__all__ = [
    'DataSplitter',
    'TextAugmenter',
    'TextCleanerTransformer',
    'ColumnSelector',
    'CVClassifierPipelineBuilder',
    'CVClassifierTrainer',
    'PipelineEvaluator'
]
