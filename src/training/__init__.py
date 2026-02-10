"""
Training module for CV Classifier with proper data leakage prevention.

This module provides:
- DataSplitter: Split raw data before any preprocessing
- TextCleanerTransformer: sklearn-compatible text cleaning
- CVClassifierPipelineBuilder: Build complete sklearn Pipelines
- CVClassifierTrainer: Cross-validation and training
- PipelineEvaluator: Final evaluation on test set
"""

from .data_splitter import DataSplitter
from .transformers import TextCleanerTransformer, ColumnSelector
from .pipeline_builder import CVClassifierPipelineBuilder
from .trainer import CVClassifierTrainer
from .evaluator import PipelineEvaluator

__all__ = [
    'DataSplitter',
    'TextCleanerTransformer',
    'ColumnSelector',
    'CVClassifierPipelineBuilder',
    'CVClassifierTrainer',
    'PipelineEvaluator'
]
