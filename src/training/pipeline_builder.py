"""
Pipeline builder for the CV Classifier.
Constructs sklearn Pipelines that encapsulate all transformations.
"""

from typing import Dict, Any, Optional, Type
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from .transformers import TextCleanerTransformer


# Available classifiers with their default parameters
CLASSIFIERS: Dict[str, tuple] = {
    'random_forest': (
        RandomForestClassifier,
        {
            'n_estimators': 200,
            'max_depth': 30,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
    ),
    'gradient_boosting': (
        GradientBoostingClassifier,
        {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42
        }
    ),
    'logistic_regression': (
        LogisticRegression,
        {
            'max_iter': 1000,
            'random_state': 42,
            'n_jobs': -1
        }
    ),
    'naive_bayes': (
        MultinomialNB,
        {}
    ),
    'svm': (
        LinearSVC,
        {
            'max_iter': 2000,
            'random_state': 42
        }
    )
}


class CVClassifierPipelineBuilder:
    """
    Builder class for creating CV classification pipelines.

    The pipeline structure is:
        TextCleaner -> TfidfVectorizer -> Classifier

    All transformations are encapsulated in the pipeline,
    ensuring they are only fitted on training data.
    """

    def __init__(
        self,
        classifier_name: str = 'random_forest',
        tfidf_params: Optional[Dict[str, Any]] = None,
        cleaner_params: Optional[Dict[str, Any]] = None,
        classifier_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the pipeline builder.

        Args:
            classifier_name: Name of classifier ('random_forest', 'gradient_boosting', etc.)
            tfidf_params: Override default TF-IDF parameters
            cleaner_params: Override default text cleaner parameters
            classifier_params: Override default classifier parameters
        """
        self.classifier_name = classifier_name
        self.tfidf_params = tfidf_params or {}
        self.cleaner_params = cleaner_params or {}
        self.classifier_params = classifier_params or {}

    def _get_default_tfidf_params(self) -> Dict[str, Any]:
        """Get default TF-IDF parameters."""
        return {
            'max_features': 5000,
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.95,
            'sublinear_tf': True
        }

    def _get_default_cleaner_params(self) -> Dict[str, Any]:
        """Get default text cleaner parameters."""
        return {
            'lowercase': True,
            'remove_urls': True,
            'remove_emails': True,
            'remove_phone_numbers': True,
            'remove_numbers': False,
            'remove_punctuation': True,
            'remove_stopwords': True,
            'lemmatize': True,
            'stem': False
        }

    def build(self) -> Pipeline:
        """
        Build and return the complete pipeline.

        Returns:
            sklearn Pipeline with text cleaning, TF-IDF, and classifier
        """
        # Merge default params with overrides
        cleaner_params = {**self._get_default_cleaner_params(), **self.cleaner_params}
        tfidf_params = {**self._get_default_tfidf_params(), **self.tfidf_params}

        # Get classifier class and default params
        if self.classifier_name not in CLASSIFIERS:
            raise ValueError(
                f"Unknown classifier: {self.classifier_name}. "
                f"Available: {list(CLASSIFIERS.keys())}"
            )

        clf_class, default_clf_params = CLASSIFIERS[self.classifier_name]
        clf_params = {**default_clf_params, **self.classifier_params}

        # Build the pipeline
        pipeline = Pipeline([
            ('text_cleaner', TextCleanerTransformer(**cleaner_params)),
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('classifier', clf_class(**clf_params))
        ])

        return pipeline

    def build_for_grid_search(self) -> Pipeline:
        """
        Build a pipeline suitable for GridSearchCV.

        Uses the same structure but allows for parameter tuning.
        Parameter names follow the format: step_name__param_name

        Returns:
            sklearn Pipeline
        """
        return self.build()

    @staticmethod
    def get_param_grid(classifier_name: str = 'random_forest') -> Dict[str, list]:
        """
        Get a parameter grid for GridSearchCV.

        Args:
            classifier_name: Name of the classifier

        Returns:
            Dictionary of parameter lists for grid search
        """
        # Common TF-IDF parameters to tune
        base_grid = {
            'tfidf__max_features': [3000, 5000, 7000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
        }

        # Classifier-specific parameters
        classifier_grids = {
            'random_forest': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [20, 30, None],
            },
            'gradient_boosting': {
                'classifier__n_estimators': [50, 100],
                'classifier__learning_rate': [0.05, 0.1],
                'classifier__max_depth': [3, 5],
            },
            'logistic_regression': {
                'classifier__C': [0.1, 1.0, 10.0],
            },
            'naive_bayes': {
                'classifier__alpha': [0.1, 0.5, 1.0],
            },
            'svm': {
                'classifier__C': [0.1, 1.0, 10.0],
            }
        }

        grid = {**base_grid}
        if classifier_name in classifier_grids:
            grid.update(classifier_grids[classifier_name])

        return grid

    @staticmethod
    def list_available_classifiers() -> list:
        """Return list of available classifier names."""
        return list(CLASSIFIERS.keys())
