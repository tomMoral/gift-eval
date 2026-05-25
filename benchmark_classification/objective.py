"""
Time Series Classification Benchmark Objective.

Evaluates time series classification models using standard metrics
(accuracy, F1-score, balanced accuracy, etc.) via scikit-learn.
"""
import logging
from benchopt import BaseObjective
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SUPPORTED_METRICS = [
    "accuracy",
    "f1_weighted",
    "balanced_accuracy",
    "precision_weighted",
    "recall_weighted",
]


class Objective(BaseObjective):
    """Time Series Classification objective.

    Solvers receive the UCR Dataset object and must return predictions as:
    - y_pred: array of shape (n_test_samples,) with class predictions
    - y_pred_proba (optional): array of shape (n_test_samples, n_classes)
      with class probabilities
    """

    name = "UCR Time Series Classification"
    # Benchopt uses the last segment of this URL to name output HTML files.
    url = "https://github.com/SalesforceAIResearch/ucr-benchmark"
    min_benchopt_version = "1.9"
    requirements = [
        "scikit-learn>=1.0.0",
        "numpy>=1.20.0",
    ]

    sampling_strategy = "run_once"

    test_dataset_name = "ucr-ts-classification"
    test_config = {
        "dataset": {"debug": True},  # use debug subset for fast checks
    }

    def set_data(self, ucr_dataset, num_classes, dataset_name):
        """Set up the classification dataset and metrics."""
        self.ucr_dataset = ucr_dataset
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.y_test = ucr_dataset.y_test

    def get_objective(self):
        """Pass the Dataset object to the solver.

        Solvers get the full Dataset so they can access X_train, y_train,
        X_test, y_test, and dataset metadata.
        """
        return dict(ucr_dataset=self.ucr_dataset)

    def evaluate_result(self, y_pred, y_pred_proba=None):
        """Compute classification metrics from the solver's predictions."""
        # Compute all metrics
        acc = accuracy_score(self.y_test, y_pred)
        
        # For multiclass classification
        if self.num_classes > 2:
            f1 = f1_score(self.y_test, y_pred, average="weighted", zero_division=0)
            precision = precision_score(
                self.y_test, y_pred, average="weighted", zero_division=0
            )
            recall = recall_score(
                self.y_test, y_pred, average="weighted", zero_division=0
            )
        else:
            # Binary classification
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            precision = precision_score(
                self.y_test, y_pred, zero_division=0
            )
            recall = recall_score(
                self.y_test, y_pred, zero_division=0
            )

        bal_acc = balanced_accuracy_score(self.y_test, y_pred)

        # Try to compute ROC-AUC if probabilities are available
        roc_auc = None
        if y_pred_proba is not None:
            try:
                if self.num_classes == 2:
                    roc_auc = roc_auc_score(
                        self.y_test, y_pred_proba[:, 1]
                    )
                else:
                    roc_auc = roc_auc_score(
                        self.y_test, y_pred_proba, multi_class="ovr"
                    )
            except Exception:
                pass

        return {
            "accuracy": acc,
            "f1_weighted": f1,
            "balanced_accuracy": bal_acc,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "roc_auc": roc_auc if roc_auc is not None else 0.0,
        }

    def get_one_result(self):
        """Return a trivial (majority class) prediction for benchopt self-checks."""
        import numpy as np

        # Majority class prediction
        unique, counts = np.unique(self.y_test, return_counts=True)
        majority_class = unique[np.argmax(counts)]
        y_pred = np.full(len(self.y_test), majority_class, dtype=int)
        
        return dict(
            y_pred=y_pred,
            y_pred_proba=None,
        )
