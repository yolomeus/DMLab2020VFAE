from abc import ABC, abstractmethod

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import Tensor


class Metric(ABC):
    @abstractmethod
    def _compute(self, y_pred: Tensor, y_true: Tensor):
        """Compute the metric given predictions and labels.

        :param y_pred: predicted scores
        :param y_true: ground truth labels
        :return: the computed metric value
        """

    def __call__(self, y_pred: Tensor, y_true: Tensor):
        return self._compute(y_pred, y_true)


class SklearnMetric(Metric, ABC):
    """
    base class that pre-processes inputs to make them compatible with sklearn metrics.
    """

    def __call__(self, y_pred: Tensor, y_true: Tensor):
        if y_pred.shape[-1] == 1:
            y_pred = torch.round(torch.sigmoid(y_pred))
        else:
            y_pred = torch.softmax(y_pred, dim=-1).argmax(dim=-1)
        return self._compute(y_pred.cpu(), y_true.cpu())


class Accuracy(SklearnMetric):
    """
    Compute accuracy using scikit-learn's `accuracy_score`.
    """

    def _compute(self, y_pred: Tensor, y_true: Tensor):
        return accuracy_score(y_true, y_pred)


class Precision(SklearnMetric):
    """
        Compute precision using scikit-learn's `precision_score`.
    """

    def __init__(self, average='macro'):
        """
        :param average: check https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html for
        details.
        """
        self.average = average

    def _compute(self, y_pred: Tensor, y_true: Tensor):
        return precision_score(y_true, y_pred, average=self.average)


class Recall(SklearnMetric):
    """
        Compute recall using scikit-learn's `recall_score`.
    """

    def __init__(self, average='macro'):
        """
        :param average: check
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
        for details.
        """
        self.average = average

    def _compute(self, y_pred: Tensor, y_true: Tensor):
        return recall_score(y_true, y_pred, average=self.average)


class F1Score(SklearnMetric):
    """
        Compute the f1 score using scikit-learn's `f1_score`.
    """

    def __init__(self, average='macro'):
        """
        :param average: check
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
        for details.
        """
        self.average = average

    def _compute(self, y_pred: Tensor, y_true: Tensor):
        return f1_score(y_true, y_pred, average=self.average)
