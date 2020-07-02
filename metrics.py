from abc import ABC, abstractmethod

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import Tensor


class Metric(ABC):
    def __init__(self):
        self.__name__ = self.__class__.__name__

    @abstractmethod
    def _compute(self, y_pred: Tensor, y_true: Tensor):
        """Compute the metric given predictions and labels.

        :param y_pred: predicted scores
        :param y_true: ground truth labels
        :return: the computed metric value
        """

    def __call__(self, y_pred: Tensor, y_true: Tensor):
        return self._compute(y_pred, y_true)


class Discrimination(Metric):
    """
    Discrimination Metric.
    """

    def __init__(self, use_probabilities=False):
        """

        :param use_probabilities: whether to use the probabilities categorical predictions.
        """
        super().__init__()
        if use_probabilities:
            self.__name__ += '_probs'
        self.use_probabilities = use_probabilities

    def _compute(self, y_pred: Tensor, y_true: dict):
        """

        :param y_pred: predictions
        :param y_true: dict containing entries y_true (ground truth labels) and is_protected (1 if protected 0 else)
        :return: the discrimination score
        """
        assert isinstance(y_true, dict)
        y_true, s = y_true['y_true'], y_true['is_protected']
        y_pred = torch.sigmoid(y_pred)
        if not self.use_probabilities:
            y_pred = torch.round(y_pred)

        # separate outputs for protected and not protected
        idx_protected = (s == 1).nonzero()[:, 0]
        idx_non_protected = (s == 0).nonzero()[:, 0]

        y_protected = y_pred[idx_protected]
        y_non_protected = y_pred[idx_non_protected]

        # compute the score
        n_protected = len(idx_protected)
        n_non_protected = len(idx_non_protected)

        protected_score = y_protected.sum() / n_protected
        non_protected_score = y_non_protected.sum() / n_non_protected

        disc = torch.abs(protected_score - non_protected_score)

        return disc.item()


class VFAEMetric(Metric):
    """Wrapper around metrics that extracts the right outputs of the VFAE depending on the metric."""

    def __init__(self, metric_name):
        """
        :param metric_name (str): name of the metric to use. Currently supported: 'accuracy'
        """
        if metric_name == 'accuracy':
            self.metric = Accuracy()
        else:
            raise NotImplementedError(f'the metric: {metric_name} is not implemented.')
        self.__class__.__name__ = self.metric.__class__.__name__.lower()

        self.__name__ = metric_name
        self.metric_name = metric_name

    def _compute(self, y_pred: dict, y_true: dict):
        y_pred = y_pred['y_decoded']
        y_true = y_true['y']
        return self.metric(y_pred, y_true)


class SklearnMetric(Metric, ABC):
    """
    base class that pre-processes inputs to make them compatible with sklearn metrics.
    """

    def __call__(self, y_pred: Tensor, y_true: Tensor):
        if y_pred.shape[-1] == 1:
            y_pred = torch.round(torch.sigmoid(y_pred))
        else:
            y_pred = torch.softmax(y_pred, dim=-1).argmax(dim=-1)

        if isinstance(y_true, dict):
            y_true = y_true['y_true']

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
        super().__init__()
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
        super().__init__()
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
        super().__init__()
        self.average = average

    def _compute(self, y_pred: Tensor, y_true: Tensor):
        return f1_score(y_true, y_pred, average=self.average)
