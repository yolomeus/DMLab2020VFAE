import torch
from torch.nn import Module, BCEWithLogitsLoss, CrossEntropyLoss


class VFAELoss(Module):
    """
    Loss function for training the Variational Fair Auto Encoder.
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.bce = BCEWithLogitsLoss()
        self.ce = CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        # TODO add KL terms
        x, s, y = y_true['x'], y_true['s'], y_true['y']
        x_s = torch.cat([x, s], dim=-1)

        supervised_loss = self.ce(y_pred['y_decoded'], y)
        reconstruction_loss = self.bce(y_pred['x_decoded'], x_s)

        loss = reconstruction_loss + self.alpha * supervised_loss
        return loss
