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
        """

        :param y_pred: dict containing the vfae outputs
        :param y_true: dict of ground truth labels for x, s and y
        :return: the loss value as Tensor
        """
        x, s, y = y_true['x'], y_true['s'], y_true['y']
        x_s = torch.cat([x, s], dim=-1)

        supervised_loss = self.ce(y_pred['y_decoded'], y)
        reconstruction_loss = self.bce(y_pred['x_decoded'], x_s)

        kl_loss_z1 = self._kl_gaussian(y_pred['z1_enc_logvar'], y_pred['z1_enc_std'],
                                       y_pred['z1_dec_logvar'], y_pred['z1_dec_std'])

        zeros = torch.zeros_like(y_pred['z1_enc_logvar'])
        kl_loss_z2 = self._kl_gaussian(y_pred['z2_enc_logvar'], y_pred['z2_enc_std'],
                                       zeros, zeros)

        loss = reconstruction_loss + kl_loss_z1 + kl_loss_z2 + self.alpha * supervised_loss
        # TODO MMD penalty
        return loss

    @staticmethod
    def _kl_gaussian(logvar_a, std_a, logvar_b, std_b):
        per_example_kl = logvar_b - logvar_a - 1 + (logvar_a.exp() + (std_a - std_b).square()) / logvar_b.exp()
        kl = 0.5 * torch.sum(per_example_kl, dim=1)
        return kl.mean()
