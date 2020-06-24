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

        kl_loss_z1 = self._kl_gaussian(y_pred['z1_enc_logvar'],
                                       y_pred['z1_enc_mu'],
                                       y_pred['z1_dec_logvar'],
                                       y_pred['z1_dec_mu'])

        # becomes kl between z2 and a standard normal when passing zeros
        zeros = torch.zeros_like(y_pred['z1_enc_logvar'])
        kl_loss_z2 = self._kl_gaussian(y_pred['z2_enc_logvar'],
                                       y_pred['z2_enc_mu'],
                                       zeros,
                                       zeros)

        loss = reconstruction_loss + kl_loss_z1 + kl_loss_z2 + self.alpha * supervised_loss
        # TODO MMD penalty
        return loss

    @staticmethod
    def _kl_gaussian(logvar_a, mu_a, logvar_b, mu_b):
        """
        Average KL divergence between two (multivariate) gaussians based on their mean and standard deviation for a
        batch of input samples.

        :param logvar_a: standard deviation a
        :param mu_a: mean a
        :param logvar_b: standard deviation b
        :param mu_b: mean b
        :return: kl divergence, mean averaged over batch dimension.
        """
        per_example_kl = logvar_b - logvar_a - 1 + (logvar_a.exp() + (mu_a - mu_b).square()) / logvar_b.exp()
        kl = 0.5 * torch.sum(per_example_kl, dim=1)
        return kl.mean()
