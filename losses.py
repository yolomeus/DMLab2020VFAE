from math import pi, sqrt

import torch
from torch.nn import Module, BCEWithLogitsLoss, CrossEntropyLoss


class VFAELoss(Module):
    """
    Loss function for training the Variational Fair Auto Encoder.
    """

    def __init__(self, alpha=1.0, beta=1.0, mmd_dim=500, mmd_gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.bce = BCEWithLogitsLoss()
        self.ce = CrossEntropyLoss()
        self.mmd = FastMMD(mmd_dim, mmd_gamma)

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

        # compute mmd only if protected and non protected in batch
        z1_enc = y_pred['z1_encoded']
        z1_protected, z1_non_protected = self._separate_protected(z1_enc, s)
        if len(z1_protected) > 0:
            loss += self.beta * self.mmd(z1_protected, z1_non_protected)

        return loss

    @staticmethod
    def _kl_gaussian(logvar_a, mu_a, logvar_b, mu_b):
        """
        Average KL divergence between two (multivariate) gaussians based on their mean and standard deviation for a
        batch of input samples. https://arxiv.org/abs/1405.2664

        :param logvar_a: standard deviation a
        :param mu_a: mean a
        :param logvar_b: standard deviation b
        :param mu_b: mean b
        :return: kl divergence, mean averaged over batch dimension.
        """
        per_example_kl = logvar_b - logvar_a - 1 + (logvar_a.exp() + (mu_a - mu_b).square()) / logvar_b.exp()
        kl = 0.5 * torch.sum(per_example_kl, dim=1)
        return kl.mean()

    @staticmethod
    def _separate_protected(batch, s):
        """separate batch based on labels indicating protected and non protected .

        :param batch: values to select from based on s.
        :param s: tensor of labels with s=1 meaning protected and s=0 non protected.
        :return:
            - protected - items from batch with protected label
            - non_protected - items from batch with non protected label
        """
        idx_protected = (s == 1).nonzero()[:, 0]
        idx_non_protected = (s == 0).nonzero()[:, 0]
        protected = batch[idx_protected]
        non_protected = batch[idx_non_protected]

        return protected, non_protected


class FastMMD(Module):
    """ Fast Maximum Mean Discrepancy approximated using the random kitchen sinks method.
    """

    def __init__(self, out_features, gamma):
        super().__init__()
        self.gamma = gamma
        self.out_features = out_features

    def forward(self, a, b):
        in_features = a.shape[-1]

        # W sampled from normal
        w_rand = torch.randn((in_features, self.out_features), device=a.device)
        # b sampled from uniform
        b_rand = torch.zeros((self.out_features,), device=a.device).uniform_(0, 2 * pi)

        phi_a = self._phi(a, w_rand, b_rand).mean(dim=0)
        phi_b = self._phi(b, w_rand, b_rand).mean(dim=0)
        mmd = torch.norm(phi_a - phi_b, 2)

        return mmd

    def _phi(self, x, w, b):
        scale_a = sqrt(2 / self.out_features)
        scale_b = sqrt(2 / self.gamma)
        out = scale_a * (scale_b * (x @ w + b)).cos()
        return out
