from hydra.utils import to_absolute_path
from torch.nn import Module, Linear

from lightning_wrapper import LightningModel


class LogisticRegression(Module):
    """Simple logistic regression model"""

    def __init__(self, in_features, out_features):
        """

        :param in_features: number of input features
        :param out_features: number of outputs
        """
        super().__init__()
        self.lin = Linear(in_features, out_features)

    def forward(self, inputs):
        return self.lin(inputs)


class LogisticRegressionVFAE(Module):
    """Perform logistic regression on the representations from a pre-trained vfae (with frozen parameters)."""

    def __init__(self, vfae_ckpt, z_dim, out_features):
        """

        :param vfae_ckpt: path to the vfae checkpoint to use for getting the latent representations.
        :param z_dim: latent dimension of the vfae
        :param out_features: number of output features
        """
        super().__init__()
        vfae_ckpt = to_absolute_path(vfae_ckpt)
        # load and freeze the vfae model
        self.vfae = LightningModel.load_from_checkpoint(vfae_ckpt).model
        for param in self.vfae.parameters():
            param.requires_grad = False

        self.log_regression = LogisticRegression(z_dim, out_features)

    def forward(self, inputs):
        x = self.vfae(inputs)['z1_encoded'].detach()
        x = self.log_regression(x)
        return x
