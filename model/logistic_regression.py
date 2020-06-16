from torch.nn import Module, Linear


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
