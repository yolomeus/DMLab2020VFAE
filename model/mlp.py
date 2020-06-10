from torch.nn import Module, Linear, ReLU, Sequential, Dropout


class MLP(Module):
    """Simple Multi-Layer Perceptron also known as Feed Forward Neural Network."""

    def __init__(self, in_dim, h_dim, out_dim, dropout):
        """Builds the MLP

        :param in_dim: dimension of the input vectors.
        :param h_dim: hidden dimension.
        :param out_dim: output dimension.
        """
        super().__init__()
        self.classifier = Sequential(Dropout(dropout),
                                     Linear(in_dim, h_dim),
                                     ReLU(),
                                     Dropout(dropout),
                                     Linear(h_dim, out_dim))

    def forward(self, inputs):
        x = self.classifier(inputs)
        return x
