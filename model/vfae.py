import torch
from torch.nn import Module, Linear, ReLU


class VariationalFairAutoEncoder(Module):
    def __init__(self,
                 x_dim,
                 s_dim,
                 y_dim,
                 z1_enc_dim,
                 z2_enc_dim,
                 z1_dec_dim,
                 x_dec_dim,
                 z_dim,
                 activation=ReLU()):
        super().__init__()
        y_out_dim = 2 if y_dim == 1 else y_dim

        self.encoder_z1 = VariationalMLP(x_dim + s_dim, z1_enc_dim, z_dim, activation)
        self.encoder_z2 = VariationalMLP(z_dim + y_dim, z2_enc_dim, z_dim, activation)

        self.decoder_z1 = VariationalMLP(z_dim + y_dim, z1_dec_dim, z_dim, activation)
        self.decoder_y = DecoderMLP(z_dim, x_dec_dim, y_out_dim, activation)
        self.decoder_x = DecoderMLP(z_dim + s_dim, x_dec_dim, x_dim + s_dim, activation)

    def forward(self, inputs):
        x, s, y = inputs['x'], inputs['s'], inputs['y']
        # encode
        x_s = torch.cat([x, s], dim=1)
        z1_encoded, z1_enc_logvar, z1_enc_std = self.encoder_z1(x_s)

        z1_y = torch.cat([z1_encoded, y], dim=1)
        z2_encoded, z2_enc_logvar, z2_enc_std = self.encoder_z2(z1_y)

        # decode
        z2_y = torch.cat([z2_encoded, y], dim=1)
        z1_decoded, z1_dec_logvar, z1_dec_std = self.decoder_z1(z2_y)

        z1_s = torch.cat([z1_decoded, s], dim=1)
        x_decoded = self.decoder_x(z1_s)

        y_decoded = self.decoder_y(z1_encoded)

        outputs = {
            # predictive outputs
            'x_decoded': x_decoded,
            'y_decoded': y_decoded,

            # outputs for regularization loss terms
            'z1_enc_logvar': z1_enc_logvar,
            'z1_enc_std': z1_enc_std,

            'z2_enc_logvar': z2_enc_logvar,
            'z2_enc_std': z2_enc_std,

            'z1_dec_logvar': z1_dec_logvar,
            'z1_dec_std': z1_dec_std
        }

        return outputs


class VariationalMLP(Module):
    def __init__(self, in_features, hidden_dim, z_dim, activation):
        super().__init__()
        self.encoder = Linear(in_features, hidden_dim)
        self.activation = activation

        self.logvar_encoder = Linear(hidden_dim, z_dim)
        self.std_encoder = Linear(hidden_dim, z_dim)

    def forward(self, inputs):
        x = self.encoder(inputs)
        logvar = (0.5 * self.logvar_encoder(x)).exp()
        std = self.std_encoder(x)

        # reparameterization trick: we draw a random z
        epsilon = torch.randn_like(std)
        z = epsilon * std + logvar
        return z, logvar, std


class DecoderMLP(Module):
    def __init__(self, in_features, hidden_dim, latent_dim, activation):
        super().__init__()
        self.lin_encoder = Linear(in_features, hidden_dim)
        self.activation = activation
        self.lin_out = Linear(hidden_dim, latent_dim)

    def forward(self, inputs):
        x = self.activation(self.lin_encoder(inputs))
        return self.lin_out(x)
