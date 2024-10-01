import numpy as np
import torch
import torch.nn as nn

ACT2FN = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class BaseCondKernel(nn.Module):
    def __init__(self, args, zc_dim, zd_dim):
        super().__init__()
        self.d = zd_dim
        self.ff_hidden_dim = (
            args.ff_dim_hidden if args.ff_dim_hidden > 0 else int(self.d * 1.4)
        )
        self.use_tanh = args.use_tanh
        self.optimize_mu = args.optimize_mu
        self.zc_dim = zc_dim

    def logpdf(self, z_c, z_d):
        raise NotImplementedError

    def forward(self, z_c, z_d):
        z = self.logpdf(z_c, z_d)
        return -torch.mean(z)


class GaussianCondKernel(BaseCondKernel):
    """
    Used to compute p(z_d | z_c)
    """

    def __init__(self, args, zc_dim, zd_dim, **kwargs):
        super().__init__(args, zc_dim, zd_dim)
        self.K = args.cond_modes
        # self.logC = torch.tensor([-self.d / 2 * np.log(2 * np.pi)])
        self.register_buffer("logC", torch.tensor([-self.d / 2 * np.log(2 * np.pi)]))

        out_ff_dim = self.K * (2 * self.d + 1)  # mu: d, logvar: d, w: 1
        if args.cov_off_diagonal == "var":
            out_ff_dim += self.K * self.d**2
            self.tri = True
        else:
            self.tri = False

        self.ff = FF(args, zc_dim, self.ff_hidden_dim, out_ff_dim)
        self.tanh = nn.Tanh()

    def predicted_average(self, z_c):
        ff_out = self.ff(z_c).view(
            z_c.shape[0], self.K, -1
        )  # [N, K, (2*d+1) + tri_dim]

        w = torch.log_softmax(ff_out[:, :, 0].squeeze(-1), dim=-1)  # [N, K]

        mu = ff_out[:, :, 1 : self.d + 1]  # [N, K * d]

        return (w * mu).sum(dim=1)

    def logpdf(self, z_c, z_d):  # H(z_d|z_c)
        z_d = z_d.unsqueeze(1)  # [N, 1, d]
        ff_out = self.ff(z_c).view(
            z_c.shape[0], self.K, -1
        )  # [N, K, (2*d+1) + tri_dim]

        w = torch.log_softmax(ff_out[:, :, 0].squeeze(-1), dim=-1)  # [N, K]

        mu = ff_out[:, :, 1 : self.d + 1]  # [N, K * d]
        logvar = ff_out[:, :, self.d + 1 : 2 * self.d + 1]
        if self.use_tanh:
            var = self.tanh(logvar).exp()
        else:
            var = logvar.exp()

        # print(f"Cond : {var.min()} | {var.max()} | {var.mean()}")

        z = z_d - mu  # [N, K, d]
        z = var * z
        if self.tri:
            tri = ff_out[:, :, -self.d**2 :].reshape(-1, self.K, self.d, self.d)
            z = z + torch.squeeze(
                torch.matmul(torch.tril(tri, diagonal=-1), z[:, :, :, None]), 3
            )
        z = torch.sum(z**2, dim=-1)  # [N, K]

        z = -z / 2 + torch.log(torch.abs(var) + 1e-8).sum(-1) + w
        z = torch.logsumexp(z, dim=-1)
        return self.logC + z


class FF(nn.Module):
    def __init__(self, args, dim_input, dim_hidden, dim_output, dropout_rate=0):
        super(FF, self).__init__()
        assert (not args.ff_residual_connection) or (dim_hidden == dim_input)
        self.residual_connection = args.ff_residual_connection
        self.num_layers = args.ff_layers
        self.layer_norm = args.ff_layer_norm
        self.activation = ACT2FN[args.ff_activation]

        self.hidden_dim = dim_hidden
        self.input_dim = dim_input
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList()

        for i in range(self.num_layers):
            layer_norm = (
                nn.LayerNorm(dim_hidden) if i != 0 else nn.LayerNorm(self.input_dim)
            )

            linear = nn.Linear(
                self.input_dim if i == 0 else self.hidden_dim, self.hidden_dim
            )
            dropout = nn.Dropout(dropout_rate)
            activation = self.activation()

            self.layers.append(nn.ModuleList([layer_norm, linear, dropout, activation]))

        self.out_layer = nn.Linear(self.hidden_dim, dim_output)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            layer_norm, linear, dropout, activation = layer
            residual = x
            x = linear(x)
            x = activation(x)
            if self.residual_connection:
                x = x + residual
        x = self.out_layer(x)

        return x
