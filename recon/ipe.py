import torch
import torch.nn.functional as F
import numpy as np


def expected_sin(x, x_var):
    def safe_trig_helper(x, fn, t=100 * torch.pi):
        return fn(torch.where(torch.abs(x) < t, x, x % t))

    """Estimates mean and variance of sin(z), z ~ N(x, var)."""
    # When the variance is wide, shrink sin towards zero.
    y = torch.exp(-0.5 * x_var) * safe_trig_helper(x, torch.sin)
    y_var = F.relu(0.5 * (1 - torch.exp(-2 * x_var) * safe_trig_helper(2 * x, torch.cos)) - y ** 2)
    return y, y_var


def integrated_pos_enc(x_coord, min_deg, max_deg, diag=False):
    """Encode `x` with sinusoids scaled by 2^[min_deg:max_deg-1].

    Args:
      x_coord: a tuple containing: x, jnp.ndarray, variables to be encoded. Should
        be in [-pi, pi]. x_cov, jnp.ndarray, covariance matrices for `x`.
      min_deg: int, the min degree of the encoding.
      max_deg: int, the max degree of the encoding.
      diag: bool, if true, expects input covariances to be diagonal (full
        otherwise).

    Returns:
      encoded: jnp.ndarray, encoded variables.
    """
    if diag:
        x, x_cov_diag = x_coord
        scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)], device=x.device)
        shape = list(x.shape[:-1]) + [-1]
        y = torch.reshape(x[..., None, :] * scales[:, None], shape)
        y_var = torch.reshape(x_cov_diag[..., None, :] * scales[:, None] ** 2, shape)
    else:
        x, x_cov = x_coord
        num_dims = x.shape[-1]
        basis = torch.cat(
            [2 ** i * torch.eye(num_dims, device=x.device) for i in range(min_deg, max_deg)], 1)
        y = x @ basis
        # Get the diagonal of a covariance matrix (ie, variance). This is equivalent
        # to jax.vmap(jnp.diag)((basis.T @ covs) @ basis).
        y_var = torch.sum((x_cov @ basis) * basis, -2)

    return expected_sin(
        torch.cat([y, y + 0.5 * torch.pi], dim=-1),
        torch.cat([y_var] * 2, dim=-1))[0]


def isotropic_cov(mean, var):
    n_dim = mean.shape[-1]
    assert n_dim <= 3
    init_shape = list(mean.shape[:-1])
    cov = torch.eye(n_dim, device=mean.device) * var
    cov = cov[None, :, :].expand(mean.view(-1, n_dim).shape[0], -1, -1)
    return cov.reshape(init_shape + [n_dim, n_dim])


class IPE(torch.nn.Module):

    def __init__(self, min_deg=0, max_deg=16, in_dim=3, diag=True):
        super(IPE, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.diag = diag
        self.in_dim = in_dim

    def forward(self, mean, cov):
        n_dim = mean.shape[-1]
        init_shape = list(mean.shape[:-1]) + [-1]
        mean = mean.view(-1, n_dim)
        cov = cov.view(-1, n_dim, n_dim)
        if not self.diag:
            cov = torch.diagonal(cov, 0, 1, 2)
        enc = integrated_pos_enc(
            (mean, cov),
            self.min_deg,
            self.max_deg,
        )
        return enc.reshape(init_shape)

    def feature_dim(self) -> int:
        return (self.max_deg - self.min_deg) * 2 * self.in_dim


class PE(torch.nn.Module):
    def __init__(self, input_dims=3, num_freq=10, include_input=True, log_sampling=True):
        super(PE, self).__init__()
        self.kwargs = {
            'input_dims': input_dims,
            'include_input': include_input,
            'max_freq_log2': num_freq - 1,
            'num_freqs': num_freq,
            'log_sampling': log_sampling,
            'periodic_fns': [torch.sin, torch.cos],
        }
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

    def forward(self, inputs):
        return self.embed(inputs)

    def feature_dim(self) -> int:
        return self.out_dim

    def get_cosine_easing_window(self):
        alpha = self.window_curve()
        window = self.cosine_easing_window(0, self.kwargs['max_freq_log2'], self.kwargs['num_freqs'], alpha)
        return window

    @classmethod
    def cosine_easing_window(cls, min_freq_log2, max_freq_log2, num_bands, alpha):
        """Eases in each frequency one by one with a cosine.

        This is equivalent to taking a Tukey window and sliding it to the right
        along the frequency spectrum.

        Args:
          min_freq_log2: the lower frequency band.
          max_freq_log2: the upper frequency band.
          num_bands: the number of frequencies.
          alpha: will ease in each frequency as alpha goes from 0.0 to num_freqs.

        Returns:
          A 1-d numpy array with num_sample elements containing the window.
        """
        if max_freq_log2 is None:
            max_freq_log2 = num_bands - 1.0
        bands = torch.linspace(min_freq_log2, max_freq_log2, num_bands)
        x = torch.clip(alpha - bands, 0.0, 1.0)
        return 0.5 * (1 + torch.cos(torch.pi * x + torch.pi))


class SDFNetwork(torch.nn.Module):
    def __init__(self,
                 d_out=1,
                 d_in=3,
                 d_hidden=64,
                 n_layers=4,
                 skip_in=(),
                 multires=6,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False,
                 embed="IPE"):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None
        self.embed_type = embed

        if multires > 0:
            if embed == "PE":
                embed_fn = PE(num_freq=multires, input_dims=d_in)
            else:
                embed_fn = IPE(max_deg=multires, in_dim=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = embed_fn.feature_dim()

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = torch.nn.Linear(dims[l], out_dim)

            # if geometric_init:
            #     if l == self.num_layers - 2:
            #         if not inside_outside:
            #             torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
            #             torch.nn.init.constant_(lin.bias, -bias)
            #         else:
            #             torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
            #             torch.nn.init.constant_(lin.bias, bias)
            #     elif multires > 0 and l == 0:
            #         torch.nn.init.constant_(lin.bias, 0.0)
            #         torch.nn.init.normal_(lin.weight[:, ], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            #     elif multires > 0 and l in self.skip_in:
            #         torch.nn.init.constant_(lin.bias, 0.0)
            #         torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            #         torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
            #     else:
            #         torch.nn.init.constant_(lin.bias, 0.0)
            #         torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            #
            # if weight_norm:
            #     lin = nn.utils.weight_norm(lin)

            torch.nn.init.orthogonal_(lin.weight)
            torch.nn.init.normal_(lin.bias)

            setattr(self, "lin" + str(l), lin)

        self.activation = torch.nn.LeakyReLU()

    def forward(self, z, var=0.002):
        init_shape = list(z.shape[:-1]) + [-1]
        z = z.view(-1, z.shape[-1])
        inputs = z * self.scale
        if self.embed_fn_fine is not None:
            if self.embed_type == "PE":
                inputs = self.embed_fn_fine(inputs)
            else:
                inputs = self.embed_fn_fine(inputs, isotropic_cov(inputs, var))

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return x.view(init_shape)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        not_eval = torch.is_grad_enabled()
        with torch.enable_grad():
            x.requires_grad_(True)
            y = self.sdf(x)
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=not_eval,
                retain_graph=not_eval,
                only_inputs=True)[0]
        return gradients.unsqueeze(1)


""" Positional encoding embedding. Code was taken from https://github.com/bmild/nerf. """
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim
