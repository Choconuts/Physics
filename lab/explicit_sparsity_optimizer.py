import torch
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


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
    init_shape = list(mean.shape[:-1])
    cov = torch.eye(3, device=mean.device) * var
    cov = cov[None, :, :].expand(mean.view(-1, 3).shape[0], -1, -1)
    return cov.reshape(init_shape + [3, 3])


class IPE(torch.nn.Module):

    def __init__(self, min_deg=0, max_deg=16, in_dim=3, diag=True):
        super(IPE, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.diag = diag
        self.in_dim = in_dim

    def forward(self, mean, cov):
        init_shape = list(mean.shape[:-1]) + [-1]
        mean = mean.view(-1, 3)
        cov = cov.view(-1, 3, 3)
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

    def forward(self, z, var=0.05):
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
        return x

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


class CRF(torch.nn.Module):

    def __init__(self, values):
        super(CRF, self).__init__()
        if isinstance(values, torch.Tensor):
            self.s = values
        else:
            self.s = torch.nn.Parameter(torch.tensor(values))

    def f(self, x):
        sig = 1 / (1 + torch.exp(-1 / self.s * x))
        return (sig - 0.5) * 1.05 + 0.5

    def inv_f(self, x):
        # x = torch.clamp(x, min=1e-4, max=1-1e-4)
        x = (x - 0.5) / 1.05 + 0.5
        return -self.s * torch.log(1 / x - 1)

    def g(self, x):
        return torch.log2(x / 2.5)

    def forward(self, x):
        return self.f(self.g(x))

    def inv_g(self, x):
        return torch.exp2(x) * 2.5

    def inv(self, x):
        y = self.inv_g(self.inv_f(x))
        return y

    def getter_and_setter(self, i):
        def setter(v):
            tmp = torch.tensor([1.0, 1.0, 1.0])
            for j in range(3):
                tmp[j] = self.s[j]
            tmp[i] = v
            self.s = torch.nn.Parameter(tmp)
            print(self.s)
        return lambda : self.s[i], setter


class Problem:

    def __init__(self):
        self.ofs_ratio = 0.0
        self.x = torch.tensor([
            [0.2, 0.8, 0.1],
            [0.6, 0.1, 0.2],
            [0.4, 0.9, 0.7],
            [0.5, 0.1, 0.8],
        ]).cuda()

        self.t = torch.tensor([[0.4]] * 5 + [[0.6]] * 10 + [[0.2]] * 5 + [[0.05]] * 2 +
                               [[0.21], [0.23], [0.25], [0.27], [0.29]]).cuda() * 2 + 2.0
        n_t = self.t.shape[0]

        self.z_to_id = lambda z: ((torch.sin(z * 8) * 20 + torch.cos(z * 4) * 10 + z ** 5 * 8) % 4).long()
        self.z_to_d = lambda z: (torch.stack([
            (torch.sin(z * 7) * 6 + torch.cos(z * 6) * 4) * 0.1,
            (torch.sin(z * 8) * 15 + torch.cos(z * 9) * 5) * 0.05,
            (torch.sin(z * 9) * 8 + torch.cos(z * 7) * 2) * 0.1
        ], -1) * 0.5 + 0.5) * self.ofs_ratio
        # self.z_to_i = lambda z: ((torch.sin(z * 7) * 3 + torch.cos(z * 6) * 2) + 5)[..., None] * 0.5
        self.z_to_i = lambda z: self.t[((torch.sin(z * 6) * 21 + torch.cos(z * 5) * 12 + z ** 4 * 600) % n_t).long()]
        self.crf = CRF([0.3, 0.7, 0.4])
        self.crf.to("cuda")

    def sample_data(self, n, noised=False):
        z = torch.rand([n], device=self.x.device)
        index = self.z_to_id(z)
        coef = self.z_to_i(z)
        # coef = torch.rand(n, 1, device=self.x.device) * 5 + 1e-3
        ofs = self.z_to_d(z)
        result = (coef + ofs) * self.x[index]

        # add noise
        if noised:
            noise = torch.rand_like(ofs) * self.ofs_ratio * 0.1
            ofs = ofs + noise

        return z[..., None], coef, ofs, self.crf(result)

    def sample_test(self, n, noised=False):
        z, coef, ofs, result = self.sample_data(n, noised)
        return z, ofs

    def gt_field(self, x):
        res = torch.exp(-((x[..., None, :] - self.x) ** 2).sum(-1) * 100)
        return res.sum(-1)


class BucketModel(torch.nn.Module):

    def __init__(self):
        super(BucketModel, self).__init__()
        self.bucket = torch.nn.Sequential(
            SDFNetwork(),
            torch.nn.Sigmoid()
        )
        self.weights = torch.nn.Sequential(
            SDFNetwork(),
        )

    def forward(self, x):
        return self.bucket(x), torch.nn.functional.softplus(self.weights(x) - 10)


class NaiveModel(torch.nn.Module):

    def __init__(self):
        super(NaiveModel, self).__init__()
        self.field = torch.nn.Sequential(
            SDFNetwork(d_out=6, d_in=1, d_hidden=128, multires=12, embed="PE"),
            torch.nn.Sigmoid()
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(6, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 4),
        )

    def forward(self, z):
        rho_hat = self.field(z)
        return torch.split(self.mlp(rho_hat), [3, 1], -1)


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


class BRDFGatherModel(torch.nn.Module):

    def __init__(self, multires=10, brdf_mlp_dims=[128, 128], n_samples=100, feat_dim=3):
        super(BRDFGatherModel, self).__init__()
        self.brdf_embed_fn, brdf_input_dim = get_embedder(multires)

        self.actv_fn = torch.nn.LeakyReLU(0.2)
        self.n_samples = n_samples

        mlp_layers = []
        dim = brdf_input_dim
        for i in range(len(brdf_mlp_dims)):
            mlp_layers.append(torch.nn.Linear(dim, brdf_mlp_dims[i]))
            mlp_layers.append(self.actv_fn)
            dim = brdf_mlp_dims[i]
        mlp_layers.append(torch.nn.Linear(dim, 1))
        self.brdf_spec = torch.nn.Sequential(*mlp_layers)

        self.weight_act = lambda x: torch.nn.functional.softplus(x - 10)
        self.weight_field = SDFNetwork(d_in=3)
        self.feat_field = SDFNetwork(feat_dim)
        # self.warp_field = SDFNetwork(3, d_in=6)

        self.naive = NaiveModel()

        self.coef_weight_field = SDFNetwork(d_in=1)

    def sample(self, ofs, res, n, rng=False):
        # find min t
        t_min = torch.max(res - ofs, dim=-1)[0]
        t_min = torch.clamp(t_min, min=1e-4)
        t_max = 5.0

        l0 = torch.norm(res, dim=-1, keepdim=True)
        d = ofs.mean(-1, keepdim=True)
        s = torch.linspace(0, 1, n, device=ofs.device)
        if rng:
            s = s + torch.rand_like(s) * (0.5 / n)
        s = (l0 / (t_min[..., None] + d) - l0 / (t_max + d)) * s
        ts = l0 / (l0 / (t_min[..., None] + d) - s) - d

        pos = res[..., None, :] / (ts[..., None] + ofs[..., None, :])

        assert not pos.isnan().any()
        return pos

    def density(self, z, gt=None):
        z = z.view(-1, 3)
        if gt is None:
            gt = torch.ones_like(z)
            gt[..., 0] = 0.3
            gt[..., 1] = 0.7
            gt[..., 2] = 0.4
        # wxs = self.warp_field(torch.cat([z, gt], -1)) * 0.1 + z
        # sigma = self.weight_field(torch.cat([z, gt], -1))
        sigma = self.weight_field(z)
        sigma = self.weight_act(sigma)
        return sigma

    def forward(self, x, ofs=None, results=None, visualize=False, manual_noised=True, cat_feat=None):
        """

        :param x: unused identifiers (position)
        :param ofs: R/G/B in (t + R) * d_r + (t + G) * d_g + (t + B) * d_b = C
        :param results: C above
        :return: [d_r, d_g, d_b]
        """
        if ofs is None:
            return torch.ones_like(x) * 0.5, torch.ones_like(x)[..., :1] * 0.1

        if manual_noised:
            noise = torch.rand_like(ofs) * 0.001
            ofs = ofs + noise

        n_sample = 50
        xs = self.sample(ofs, results, n_sample)

        if visualize:
            vis_samples(xs[:30])

        inputs = torch.cat([xs.view(-1, 3), cat_feat[..., None, :].expand(xs.shape[0], n_sample, -1).reshape(-1, 3)], -1)
        # dxs = self.warp_field(inputs.view(-1, 6))
        # wxs = xs.view(-1, 3) + dxs * 0.1
        c = self.feat_field(xs.view(-1, 3))     # wxs, wxs.view(-1, 3)
        sigma = self.weight_field(xs)
        sigma = self.weight_act(sigma)

        c = c.reshape(-1, n_sample, c.shape[-1])
        sigma = sigma.reshape(-1, n_sample, 1)

        # weight = (1 - torch.exp(-sigma))
        # weight = weight / (torch.sum(weight, -2, keepdim=True) + 1e-6)
        alpha = (1.0 - torch.exp(-sigma))[..., 0]
        T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)
        weight = (alpha * T[:, :-1])[..., None]

        final_x = (xs * weight).sum(-2)
        final_c = (c * weight).sum(-2)
        return final_x, self.naive(x)[1], final_c, weight.sum(-2)[..., 0] # self.brdf_spec(self.brdf_embed_fn(x.expand(-1, 3)))


class SpaceDistrib(torch.nn.Module):

    def __init__(self):
        super(SpaceDistrib, self).__init__()
        self.weight_act = lambda x: torch.nn.functional.softplus(x - 10)
        self.weight_field = SDFNetwork(d_in=3)

    def forward(self, x):
        w = self.weight_field(x)
        w = self.weight_act(w)
        return w, w.sum(-2, keepdim=True)


class SampleModel(torch.nn.Module):

    def __init__(self):
        super(SampleModel, self).__init__()
        self.mlp = torch.nn.Sequential(
            SDFNetwork(d_out=4, d_in=1, d_hidden=128, multires=20),
        )
        self.bucket = BucketModel()

    def sample_around(self, z):
        x, c0 = torch.split(self.mlp(z), [3, 1], -1)
        n_sample = 125
        lins = torch.linspace(0, 1, 5, device=z.device)
        rng = torch.cat(torch.meshgrid(lins, lins, lins), -1).view(-1, 3)
        rng = (rng - 0.5) * 0.1
        xs = x[..., None, :] + rng
        return xs

    def forward(self, z, bucket=None):
        if bucket is None:
            bucket = self.bucket

        x, c0 = torch.split(self.mlp(z), [3, 1], -1)
        n_sample = 125
        lins = torch.linspace(0, 1, 5, device=z.device)
        rng = torch.cat(torch.meshgrid(lins, lins, lins), -1).view(-1, 3)
        rng = (rng - 0.5) * 0.1
        xs = x[..., None, :] + rng

        c, sigma = bucket(xs.view(-1, 3))
        c, sigma = c.reshape(-1, n_sample, 1), sigma.reshape(-1, n_sample, 1)

        weight = (1 - torch.exp(-sigma))
        weight = weight / (torch.sum(weight, -2, keepdim=True) + 1e-6)

        final_x = (xs * weight).sum(-2)
        final_c = (c * weight).sum(-2)

        return final_x, final_c


def vis(f):
    if not hasattr(vis, "i"):
        vis.i = 0
    plt.figure()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ind = torch.linspace(0, 1, 50)
    z = torch.stack(torch.meshgrid(ind, ind, ind), -1).cuda().view(-1, 3)
    x = f(z)

    z = z.view(50, 50, 50, 3)[:, :, 0].view(-1, 3)
    x = x.view(50, -1, 1).sum(0)

    x = x.cpu().detach().numpy()
    z = z.cpu().detach().numpy()
    plt.scatter(z[..., 0], z[..., 1], c=x)
    plt.savefig(f"logs/opt/vis-{vis.i}.png")
    vis.i += 1
    plt.close()


def solve_line(coefs, results):
    zeros = torch.zeros_like(coefs[..., 0])
    abc = torch.stack([coefs[..., 0], -coefs[..., 1], zeros], -1)
    efg = torch.stack([zeros, -coefs[..., 1], coefs[..., 2]], -1)

    vec = torch.cross(abc, efg)
    pos = torch.stack([
        (results[..., 0] - results[..., 1]) / coefs[..., 0],
        zeros,
        (results[..., 2] - results[..., 1]) / coefs[..., 2]], -1)

    vec = vec / torch.linalg.norm(vec, dim=-1, keepdim=True)

    return pos, vec


def box_intersection(aabb, origins, directions, forward_only=True):
    inv_dir = 1.0 / directions
    t_min = (aabb[0] - origins) * inv_dir
    t_max = (aabb[1] - origins) * inv_dir
    t1 = torch.minimum(t_min, t_max)
    t2 = torch.maximum(t_min, t_max)

    near = torch.maximum(torch.maximum(t1[..., 0:1], t1[..., 1:2]), t1[..., 2:3])
    far = torch.minimum(torch.minimum(t2[..., 0:1], t2[..., 1:2]), t2[..., 2:3])

    if forward_only:
        return torch.maximum(near, torch.zeros_like(near)), far, torch.logical_and(near < far, far > 0)

    return near, far, near < far


def sample_line(pos, vec, n, rng=False):
    aabb = torch.tensor([
        [0, 0, 0],
        [1, 1, 1.]
    ], device=pos.device)
    near, far, valid = box_intersection(aabb, pos, vec, False)
    ts = torch.linspace(0, 1, n, device=pos.device)
    if rng:
        ts = ts + torch.rand_like(ts) * (0.5 / n)
    ts = near * (1 - ts) + far * ts

    return ts[..., :, None] * vec[..., None, :] + pos[..., None, :], (far - near) / n


def vis_line(pos, vec):
    def f(x):
        d = (x - pos)[..., :2]
        r = torch.linalg.norm(d, dim=-1)
        v = vec[..., :2] / torch.linalg.norm(vec[..., :2], dim=-1, keepdim=True)
        b = (d * v).sum(-1)
        return (r ** 2 - b ** 2) < 0.001
    vis(f)


def vis_samples(xs):
    plt.figure()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    x = xs[..., 1].view(-1, )
    y = xs[..., 2].view(-1, )

    # x = torch.cat([x, torch.tensor([0.9], device=x.device)])
    # y = torch.cat([y, torch.tensor([0.7], device=x.device)])
    plt.scatter(x.cpu().detach().numpy(), y.cpu().detach().numpy())
    plt.savefig(f"logs/opt/samples.png")
    plt.close()


def vis_crf():
    plt.figure()
    plt.xlim(0, 20)
    plt.ylim(0, 1)
    prob = Problem()
    x = torch.linspace(1e-4, 20, 500, device='cuda')[..., None]
    # y = -prob.crf.g(x) * 0.05
    y = prob.crf(x)

    plt.plot(x.cpu().detach().numpy(), y.cpu().detach().numpy())
    plt.savefig(f"logs/opt/crf.png")
    plt.close()

    plt.figure()
    plt.xlim(0, 1)
    plt.ylim(0, 20)
    x = torch.linspace(1e-4, 1, 500, device='cuda')[..., None]
    # y = prob.crf.inv_g(-x * 20)
    y = prob.crf.inv(x)

    plt.plot(x.cpu().detach().numpy(), y.cpu().detach().numpy())
    plt.savefig(f"logs/opt/inv_crf.png")
    plt.close()


def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
    rho = torch.tensor([rho] * len(rho_hat)).cuda()
    return torch.mean(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))


def naive_train():
    torch.manual_seed(3)
    model = NaiveModel()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), 0.0005, (0.9, 0.99))
    problem = Problem()

    vis(problem.gt_field)

    pbar = trange(30001)
    for i in pbar:
        z, coef, ofs, result = problem.sample_data(5000, True)
        # index = problem.z_to_id(z).long()
        x, c = model(z)

        final_result = x * (c + ofs)
        loss = (final_result - result) ** 2
        loss = loss.mean()
        # loss = (index - c) ** 2

        KL = kl_divergence(0.05, model.field(z))

        total_loss = loss + KL * 0.1

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 10 == 0:
            x_gt = problem.x[problem.z_to_id(z)][:, 0, :]
            x_diff = (x - x_gt).abs()
            loss = (x * (coef + ofs) - result) ** 2
            loss = loss.mean()

            diff = ((x[..., None, :] - problem.x).abs()).min(-2)[0]
            pbar.set_postfix(Loss=loss.item(), KL=KL.item(), xc_diff=diff.mean().item(), x_diff=x_diff.mean().item())

        if i % 1000 == 0:
            # vis(lambda x: model(x)[1])
            print(list(map(torch.Tensor.item, x[0])), list(map(torch.Tensor.item, c[0])))
            # print((model.field(z)[0] > 1e-3).long())


def improve_train():
    torch.manual_seed(3)
    # model = NaiveModel()
    model = BRDFGatherModel()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), 0.0005, (0.9, 0.99))
    problem = Problem()

    vis(problem.gt_field)

    pbar = trange(30001)
    for i in pbar:
        z, coef, ofs, result = problem.sample_data(5000)
        # index = problem.z_to_id(z).long()

        rand_crf = torch.rand_like(result)
        crf = CRF(rand_crf)
        crf.cuda()
        result = crf.inv(result)

        x, t, c, acc = model(z, ofs, result, i % 1000 == 200, cat_feat=rand_crf)

        final_result = x * (t + ofs)
        loss = ((final_result - result) ** 2).mean()
        # loss = (index - c) ** 2

        # KL = torch.tensor(0., device=z.device)
        rand_input = torch.rand(x.shape[0] * 50, x.shape[1], device=x.device)
        KL = model.density(rand_input, torch.rand_like(rand_input)).abs().mean()

        acc_loss = (acc - 1).abs().mean()

        feat_loss = ((c - rand_crf) ** 2).mean()

        total_loss = loss + KL * 0.1 + acc_loss     # + feat_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 10 == 0:
            x_gt = problem.x[problem.z_to_id(z)][:, 0, :]
            x_diff = (x - x_gt).abs()
            loss = (x * (coef + ofs) - result) ** 2
            loss = loss.mean()

            diff = ((x[..., None, :] - problem.x).abs()).min(-2)[0]
            pbar.set_postfix(Loss=loss.item(), KL=KL.item(), xc_diff=diff.mean().item(), x_diff=x_diff.mean().item(), feat=feat_loss.item())

        if i % 1000 == 0:
            vis(lambda x: torch.clamp(model.density(x), max=1.0))
            # print(list(map(torch.Tensor.item, x[0])), list(map(torch.Tensor.item, c[0])))
            print(list(map(torch.Tensor.item, x[0])))
            # print((model.field(z)[0] > 1e-3).long())
            pass


# def special_train():
#     torch.manual_seed(3)
#
#     ann_crf = AnnealCRF()
#     # {"lr": 0.0005, "params": crf.parameters()},
#     problem = Problem()
#
#     vis(problem.gt_field)
#
#     pbar = trange(5000)
#     for j in pbar:
#         crf = ann_crf()
#         model = BRDFGatherModel()
#         model.cuda()
#         optimizer = torch.optim.Adam([{"lr": 0.0005, "params": model.parameters()}], betas=(0.9, 0.99))
#
#         for i in range(5):
#             z, coef, ofs, result = problem.sample_data(5000)
#
#             result = crf.inv(result)
#
#             # index = problem.z_to_id(z).long()
#             x, c, acc = model(z, ofs, result, i % 1000 == 200)
#
#             final_result = x * (c + ofs)
#             loss = ((final_result - result) ** 2).mean()
#             # loss = (index - c) ** 2
#
#             # KL = torch.tensor(0., device=z.device)
#             KL = model.density(torch.rand_like(x)).abs().mean()
#
#             acc_loss = (acc - 1).abs().mean()
#
#             total_loss = loss + KL * 0.1 + acc_loss
#
#             optimizer.zero_grad()
#             total_loss.backward()
#             optimizer.step()
#
#         loss = ((crf.s - problem.crf.s) ** 2).mean()
#         ann_crf.step(loss)
#
#         if j % 10 == 0:
#             x_gt = problem.x[problem.z_to_id(z)][:, 0, :]
#             x_diff = (x - x_gt).abs()
#             loss = (x * (coef + ofs) - result) ** 2
#             loss = loss.mean()
#
#             diff = ((x[..., None, :] - problem.x).abs()).min(-2)[0]
#             pbar.set_postfix(Loss=loss.item(), KL=KL.item(), xc_diff=diff.mean().item(),
#                              x_diff=x_diff.mean().item())
#
#         if j % 100 == 0:
#             vis(lambda x: torch.clamp(model.density(x), max=1.0))
#             # print(list(map(torch.Tensor.item, x[0])), list(map(torch.Tensor.item, c[0])))
#             print(list(map(torch.Tensor.item, x[0])))
#             print(list(map(torch.Tensor.item, ann_crf.ann.s)), ann_crf.ann.t)
#             # print((model.field(z)[0] > 1e-3).long())
#             pass
#
#
# class Annealing:
#
#     def __init__(self, s0, t0=100.):
#         self.t = t0
#         self.s = s0
#         self.last_loss = 10000
#         self.next_s = s0
#
#     def get_s(self):
#         ofs = torch.rand_like(self.s) * 2 - 1.0
#         s = self.s + ofs * 0.1
#         self.next_s = s
#         return s
#
#     def step(self, loss):
#         delta = loss - self.last_loss
#         prob = torch.exp(-delta / self.t)
#         if loss < self.last_loss or (np.random.rand(1)[0] < prob).all():
#             self.last_loss = loss
#             self.s = self.next_s
#         self.t *= 0.99
#
#
# class AnnealCRF:
#
#     def __init__(self):
#         s = torch.tensor([0., 0., 0.]).cuda()
#         self.ann = Annealing(s)
#
#     def __call__(self):
#         s = self.ann.get_s()
#         return CRF(s)
#
#     def step(self, loss):
#         self.ann.step(loss)
#
#
# def tst_annealing():
#     s = torch.tensor([0.5, 0.5, 0.5])
#     ann = Annealing(s)
#     ann_crf = AnnealCRF()
#
#     def c(x): return ((x - torch.tensor([0.3, 0.7, 0.4], device=x.device)) ** 2).mean()
#
#     for i in range(10000):
#         s = ann.get_s()
#         crf = ann_crf()
#         loss = c(s)
#         ann.step(loss)
#         # ann_crf.step(loss)
#         # torch.rand(100, device=loss.device)
#         print(s)
#         print(loss)


def moving_train():
    torch.manual_seed(3)
    # model = NaiveModel()
    model = BRDFGatherModel()
    model.cuda()

    crf = CRF([0.5, 0.5, 0.5])
    val_crf = CRF([0., 0., 0.])
    crf.cuda()
    val_crf.cuda()

    distrib = SpaceDistrib()
    distrib.cuda()

    optimizer = torch.optim.Adam([{"lr": 0.0005, "params": model.parameters()},
                                  {"lr": 0.001, "params": crf.parameters()}], betas=(0.9, 0.99))
    problem = Problem()

    vis(problem.gt_field)

    pbar = trange(30001)
    for i in pbar:
        z, coef, ofs, result = problem.sample_data(5000)
        # index = problem.z_to_id(z).long()

        # linsp = torch.linspace(0, 1.0, 10, device=result.device)
        # ss = torch.stack(torch.meshgrid([linsp, linsp, linsp]), -1).view(1, -1, 3)
        # ws, wacc = distrib(ss)
        #
        # wacc_loss = (wacc - 1).abs().mean()
        #
        # rand_crf = CRF((ws * ss).sum(-2))
        # rand_crf.cuda()
        #
        # result = rand_crf.inv(result)
        result = crf.inv(result)
        rand_crf = crf
        wacc_loss = 0.

        x, t, c, acc = model(z, ofs, result, i % 1000 == 200, cat_feat=rand_crf.s)

        final_result = x * (t + ofs)
        loss = ((final_result - result) ** 2).mean()
        # loss = (index - c) ** 2

        # KL = torch.tensor(0., device=z.device)
        rand_input = torch.rand(x.shape[0] * 50, x.shape[1], device=x.device)
        KL = model.density(rand_input, torch.rand_like(rand_input)).abs().mean()

        acc_loss = (acc - 1).abs().mean()

        feat_loss = ((c - rand_crf.s) ** 2).mean()

        total_loss = loss + KL * 0.1 + acc_loss + wacc_loss + feat_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 10 == 0:
            x, t, c, acc = model(z, ofs, result, i % 1000 == 200, cat_feat=torch.zeros_like(result) + problem.crf.s)
            x_gt = problem.x[problem.z_to_id(z)][:, 0, :]
            x_diff = (x - x_gt).abs()
            loss = (x * (coef + ofs) - result) ** 2
            loss = loss.mean()

            diff = ((x[..., None, :] - problem.x).abs()).min(-2)[0]
            pbar.set_postfix(Loss=loss.item(), KL=KL.item(), xc_diff=diff.mean().item(), x_diff=x_diff.mean().item(), feat=feat_loss.item())

        if i % 1000 == 0:
            vis(lambda x: torch.clamp(model.density(x), max=1.0))
            # print(list(map(torch.Tensor.item, x[0])), list(map(torch.Tensor.item, c[0])))
            print(list(map(torch.Tensor.item, x[0])))
            # print((model.field(z)[0] > 1e-3).long())
            print(list(map(torch.Tensor.item, crf.s)))
            pass


if __name__ == '__main__':
    torch.manual_seed(6)
    # naive_train()

    # special_train()
    # tst_annealing()
    moving_train()


