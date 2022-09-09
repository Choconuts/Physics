import torch
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt


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
                 skip_in=(2,),
                 multires=6,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn = PE(num_freq=multires, input_dims=d_in)
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

    def forward(self, z):
        inputs = z * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

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


class Problem:

    def __init__(self):
        self.ofs_ratio = 0.5
        self.x = torch.tensor([
            [0.2, 0.8, 0.1],
            [0.6, 0.1, 0.2],
            [0.4, 0.9, 0.7],
            [0.5, 0.1, 0.8],
        ]).cuda()

        self.z_to_id = lambda z: ((torch.sin(z * 8) * 20 + torch.cos(z * 4) * 10 + z ** 5 * 8) % 4).long()
        self.z_to_d = lambda z: (torch.stack([
            (torch.sin(z * 7) * 6 + torch.cos(z * 6) * 4) * 0.1,
            (torch.sin(z * 8) * 15 + torch.cos(z * 9) * 5) * 0.05,
            (torch.sin(z * 9) * 8 + torch.cos(z * 7) * 2) * 0.1
        ], -1) * 0.5 + 0.5) * self.ofs_ratio
        self.z_to_i = lambda z: ((torch.sin(z * 7) * 3 + torch.cos(z * 6) * 2) + 5)[..., None] * 0.5
        # self.z_to_i = lambda z: ((torch.sin(z * 7) * 3 + torch.cos(z * 6) * 2) + 5)[..., None] * 0.2 + 2        # 变幅小点

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

        return z[..., None], coef, ofs, result

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
            SDFNetwork(d_out=6, d_in=1, d_hidden=128, multires=12),
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

    def __init__(self, multires=10, brdf_mlp_dims=[128, 128], n_samples=100):
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
        self.weight_field = SDFNetwork()

        self.naive = NaiveModel()

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

    def density(self, z):
        sigma = self.weight_field(z.view(-1, 3))
        sigma = self.weight_act(sigma)
        return sigma

    def forward(self, x, ofs=None, results=None, visualize=False, manual_noised=True):
        """

        :param x: unused identifiers (position)
        :param ofs: R/G/B in (t + R) * d_r + (t + G) * d_g + (t + B) * d_b = C
        :param results: C above
        :return: [d_r, d_g, d_b]
        """
        if ofs is None:
            return torch.ones_like(x) * 0.5, torch.ones_like(x)[..., :1] * 0.1

        if manual_noised:
            noise = torch.rand_like(ofs) * 0.05
            ofs = ofs + noise

        n_sample = 32
        xs = self.sample(ofs, results, n_sample)

        if visualize:
            vis_samples(xs[:30])

        # c = self.brdf_spec(xs.view(-1, 3))
        sigma = self.weight_field(xs.view(-1, 3))
        sigma = self.weight_act(sigma)

        # c = c.reshape(-1, n_sample, 1)
        sigma = sigma.reshape(-1, n_sample, 1)

        # weight = (1 - torch.exp(-sigma))
        # weight = weight / (torch.sum(weight, -2, keepdim=True) + 1e-6)
        alpha = (1.0 - torch.exp(-sigma))[..., 0]
        T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)
        weight = (alpha * T[:, :-1])[..., None]

        final_x = (xs * weight).sum(-2)
        # final_c = (c * weight).sum(-2)
        return final_x, self.naive(x)[1], weight.sum(-2)[..., 0] # self.brdf_spec(self.brdf_embed_fn(x.expand(-1, 3)))


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
    plt.savefig(f"logs/gat/vis-{vis.i}.png")
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
    plt.savefig(f"logs/gat/samples.png")
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
        x, c, acc = model(z, ofs, result, i % 1000 == 200)

        final_result = x * (c + ofs)
        loss = ((final_result - result) ** 2).mean()
        # loss = (index - c) ** 2

        # KL = torch.tensor(0., device=z.device)
        KL = model.density(torch.rand_like(x)).abs().mean()

        acc_loss = (acc - 1).abs().mean()

        total_loss = loss + KL * 0.1 + acc_loss

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
            vis(lambda x: torch.clamp(model.density(x), max=1.0))
            # print(list(map(torch.Tensor.item, x[0])), list(map(torch.Tensor.item, c[0])))
            print(list(map(torch.Tensor.item, x[0])))
            # print((model.field(z)[0] > 1e-3).long())
            pass


class ProblemMinMax:

    def __init__(self):
        self.z_to_d = lambda z: (torch.stack([
            (torch.sin(z * 7) * 6 + torch.cos(z * 6) * 4) * 0.1,
            (torch.sin(z * 8) * 15 + torch.cos(z * 9) * 5) * 0.05,
            (torch.sin(z * 9) * 8 + torch.cos(z * 7) * 2) * 0.1
        ], -1) * 0.5 + 0.5)

    def sample_data(self, n):
        ofs = torch.rand(n, 3).cuda()
        z = torch.rand(n).cuda()
        gt = self.z_to_d(z)
        return z[..., None], gt, gt + ofs


def naive_train_minmax():
    torch.manual_seed(3)
    model = NaiveModel()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), 0.0005, (0.9, 0.99))
    problem = ProblemMinMax()

    pbar = trange(30001)
    for i in pbar:
        z, gt, res = problem.sample_data(5000)
        # index = problem.z_to_id(z).long()
        x, c = model(z)

        s_loss = torch.relu(x - res).abs().mean() * 100
        s_loss = s_loss.mean()
        g_loss = ((res - x) ** 2).mean()
        g_loss = g_loss.mean()

        loss = g_loss + s_loss

        KL = kl_divergence(0.05, model.field(z))

        total_loss = loss + KL * 0.1

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 10 == 0:
            x_diff = (x - gt).abs()
            pbar.set_postfix(Loss=loss.item(), KL=KL.item(), x_diff=x_diff.mean().item())

        if i % 1000 == 0:
            # vis(lambda x: model(x)[1])
            print(list(map(torch.Tensor.item, x[0])), list(map(torch.Tensor.item, c[0])))
            # print((model.field(z)[0] > 1e-3).long())


if __name__ == '__main__':
    torch.manual_seed(6)
    # naive_train()

    improve_train()

    # model = BRDFGatherModel()
    # model.cuda()
    # prob = Problem()
    # z, coef, ofs, result = prob.sample_data(10)
    # x, c = model(z, ofs, result, True)
    # gt = prob.x[prob.z_to_id(z)][0]
    # vis(lambda a: (a - gt).norm(dim=-1))
    # print(coef, gt, ofs)

