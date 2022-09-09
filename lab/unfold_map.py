import torch
from tqdm import tqdm, trange
from nn_sampler import SDFNetwork
import  matplotlib.pyplot as plt


class Problem:

    def __init__(self):
        self.x = torch.tensor([
            [0.2, 0.8, 0.1],
            [0.6, 0.1, 0.2],
            [0.4, 0.9, 0.7],
            [0.5, 0.1, 0.8],
        ]).cuda()
        self.c = torch.tensor([
            [0.2],
            [0.9],
            [0.0],
            [0.5],
        ]).cuda()

        self.z_to_id = lambda z: (torch.sin(z * 8) * 20 + torch.cos(z * 4) * 10 + z ** 5 * 8) % 4

    def sample_data(self, n):
        z = torch.rand([n], device=self.x.device)
        index = self.z_to_id(z).long()
        coef = torch.rand(n, 3, device=self.x.device) + 1e-3
        theta = torch.rand(n, 1, device=self.x.device) * 9 + 1
        result = self.x[index] * coef + torch.exp(-self.c[index] * theta)
        return z[..., None], coef, theta, result

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
        if multires > 0:
            self.brdf_embed_fn, brdf_input_dim = get_embedder(multires)

        self.actv_fn = torch.nn.LeakyReLU(0.2)
        self.n_samples = n_samples

        mlp_layers = []
        dim = 3
        for i in range(len(brdf_mlp_dims)):
            mlp_layers.append(torch.nn.Linear(dim, brdf_mlp_dims[i]))
            mlp_layers.append(self.actv_fn)
            dim = brdf_mlp_dims[i]
        mlp_layers.append(torch.nn.Linear(dim, 1))
        self.brdf_weight = torch.nn.Sequential(*mlp_layers)

        mlp_layers = []
        dim = 3
        for i in range(len(brdf_mlp_dims)):
            mlp_layers.append(torch.nn.Linear(dim, brdf_mlp_dims[i]))
            mlp_layers.append(self.actv_fn)
            dim = brdf_mlp_dims[i]
        mlp_layers.append(torch.nn.Linear(dim, 1))
        self.brdf_spec = torch.nn.Sequential(*mlp_layers)

        self.weight_act = lambda x: torch.nn.functional.softplus(x - 10)
        self.weight_field = SDFNetwork()

    def solve_line(self, coefs, results):
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

    def box_intersection(self, aabb, origins, directions, forward_only=True):
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

    def sample_line(self, pos, vec, n, rng=False):
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

    def forward(self, x, coefs=None, results=None):
        """

        :param x: unused identifiers (position)
        :param coefs: R/G/B in R * d_r + G * d_g + B * d_b + f(s) = C
        :param results: C above
        :return: [d_r, d_g, d_b], [c]
        """
        if coefs is None:
            return torch.ones_like(x) * 0.5, torch.ones_like(x)[..., :1] * 0.1

        pos, vec = solve_line(coefs, results)
        n_sample = 100
        xs, dists = sample_line(pos, vec, n_sample)

        # c, sigma = self.bucket(xs.view(-1, 3))
        c = self.brdf_spec(xs.view(-1, 3))
        sigma = self.weight_field(xs.view(-1, 3))
        sigma = self.weight_act(sigma)

        c = c.reshape(-1, n_sample, 1)
        sigma = sigma.reshape(-1, n_sample, 1)

        weight = (1 - torch.exp(-sigma))
        weight = weight / (torch.sum(weight, -2, keepdim=True) + 1e-6)

        final_x = (xs * weight).sum(-2)
        final_c = (c * weight).sum(-2)
        return final_x, final_c


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
    plt.savefig(f"logs/buc/vis-{vis.i}.png")
    vis.i += 1


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


def vis_samples():
    """ """
    # plt.figure()
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.scatter(xs[ ..., 0].cpu().detach().numpy(), xs[ ..., 1].cpu().detach().numpy())
    # plt.show()


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
        z, coef, theta, result = problem.sample_data(5000)
        # index = problem.z_to_id(z).long()
        x, c = model(z)

        final_result = x * coef + torch.exp(-c * theta)
        loss = (final_result - result) ** 2
        # loss = (index - c) ** 2

        KL = kl_divergence(0.05, model.field(z))

        loss = loss.mean() + KL * 0.1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            xc = torch.cat([x, c], -1)
            xc_gt = torch.cat([problem.x, problem.c], -1)
            diff = ((xc[..., None, :] - xc_gt).abs()).min(-2)[0]
            pbar.set_postfix(Loss=loss.item(), KL=KL.item(), xc_diff=diff.mean().item())

        if i % 1000 == 0:
            # vis(lambda x: model(x)[1])
            # print(list(map(torch.Tensor.item, x[0])), list(map(torch.Tensor.item, c[0])))
            print((model.field(z)[0] > 1e-3).long())


def train():
    torch.manual_seed(3)
    model = SampleModel()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), 0.0005, (0.9, 0.99))
    problem = Problem()

    vis(problem.gt_field)

    # Pre

    # pbar = trange(3001)
    # for i in pbar:
    #     z, coef, theta, result = problem.sample_data(5000)
    #     # index = problem.z_to_id(z).long()
    #     x, c = x, c0 = torch.split(model.mlp(z), [3, 1], -1)
    #
    #     final_result = x * coef + torch.exp(-c * theta)
    #     loss = (final_result - result) ** 2
    #     # loss = (index - c) ** 2
    #     loss = loss.mean()
    #
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #
    #     if i % 10 == 0:
    #         pbar.set_postfix(Loss=loss.item())

    pbar = trange(3001)
    for i in pbar:
        z, coef, theta, result = problem.sample_data(5000)
        pos, vec = solve_line(coef, result)
        n_sample = 100
        xs, dists = sample_line(pos, vec, n_sample)

        c, sigma = model.bucket(xs.view(-1, 3))
        c, sigma = c.reshape(-1, n_sample, 1), sigma.reshape(-1, n_sample, 1)

        weight = (1 - torch.exp(-sigma))
        weight = weight / (torch.sum(weight, -2, keepdim=True) + 1e-6)

        final_x = (xs * weight).sum(-2)
        final_c = (c * weight).sum(-2)
        final_result = final_x * coef + torch.exp(-final_c * theta)
        buc_loss = (final_result - result) ** 2
        buc_loss = buc_loss.mean()

        # x, c0 = model(z)
        #
        # naive_result = x * coef + torch.exp(-c0 * theta)
        # naive_loss = ((naive_result - result) ** 2).mean()
        # loss = buc_loss + naive_loss

        x_, c0_ = torch.split(model.mlp(z), [3, 1], -1)
        naive_loss = ((x_ - final_x.detach()) ** 2).mean() + ((c0_ - final_x.detach()) ** 2).mean()
        loss = buc_loss # + naive_loss

        sp_loss = model.bucket(torch.rand(10000, 3, device=z.device))[1].abs().mean()
        loss = loss + sp_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            xc = torch.cat([final_x, final_c], -1)
            xc_gt = torch.cat([problem.x, problem.c], -1)
            diff = ((xc[..., None, :] - xc_gt).abs()).min(-2)[0].mean()
            pbar.set_postfix(Loss=loss.item(), L0=naive_loss.item(), L1=buc_loss.item(), Sp=sp_loss.item(),
                             xc_diff=diff.item())

        if i % 200 == 0:
            vis(lambda x: model.bucket(x)[1])


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
        z, coef, theta, result = problem.sample_data(5000)
        # index = problem.z_to_id(z).long()
        x, c = model(z, coef, result)

        final_result = x * coef + torch.exp(-c * theta)
        loss = (final_result - result) ** 2
        # loss = (index - c) ** 2

        KL = torch.tensor(0., device=z.device)
        # KL = kl_divergence(0.05, model.field(z))

        loss = loss.mean() + KL * 0.1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            xc = torch.cat([x, c], -1)
            xc_gt = torch.cat([problem.x, problem.c], -1)
            diff = ((xc[..., None, :] - xc_gt).abs()).min(-2)[0]
            pbar.set_postfix(Loss=loss.item(), KL=KL.item(), xc_diff=diff.mean().item())

        if i % 1000 == 0:
            # vis(lambda x: model(x)[1])
            # print(list(map(torch.Tensor.item, x[0])), list(map(torch.Tensor.item, c[0])))
            # print((model.field(z)[0] > 1e-3).long())
            pass


if __name__ == '__main__':
    improve_train()

