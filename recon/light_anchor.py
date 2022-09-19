import torch
from tqdm import trange
import matplotlib.pyplot as plt
from recon.ipe import *


class ZtoT(torch.nn.Module):

    def __init__(self, theta=(0.5, 0.2, 0.7)):
        super(ZtoT, self).__init__()
        self.theta = torch.nn.Parameter(torch.tensor(theta))
        self.cuda()

    def forward(self, z):
        y = (torch.sin(z * 7 + self.theta[2] * 5) * self.theta[0] + torch.cos(z * 12 + self.theta[1] * 5) * self.theta[1])
        return (y * 4 + 4 ** self.theta[2] + 2)[..., None]


class Problem:

    def __init__(self):
        self.ofs_ratio = 0.02
        self.x = torch.tensor([
            [0.2, 0.8, 0.1],
            [0.6, 0.1, 0.2],
            [0.4, 0.9, 0.7],
            [0.5, 0.1, 0.8],
            [0.2, 0.45, 0.35],
        ]).cuda()

        self.z_to_id = lambda z: ((torch.sin(z * 8) * 20 + torch.cos(z * 4) * 10 + z ** 5 * 8) % 5).long()
        self.z_to_d = lambda z: (torch.stack([
            (torch.sin(z * 7) * 6 + torch.cos(z * 6) * 4) * 0.1,
            (torch.sin(z * 8) * 15 + torch.cos(z * 9) * 5) * 0.05,
            (torch.sin(z * 9) * 8 + torch.cos(z * 7) * 2) * 0.1
        ], -1) * 0.5 + 0.5) * self.ofs_ratio
        self.z_to_t = ZtoT()

    def sample_data(self, n, noised=True):
        z = torch.rand([n], device=self.x.device)
        index = self.z_to_id(z)
        coef = self.z_to_t(z)
        ofs = self.z_to_d(z)
        if noised:
            noise = (torch.rand_like(ofs) * 0.4 - 0.2) * self.ofs_ratio
        else:
            noise = 0.

        result = (coef + ofs) * (self.x[index] + noise)

        return z[..., None], coef, ofs, result

    def sample_test(self, n, noised=False):
        z, coef, ofs, result = self.sample_data(n, noised)
        return z, ofs

    def gt_field(self, x):
        res = torch.exp(-((x[..., None, :] - self.x) ** 2).sum(-1) * 100)
        return res.sum(-1)


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

        self.coef_weight_field = SDFNetwork(d_in=1, embed="PE")

    def sample(self, ofs, res, n, rng=False, lgt_min=1e-4, lgt_max=5.0):
        # find min t
        t_min = torch.max(res - ofs, dim=-1)[0]
        if isinstance(lgt_min, torch.Tensor):
            t_min[t_min < lgt_min] = lgt_min[t_min < lgt_min]
        else:
            t_min = torch.clamp(t_min, min=lgt_min)
        if isinstance(lgt_max, torch.Tensor):
            t_max = lgt_max[..., None]
        else:
            t_max = lgt_max

        l0 = torch.norm(res, dim=-1, keepdim=True)
        d = ofs.mean(-1, keepdim=True)
        s = torch.linspace(0, 1, n, device=ofs.device)
        if rng:
            s = s + torch.rand_like(s) * (0.5 / n)
        s = (l0 / (t_min[..., None] + d) - l0 / (t_max + d)) * s
        ts = l0 / (l0 / (t_min[..., None] + d) - s) - d

        pos = res[..., None, :] / (ts[..., None] + ofs[..., None, :])

        assert not pos.isnan().any()
        return pos, ts

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

    def forward(self, x, ofs=None, results=None, lgt_min=1e-4, lgt_max=5.0, manual_noised=True):
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
        xs, ts = self.sample(ofs, results, n_sample, lgt_min=lgt_min, lgt_max=lgt_max)

        c = self.feat_field(xs.view(-1, 3))
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
        return final_x, self.naive(x)[1], final_c, weight.sum(-2)[..., 0]  # self.brdf_spec(self.brdf_embed_fn(x.expand(-1, 3)))


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
    plt.savefig(f"logs/lgt/vis-{vis.i}.png")
    vis.i += 1
    plt.close()


def vis_img(f):
    if not hasattr(vis, "i"):
        vis.i = 0
    plt.figure()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ind = torch.linspace(0, 1, 400)
    ind_z = torch.linspace(0, 1, 2)
    z = torch.stack(torch.meshgrid(ind, ind, ind_z), -1).cuda().view(-1, 3)
    x = f(z)

    z = z.view(400, 400, 2, 3)[:, :, 0].view(-1, 3)
    x = x.view(-1, 2, x.shape[-1]).mean(1)

    x = x.cpu().detach().numpy()
    z = z.cpu().detach().numpy()
    plt.scatter(z[..., 0], z[..., 1], c=x, s=0.01)
    plt.savefig(f"logs/opt/vis-{vis.i}.png")
    vis.i += 1
    plt.close()


def vis_1d(f):
    if not hasattr(vis, "i"):
        vis.i = 0
    plt.figure()
    plt.xlim(0, 1)
    plt.ylim(0, 30)
    z = torch.linspace(0, 1, 200).cuda()
    x = f(z[..., None])

    x = x.cpu().detach().numpy()
    z = z.cpu().detach().numpy()
    plt.scatter(z, x)
    plt.savefig(f"logs/opt/vis-1d-{vis.i}.png")
    plt.close()


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


def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
    rho = torch.tensor([rho] * len(rho_hat)).cuda()
    return torch.mean(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))


def moving_train():
    torch.manual_seed(3)
    # model = NaiveModel()
    model = BRDFGatherModel()
    model.cuda()

    z2t = ZtoT([0.5, 0.5, 0.5])
    T_var = 5.0

    optimizer = torch.optim.Adam([{"lr": 0.0005, "params": model.parameters()},
                                  {"lr": 0.0005, "params": z2t.parameters()}], betas=(0.9, 0.99))
    problem = Problem()

    vis(problem.gt_field)

    pbar = trange(30001)
    for i in pbar:
        T_var = T_var * 0.9995
        z, coef, ofs, result = problem.sample_data(5000)

        t_prox = z2t(z).squeeze()
        t_min = t_prox - T_var
        t_max = t_prox + T_var
        if T_var < 0.5:
            T_var = 0.5
            t_min = t_min.detach()
            t_max = t_max.detach()

        x, t, c, acc = model(z, ofs, result, t_min, t_max)

        final_result = x * (t + ofs)
        loss = ((final_result - result) ** 2).mean()
        # loss = (index - c) ** 2

        # KL = torch.tensor(0., device=z.device)
        rand_input = torch.rand(x.shape[0] * 50, x.shape[1], device=x.device)
        KL = model.density(rand_input, torch.rand_like(rand_input)).abs().mean()

        acc_loss = (acc - 1).abs().mean()

        total_loss = loss + KL * 0.1 + acc_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 10 == 0:
            x, t, c, acc = model(z, ofs, result)
            x_gt = problem.x[problem.z_to_id(z)][:, 0, :]
            x_diff = (x - x_gt).abs()
            loss = (x * (coef + ofs) - result) ** 2
            loss = loss.mean()

            diff = ((x[..., None, :] - problem.x).abs()).min(-2)[0]
            pbar.set_postfix(Loss=loss.item(), KL=KL.item(), xc_diff=diff.mean().item(), x_diff=x_diff.mean().item(), acc=acc_loss.item(), tvar=T_var)

        if i % 100 == 0:
            vis(lambda x: torch.clamp(model.density(x), max=1.0))
            # vis_1d(lambda x: model.weight_act(model.coef_weight_field(x)))
            # print(list(map(torch.Tensor.item, x[0])), list(map(torch.Tensor.item, c[0])))
            print(list(map(torch.Tensor.item, x[0])))
            # print((model.field(z)[0] > 1e-3).long())
            print(list(map(torch.Tensor.item, z2t.theta)))
            pass


def tst_problem(z_to_t: ZtoT):
    z = torch.linspace(0, 1, 100).cuda()
    t = z_to_t(z)
    z = z.detach().cpu().numpy()
    t = t.detach().cpu().numpy()
    plt.plot(z, t)
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(6)
    moving_train()

    # tst_problem(Problem().z_to_t)
    # tst_problem(ZtoT([0.5, 0.5, 0.5]))
