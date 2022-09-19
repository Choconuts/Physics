import torch.nn

from recon.dataset import *
from recon.ipe import *
from tqdm import trange, tqdm


class CCF(torch.nn.Module):

    def __init__(self, s=2.0):
        super(CCF, self).__init__()
        self.s = torch.nn.Parameter(torch.tensor(s))
        self.t_min = torch.nn.Parameter(torch.tensor(0.01))
        self.t_max = torch.nn.Parameter(torch.tensor(1.43))

    def forward(self, x):
        return x * torch.clamp(self.s, min=1e-4)

    def inv(self, x):
        return x / torch.clamp(self.s, min=1e-4)


class NaiveModel(torch.nn.Module):

    def __init__(self):
        super(NaiveModel, self).__init__()
        self.field = torch.nn.Sequential(
            SDFNetwork(d_out=16, d_in=2, d_hidden=128, multires=12, embed="PE"),
            torch.nn.Sigmoid()
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 4),
        )

    def forward(self, z):
        rho_hat = self.field(z)
        return torch.split(self.mlp(rho_hat), [3, 1], -1)


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

    # TODO: rng is important in current fixed t_range
    def sample(self, ofs, res, n, rng=True, lgt_min=1e-4, lgt_max=5.0, sharpness=6.0):
        t_min = torch.max(res - ofs, dim=-1)[0]
        l0 = torch.norm(res, dim=-1, keepdim=True)
        d = ofs.mean(-1, keepdim=True)
        s = torch.sigmoid(torch.linspace(-sharpness, sharpness, n, device=ofs.device))
        if rng:
            s = s + torch.rand_like(s) * (0.5 / n)
            s = torch.clamp(s, min=0.0, max=1.0)
        s = (l0 / (lgt_min[..., None] + d) - l0 / (lgt_max[..., None] + d)) * s
        ts = l0 / (l0 / (lgt_min[..., None] + d) - s) - d

        pos = res[..., None, :] / (ts[..., None] + ofs[..., None, :])
        valid = ts > t_min[..., None]

        if pos.isnan().any():
            print("[NAN]", pos.isnan().nonzero())
        return pos, ts, valid

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

    def forward(self, x, ofs=None, results=None, lgt_min=1e-4, lgt_max=5.0, manual_noised=True, sharpness=6.0):
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
        xs, ts, valid = self.sample(ofs, results, n_sample, lgt_min=lgt_min, lgt_max=lgt_max, sharpness=sharpness)

        c = self.feat_field(xs.view(-1, 3))
        sigma = self.weight_field(xs, 0.02)
        sigma = self.weight_act(sigma)

        c = c.reshape(-1, n_sample, c.shape[-1])
        sigma = sigma.reshape(-1, n_sample, 1)

        # weight = (1 - torch.exp(-sigma))
        # weight = weight / (torch.sum(weight, -2, keepdim=True) + 1e-6)
        alpha = (1.0 - torch.exp(-sigma))[..., 0]
        T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)
        weight = (alpha * T[:, :-1])[..., None]

        weight[~valid] = 0

        final_x = (xs * weight).sum(-2)
        final_t = (ts[..., None] * weight).sum(-2)
        return final_x, final_t, self.naive(x)[1], weight.sum(-2)[..., 0]  # self.brdf_spec(self.brdf_embed_fn(x.expand(-1, 3)))


def moving_train():
    torch.manual_seed(3)
    light = torch.nn.Sequential(
        SDFNetwork(d_out=1, d_in=2, d_hidden=256, multires=12, embed="IPE"),
        torch.nn.Sigmoid()
    )
    light.cuda()
    ccf = CCF()
    ccf.cuda()
    model = BRDFGatherModel()
    model.cuda()

    albedo = SDFNetwork(d_out=3, d_in=2, d_hidden=128, multires=12, embed="IPE")
    albedo.cuda()

    optimizer = torch.optim.Adam([{"lr": 0.0005, "params": model.parameters()},
                                  {"lr": 0.0005, "params": ccf.parameters()},
                                  {"lr": 0.0005, "params": albedo.parameters()},
                                  {"lr": 0.0001, "params": light.parameters()}], betas=(0.9, 0.99))

    data = LabData()

    vis_img(lambda x: data.sample_image(x, data.ind_light))

    T_var = 6.0
    pbar = trange(10001)
    for i in pbar:
        T_var *= 1.0001

        x, l, _, e, a, s, c = data.sample(5000)
        c = torch.clamp(c, min=1e-4)
        c = c * 1.43                                     # TODO: unknown param 1.43

        t_min = torch.ones_like(c[..., 0]) * torch.clamp(ccf.t_min, min=0.01)
        t_max = t_min + torch.clamp(ccf.t_max, min=0.1)

        z, t, _, acc = model(x, e, c, t_min, t_max, sharpness=T_var)

        final_result = z * (t + e)
        loss = ((final_result - c) ** 2).mean()

        rand_input = torch.rand(z.shape[0] * 50, z.shape[1], device=z.device)
        KL = model.density(rand_input, torch.rand_like(rand_input)).abs().mean()

        acc_loss = (acc - 1).abs().mean()

        fit_loss = ((albedo(x) - z.detach()) ** 2).mean()

        total_loss = loss + KL * 0.1 + acc_loss + fit_loss

        # grad = torch.autograd.grad(total_loss, model.parameters(), retain_graph=True)[0]
        # if grad.isnan().any():
        #     print("[NAN]", t_prox.isnan().nonzero())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 10 == 0:
            pbar.set_postfix(Loss=loss.item(), KL=KL.item(), acc=acc_loss.item(), tvar=T_var, t_min=ccf.t_min.item(), t_max=ccf.t_max.item())

        if i % 200 == 0:
            @batchify(10000)
            def sample_and_calc(x):
                with torch.no_grad():
                    c = data.sample_image(x, data.color)
                    c = torch.clamp(c, min=1e-4)
                    c = c * 1.43                # TODO: unknown param 1.43
                    e = data.sample_image(x, data.ind_light)
                    t_min = torch.ones_like(c[..., 0]) * 0.01
                    t_max = t_min + torch.clamp(ccf.t_max, min=0.1)

                    z, t, _, acc = model(x, e, c, t_min, t_max)
                    calc = c / (z + 1e-4) - e
                    return t, z
            vis_imgs(sample_and_calc, lambda x: data.sample_image(x, data.light), lambda x: data.sample_image(x, data.albedo))


if __name__ == '__main__':
    moving_train()



