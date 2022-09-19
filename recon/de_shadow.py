import torch
from tqdm import trange
import matplotlib.pyplot as plt
import cv2
from recon.ipe import *


class SimpleCRF(torch.nn.Module):

    def __init__(self, gamma=2.2):
        super(SimpleCRF, self).__init__()
        if isinstance(gamma, torch.Tensor):
            self.s = gamma
        else:
            self.gamma = torch.nn.Parameter(torch.tensor(gamma))

    def forward(self, x):
        return torch.pow(x, 1.0 / self.gamma)

    def inv(self, x):
        return torch.pow(x, self.gamma)

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

    def __init__(self, img_path="dh.jpg"):
        img = cv2.imread(img_path) / 255.0
        img = torch.tensor(img).permute(2, 0, 1)[None].float().cuda()
        self.img = img

    def sample_data(self, n, noised=False):
        x = torch.rand(1, 1, n, 2).cuda()
        y = F.grid_sample(self.img, x, 'bilinear', 'zeros', True)
        y = y.view(3, -1).permute(1, 0)
        ofs = torch.ones_like(y) * 0.001
        return x.view(-1, 2), ofs, y

    def sample_test(self, n):
        i = torch.linspace(-1, 1, n).cuda()
        x = torch.stack(torch.meshgrid([i, i]), -1)[None]
        y = F.grid_sample(self.img, x, 'bilinear', 'zeros', True)
        y = y.view(3, -1).permute(1, 0)
        ofs = torch.ones_like(y) * 0.
        return x.view(-1, 2), ofs, y


class NaiveModel(torch.nn.Module):

    def __init__(self):
        super(NaiveModel, self).__init__()
        self.field = torch.nn.Sequential(
            SDFNetwork(d_out=32, d_in=2, d_hidden=128, multires=12, embed="PE"),
            torch.nn.Sigmoid()
        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 4),
        )

    def forward(self, z):
        rho_hat = self.field(z)
        return torch.split(self.mlp(rho_hat), [3, 1], -1)


class BRDFGatherModel(torch.nn.Module):

    def __init__(self, n_samples=100, feat_dim=3):
        super(BRDFGatherModel, self).__init__()
        self.n_samples = n_samples
        self.weight_act = lambda x: torch.nn.functional.softplus(x - 10)
        self.weight_field = SDFNetwork(d_in=3)
        self.feat_field = SDFNetwork(feat_dim)
        self.coef_weight_field = SDFNetwork(d_in=1, embed="PE")

        self.light = NaiveModel()

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
        return pos, ts

    def density(self, z):
        sigma = self.weight_field(z)
        sigma = self.weight_act(sigma)
        return sigma

    def forward(self, x, ofs=None, results=None, manual_noised=True):
        """

        :param x: unused identifiers (position)
        :param ofs: R/G/B in (t + R) * d_r + (t + G) * d_g + (t + B) * d_b = C
        :param results: C above
        :return: [d_r, d_g, d_b]
        """
        if ofs is None:
            return torch.ones_like(x) * 0.5, torch.ones_like(x)[..., :1] * 0.1

        if manual_noised:
            noise = torch.rand_like(ofs) * 0.00001
            ofs = ofs + noise

        n_sample = 50
        xs, ts = self.sample(ofs, results, n_sample)

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
        return final_x, self.light(x)[1], final_c, weight.sum(-2)[..., 0]   # self.brdf_spec(self.brdf_embed_fn(x.expand(-1, 3)))


def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
    rho = torch.tensor([rho] * len(rho_hat)).cuda()
    return torch.mean(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))


def vis(grid):
    if not hasattr(vis, "i"):
        vis.i = 0
    img = grid.permute(1, 0, 2).detach().cpu().numpy()
    cv2.imwrite(f"logs/dsd/vis-{vis.i}.png", img * 255)
    # plt.figure()
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # ind = torch.linspace(0, 1, 50)
    # z = torch.stack(torch.meshgrid(ind, ind, ind), -1).cuda().view(-1, 3)
    # x = f(z)
    #
    # z = z.view(50, 50, 50, 3)[:, :, 0].view(-1, 3)
    # x = x.view(50, -1, 1).sum(0)
    #
    # x = x.cpu().detach().numpy()
    # z = z.cpu().detach().numpy()
    # plt.scatter(z[..., 0], z[..., 1], c=x)
    # plt.savefig(f"logs/opt/vis-{vis.i}.png")
    vis.i += 1
    # plt.close()


def moving_train():
    torch.manual_seed(3)
    model = BRDFGatherModel()
    model.cuda()
    crf = SimpleCRF(1.0)
    crf.cuda()

    optimizer = torch.optim.Adam([{"lr": 0.0005, "params": model.parameters()},
                                  {"lr": 0.001, "params": crf.parameters()}], betas=(0.9, 0.99))
    problem = Problem()

    pbar = trange(30001)
    for i in pbar:
        z, ofs, result = problem.sample_data(10000)
        result = crf.inv(result)

        x, t, c, acc = model(z, ofs, result)

        final_result = x * (t + ofs)
        loss = ((final_result - result) ** 2).mean()
        # loss = (index - c) ** 2

        # KL = torch.tensor(0., device=z.device)
        rand_input = torch.rand(x.shape[0] * 50, x.shape[1], device=x.device)
        KL = model.density(rand_input).abs().mean()

        acc_loss = (acc - 1).abs().mean()

        total_loss = loss + KL * 0.1 + acc_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 10 == 0:
            pbar.set_postfix(Loss=loss.item(), KL=KL.item(), acc=acc_loss.item(), gamma=crf.gamma.item())

        if i % 500 == 0:
            with torch.no_grad():
                z, ofs, result = problem.sample_test(200)
                result = crf.inv(result)
                x, t, c, acc = model(z, ofs, result)
                final_result = x * (t + ofs)
                light = ((result - ofs * x) / (x + 1e-6))
                light = torch.sigmoid(light) * 2 - 1.0
            if i == 0:
                vis(result.view(200, 200, 3))
                vis(result.mean(-1, keepdim=True).view(200, 200, 1))
            else:
                vis(x.view(200, 200, 3))
                vis(light.view(200, 200, 3))
            # vis(lambda x: torch.clamp(model.density(x), max=1.0))
            # vis_1d(lambda x: model.weight_act(model.coef_weight_field(x)))
            pass


if __name__ == '__main__':
    torch.manual_seed(6)
    moving_train()

