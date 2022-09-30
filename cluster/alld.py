import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import trange


class DepthGaussian(torch.nn.Module):

    def __init__(self, n_image, H, W, init_mu=3.0, init_sigma=1.0):
        super(DepthGaussian, self).__init__()
        self.n = n_image
        self.h = H
        self.w = W
        data = torch.ones(n_image, 2, H, W)
        data[:, 0, :, :] = init_mu
        data[:, 1, :, :] = 1.0 / init_sigma
        self.mu_sigma = torch.nn.Parameter(data)

    def forward(self, uv):
        """

        :param uv: [M, B, N, 2 ]
        :return: [M, B, N, 1], [M, B, N, 1]
        """
        mu_sigma = F.grid_sample(self.mu_sigma, uv).view(uv.shape[0], 2, -1)
        mu = mu_sigma[:, 0]
        sigma = mu_sigma[:, 1]
        return mu.view(*uv.shape[:-1], -1), 1.0 / torch.clamp(sigma, min=0.001).view(*uv.shape[:-1], -1)

    def probability(self, x, uv):
        uv = uv.view(uv.shape[0], uv.shape[1], -1, 2)
        mu, sigma = self.forward(uv)
        power = (- (x - mu) ** 2 / (2 * sigma ** 2)).sum(-1)
        return 1.0 / (np.sqrt(2 * np.pi) * sigma).prod(-1) * torch.exp(power)

    def sample(self, n, uv):
        uv = uv.view(uv.shape[0], uv.shape[1], -1, 2)
        mu, sigma = self.forward(uv)
        x = torch.randn(uv.shape[0], uv.shape[1], n, mu.shape[-1]).to(mu.device)
        return x * sigma + mu


def plot_fn(fn):
    idx = torch.randint(100, [1]).cuda()
    uv = torch.rand(1, 2).cuda() * 2 - 1
    x = np.linspace(0, 10, 500)
    y = fn.probability(torch.tensor(x)[..., None].cuda(), idx, uv)
    plt.plot(x, y.cpu().detach().numpy())
    plt.show()


def vis_image(fn, tag=""):
    img = fn.mu_sigma[-1, 1:]
    from torchvision.io import write_png
    img = img.expand(3, -1, -1).cpu() * 40
    img = img.type(torch.uint8)
    write_png(img, f"vis/tmp{tag}.png")


def trans_int(probs):
    assert (probs <= 1).all()
    alpha = probs       # torch.exp(-density_delta)
    trans = torch.cat([
        torch.ones_like(probs[..., :1]),
        torch.cumprod(1 - probs[..., :-1] + 1e-4, dim=-1)
    ], dim=-1)
    weights = alpha * trans

    return weights


def weight_fn(x, uv):
    d = 4.0 - torch.norm(uv, dim=-1) ** 2
    return torch.exp(-(x - d[..., None]) ** 2 * 5).detach()


if __name__ == '__main__':
    image_num = 100
    batch_size = 2048
    sample_num = 64
    resolution = 256
    far = 6.0
    g = DepthGaussian(image_num, resolution, resolution)
    g.cuda()

    optimizer = torch.optim.Adam(params=g.parameters(), lr=0.005)

    pbar = trange(10001)
    for i in pbar:
        uv = torch.rand(image_num, batch_size, 2).cuda() * 2 - 1
        t = g.sample(sample_num + 1, uv)
        t = torch.sort(t[..., 0], dim=-1)[0]

        dist = t[..., 1:] - t[..., :-1]
        mid_t = (t[..., 1:] + t[..., :-1]) / 2

        norm_prob = g.probability(mid_t[..., None].detach(), uv) * dist.detach()

        q = weight_fn(mid_t, uv)
        w = trans_int(norm_prob * q)

        loss = ((w.sum(-1) - 1.0) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            pbar.set_postfix({"Loss": loss.item()})
        if i % 200 == 0:
            plt.xlim(0, far)
            # plt.plot(t.cpu().detach().numpy(), w.cpu().detach().numpy())
            # plt.show()
            print(w.max().item(), w.sum().item())
            # plot_fn(g)
            vis_image(g)

