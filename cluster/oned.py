import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import trange


class Gaussian(torch.nn.Module):

    def __init__(self, shape, init_mu=0.0, init_sigma=1.0):
        super(Gaussian, self).__init__()
        self.mu = torch.nn.Parameter(torch.ones(shape) * init_mu)
        self.sigma = torch.nn.Parameter(torch.ones(shape) * init_sigma)

    def forward(self, x):
        assert x.shape[-1] == self.mu.shape[-1]
        sigma = 1.0 / torch.clamp(self.sigma, min=0.001)
        power = (- (x - self.mu) ** 2 / (2 * sigma ** 2)).sum(-1)
        return 1.0 / (np.sqrt(2 * np.pi) * sigma).prod(-1) * torch.exp(power)

    def sample(self, n):
        x = torch.randn(n, self.mu.shape[-1]).to(self.mu.device)
        sigma = 1.0 / torch.clamp(self.sigma, min=0.001)
        return x * sigma + self.mu


def plot_fn(fn, tag):
    x = np.linspace(0, 10, 2000)
    y = fn(torch.tensor(x)[..., None].cuda())
    plt.figure()
    plt.xlim(0, 6)
    plt.ylim(0, 2)
    plt.plot(x, y.cpu().detach().numpy())
    plt.savefig(f"vis/tmp{tag}.png")


def trans_int(probs):
    assert (probs <= 1).all()
    alpha = probs       # torch.exp(-density_delta)
    trans = torch.cat([
        torch.ones_like(probs[..., :1]),
        torch.cumprod(1 - probs[..., :-1], dim=-1)
    ])
    weights = alpha * trans

    return weights


def weight_fn(x):
    return torch.exp(-(x - 2.1) ** 2 * 5).detach()


if __name__ == '__main__':
    g = Gaussian(1, 3.0, 1.0)
    g.cuda()
    batch_size = 128
    far = 6.0

    optimizer = torch.optim.Adam(params=g.parameters(), lr=0.005)

    plt.xlim(0, far)
    plot_fn(weight_fn, "")

    pbar = trange(5001)
    for i in pbar:
        t = g.sample(batch_size + 1)
        t = torch.sort(t[..., 0])[0]
        dist = t[1:] - t[:-1]
        mid_t = (t[1:] + t[:-1]) / 2

        norm_prob = g(mid_t[..., None].detach()) * dist.detach()

        q = weight_fn(mid_t)
        w = trans_int(norm_prob * q)

        loss = ((w.sum(-1) - 1) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            pbar.set_postfix({"Loss": loss.item()})
        if i % 1000 == 0:
            plt.xlim(0, far)
            # plt.plot(t.cpu().detach().numpy(), w.cpu().detach().numpy())
            # plt.show()
            plot_fn(g, i)

