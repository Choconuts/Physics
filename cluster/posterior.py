import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def ln_var(x, sharpness=10):
    if isinstance(x, torch.Tensor):
        return -torch.log(1 + torch.exp(sharpness * x - 5))
    return -np.log(1 + np.exp(sharpness * x - 5))


def plot(fn):
    x = np.linspace(0, 1, 200)
    y = fn(x)
    plt.plot(x, y)
    plt.show()


def posterior_prob(rgb):
    """

    :param rgb: [N, B, 3]
    :return:
    """
    mean_rgb = torch.mean(rgb, dim=0)
    prob = torch.exp(ln_var((rgb - mean_rgb).norm(dim=-1), sharpness=6).mean(0))
    return prob


if __name__ == '__main__':
    from cluster.oned import Gaussian

    xs = []
    for i in range(100):
        g = Gaussian(1, 0.0, i / 100)
        x = g.sample(20)[..., 0]
        xs.append(x)

    xs = torch.stack(xs, 0)
    z = np.linspace(0, 1, 100)
    y = torch.exp(ln_var(xs).mean(-1))
    plt.scatter(z, y.detach().cpu().numpy())
    plt.show()

