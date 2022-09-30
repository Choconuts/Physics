import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.rend_util import load_rgb


def vis_image(img, tag=""):
    from torchvision.io import write_png
    img = img.expand(3, -1, -1).cpu() * 255
    img = img.type(torch.uint8)
    write_png(img, f"vis/tmp{tag}.png")


def ln_var(x, sharpness=10):
    if isinstance(x, torch.Tensor):
        return -0.2 * torch.log(1 + torch.exp(sharpness * x - 3))
    return -np.log(1 + np.exp(sharpness * x - 3))


def plot(fn):
    x = np.linspace(0, 1, 200)
    y = fn(x)
    plt.plot(x, y)
    plt.show()


def show_ln_var():
    from cluster.oned import Gaussian

    xs = []
    for i in range(100):
        g = Gaussian(1, 0.0, i / 100.0)
        x = g.sample(1024)[..., 0]
        xs.append(x)

    xs = torch.stack(xs, 0)
    z = np.linspace(0, 1, 100)
    y = torch.exp(ln_var(xs).mean(-1))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.scatter(z, y.detach().cpu().numpy())
    plt.show()


def posterior_prob(rgb, mask):
    """

    :param rgb: [N, B, 3]
    :return:
    """
    mask_sum = torch.sum(mask[..., None], dim=0) + 1e-5
    mean_rgb = torch.sum(mask[..., None] * rgb, dim=0) / mask_sum
    mean_var = (ln_var((rgb - mean_rgb).norm(dim=-1)) * mask).sum(0) / mask_sum[..., 0]
    prob = torch.exp(mean_var * 2)
    return prob


if __name__ == '__main__':
    show_ln_var()
    selects = [20, 28, 63, 70, 73, 6, 36, 55]
    # selects = [k for k in range(100)]

    for k in range(5, 9):
        imgs = [torch.tensor(load_rgb(fr"G:\Repository\nerf-pytorch\data\nerf_synthetic\lego\train\r_{i}.png"))
                for i in selects[k - 5:k]]
        rgbs = torch.stack(imgs, 0)
        prob = posterior_prob(rgbs, torch.ones(rgbs.shape[:-1], dtype=bool))
        print(prob.max())
        vis_image(prob, k)


