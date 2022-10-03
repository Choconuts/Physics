import torch
from torch import nn
import torch.nn.functional as F
from cluster.third.model import PE


class Posterior(torch.nn.Module):

    def __init__(self, n_layers=2, width=32):
        super(Posterior, self).__init__()
        self.rgb_embed = PE(3, num_freq=4)
        layers = []
        in_dim = self.rgb_embed.feature_dim()
        for i in range(n_layers):
            layer = nn.Sequential(
                nn.Linear(in_dim, width),
                nn.ReLU(),
                nn.Linear(width, width),
            )
            in_dim = width + self.rgb_embed.feature_dim()
            layers.append(layer)
        self.n_layers = n_layers
        self.layers = nn.ModuleList(layers)
        self.out_linear = nn.Linear(width, 1)

    def forward(self, x, rgb_all, dirs_all, mask_all):
        """

        :param x: N, 3
        :param rgb_all: M, N, 3
        :param dirs_all: M, N, 3
        :param mask_all: M, N
        :return:
        """
        rgb_all = torch.where(mask_all[..., None].expand(rgb_all.shape), rgb_all, torch.ones_like(rgb_all))
        rgb_embed = self.rgb_embed(rgb_all)
        # inputs = torch.cat([rgb_all, dirs_embed], -1)
        inputs = rgb_embed
        h = inputs
        for i in range(self.n_layers):
            h = self.layers[i](h)
            h = h.mean(0)
            if i < self.n_layers - 1:
                h = torch.cat([inputs, h[None].expand(inputs.shape[0], -1, -1)], -1)
        h = self.out_linear(h)
        return h


if __name__ == '__main__':
    posterior = Posterior()
    x = torch.rand(2048, 3)
    rgb = torch.rand(100, 2048, 3)
    dirs = torch.rand(100, 2048, 3)
    mask = torch.rand(100, 2048) > 0.5
    a = posterior(x, rgb, dirs, mask)
    print(a)
