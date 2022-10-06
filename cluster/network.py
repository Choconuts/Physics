import torch
from torch import nn
import torch.nn.functional as F
from cluster.third.model import PE
from cluster.third.neus_model import SingleVarianceNetwork


class Posterior(torch.nn.Module):

    def __init__(self, n_layers=3, width=128):
        super(Posterior, self).__init__()
        self.rgb_embed = PE(3, num_freq=12)
        layers = []
        norms = []
        in_dim = 100 * 3
        for i in range(n_layers):
            layer = nn.Sequential(
                nn.Linear(in_dim, width),
            )
            in_dim = width
            norms.append(nn.BatchNorm1d(width))
            layers.append(layer)
        self.n_layers = n_layers
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)
        # self.out_linear = nn.Linear(width, 1)
        self.out_linear = nn.Linear(width, 3)
        self.s = SingleVarianceNetwork(0.1)

    # def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=True, alpha_chan=1):
    #     """
    #     """
    #     super(Posterior, self).__init__()
    #     self.D = D
    #     self.W = W
    #     self.input_ch = input_ch
    #     self.input_ch_views = input_ch_views
    #     self.skips = skips
    #     self.use_viewdirs = use_viewdirs
    #
    #     self.pts_linears = nn.ModuleList(
    #         [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
    #                                     range(D - 1)])
    #
    #     ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
    #     self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])
    #
    #     ### Implementation according to the paper
    #     # self.views_linears = nn.ModuleList(
    #     #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
    #
    #     self.alpha_linear = nn.Linear(W, alpha_chan)
    #     if use_viewdirs:
    #         self.feature_linear = nn.Linear(W, W)
    #         self.rgb_linear = nn.Linear(W // 2, 3)
    #     else:
    #         self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, rgb_all, dirs_all, mask_all):
        """

        :param x: N, 3
        :param rgb_all: M, N, 3
        :param dirs_all: M, N, 3
        :param mask_all: M, N
        :return:
        """
        rgb_all[~mask_all] = 0
        inputs = rgb_all.permute(1, 0, 2).reshape(rgb_all.shape[1], -1)
        h = inputs
        for i in range(self.n_layers):
            h = self.layers[i](h)
            # h = self.norms[i](h)
            h = torch.relu(h)
        h = self.out_linear(h)

        # # classification process
        # h = F.softmax(h, dim=-1)
        return h

    # def forward(self, input_pts, rgb_all, dirs_all, mask_all):
    #     h = input_pts
    #     for i, l in enumerate(self.pts_linears):
    #         h = self.pts_linears[i](h)
    #         h = F.relu(h)
    #         if i in self.skips:
    #             h = torch.cat([input_pts, h], -1)
    #     alpha = self.alpha_linear(h)
    #     return alpha

    def l1(self, batch_size):
        return torch.tensor(0.).cuda()
        x = torch.rand(115, batch_size, 3).cuda()
        rgb_all = x
        dirs_all = x
        mask_all = torch.ones_like(x[..., 0]).bool()
        res = self(x, rgb_all, dirs_all, mask_all)
        return res.abs().mean() * 0.1


class EasyPosterior(torch.nn.Module):
    def __init__(self, n_layers=2, width=32):
        super(EasyPosterior, self).__init__()
        self.s = torch.nn.Parameter(torch.tensor(5.0))
        self.out_linear = nn.Linear(3, 1)

    def forward(self, x, rgb_all, dirs_all, mask_all):
        """

        :param x: N, 3
        :param rgb_all: M, N, 3
        :param dirs_all: M, N, 3
        :param mask_all: M, N
        :return:
        """
        rgb_all = rgb_all / (rgb_all.reshape(-1, 3).max(0)[0] - rgb_all.reshape(-1, 3).max(0)[0] + 1e-4)
        rgb_mean = rgb_all.mean(0)
        diff = self.activation((rgb_all - rgb_mean).abs())
        diff = diff.sum(0)
        likelihood = torch.exp(-diff)
        return self.out_linear(likelihood)

    def activation(self, diffs):
        x = torch.clamp(self.s, min=0.02) * diffs - 5
        return F.softplus(x)

    def l1(self, batch_size):
        return torch.tensor(0.).cuda()


if __name__ == '__main__':
    posterior = Posterior()
    x = torch.rand(2048, 3)
    rgb = torch.rand(115, 2048, 3)
    dirs = torch.rand(115, 2048, 3)
    mask = torch.rand(115, 2048) > 0.5
    a = posterior(x, rgb, dirs, mask)
    print(a)
