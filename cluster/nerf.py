import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
from tqdm import trange
from cluster.syn_dataset import *
from cluster.focus_sampler import *
from cluster.alld import DepthGaussian, weight_fn, trans_int
from tensorf.tensoRF import TensorVM
from functional import *


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


class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=True, alpha_chan=1):
        """
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W

        self.embed, input_ch = get_embedder(10)
        self.view_embed, input_ch_views = get_embedder(4)
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        self.alpha_linear = nn.Linear(W, alpha_chan)
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def density(self, input_pts):
        input_pts = self.embed(input_pts)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        alpha = self.alpha_linear(h)
        return alpha

    def density_and_feature(self, input_pts):
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        return alpha, feature

    def color(self, input_pts, input_views, feature):
        assert self.use_viewdirs
        h = torch.cat([feature, input_views], -1)
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)
        return rgb

    def forward(self, input_pts, input_views):
        input_views = input_views.unsqueeze(-2).expand(input_pts.shape)

        input_pts = self.embed(input_pts)
        input_views = self.view_embed(input_views)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = F.softplus(alpha - 10), F.sigmoid(rgb)
        else:
            alpha = self.alpha_linear(h)
            sh = self.output_linear(h)
            outputs = alpha, sh

        return outputs


def uv_to_tex(uv):
    return torch.stack([(uv[..., 0] + 1) / 2 * data.img_res[0], (uv[..., 1] + 1) / 2 * data.img_res[1]], dim=-1)


def sample_gt_mask(uv):
    masks = torch.cat(data.object_masks, 0).view(-1, *data.img_res, 1).permute(0, 3, 1, 2)
    mask = F.grid_sample(masks.cuda().float(), uv.view(uv.shape[0], 1, -1, 2)).view(uv.shape[:-1])
    return mask > 0.5


def sample_gt(uv):
    images = torch.cat(data.rgb_images, 0).view(-1, *data.img_res, 3).permute(0, 3, 1, 2)
    rgb = F.grid_sample(images.cuda().float(), uv.view(uv.shape[0], 1, -1, 2)).view(uv.shape[0], 3, -1)
    return rgb.permute(0, 2, 1).contiguous()


def cast_ray(t, uv):
    pose = torch.stack(data.pose_all).cuda()
    intrinsics = torch.stack(data.intrinsics_all).cuda()
    uv_tex = uv_to_tex(uv)
    rays_d, rays_o = rend_util.get_camera_params(uv_tex, pose, intrinsics)
    x = rays_d[:, :, None, :] * t[:, :, :, None] + rays_o[:, None, None, :]
    return x, rays_d


def cast_ray_plot(t, uv, idx=0):
    pose = torch.stack(data.pose_all).cuda()[idx:idx+1]
    intrinsics = torch.stack(data.intrinsics_all).cuda()[idx:idx+1]
    uv_tex = uv_to_tex(uv.view(1, -1, 2))
    rays_d, rays_o = rend_util.get_camera_params(uv_tex, pose, intrinsics)
    x = rays_d[:, :, None, :] * t[:, :, :, None] + rays_o[:, None, None, :]
    return x, rays_d


def vis_image(img, tag=""):
    from torchvision.io import write_png
    img = img.expand(3, -1, -1).cpu() * 256
    img = img.type(torch.uint8)
    write_png(img, f"vis/tmp{tag}.png")


# torch.random.manual_seed(0)
#
# data = SynDataset(r"G:\Repository\nerf-pytorch\data\nerf_synthetic\lego", 10)
# focus_sampler = FocusSampler(data)
#
# nerf = NeRF()
# nerf.cuda()
# tensorf = TensorVM(torch.tensor([[-1.5] * 3, [1.5] * 3]).cuda(), [128] * 3, 'cuda', shadingMode="MLP_Fea")
# optimizer = torch.optim.Adam(params=tensorf.get_optparam_groups(), betas=(0.9, 0.999))
#
# batch_size = 2048
# sample_num = 256
# pbar = trange(10001)
# for i in pbar:
#     uv = torch.rand(data.n_cameras, batch_size, 2).cuda() * 2 - 1
#     mask = sample_gt_mask(uv)
#     rgb = sample_gt(uv)
#
#     t = torch.linspace(2.0, 6.0, sample_num)[None, None].cuda()
#     # x, rays_d = cast_ray(t, uv)
#     pose = torch.stack(data.pose_all).cuda()
#     intrinsics = torch.stack(data.intrinsics_all).cuda()
#     uv_tex = uv_to_tex(uv)
#     rays_d, rays_o = rend_util.get_camera_params(uv_tex, pose, intrinsics)
#     rays_o = rays_o.unsqueeze(1).expand(rays_d.shape)
#     rays_d = rays_d.reshape(-1 ,3)
#     rays_o = rays_o.reshape(-1 ,3)
#
#     # class Cam(Simulatable):
#     #     def __init__(self, n_seg=10):
#     #         super().__init__(n_seg)
#     #
#     #     def step(self, dt):
#     #         visualize_field(
#     #             rays_o.view(-1, 3)[:, :3] * 0.5,
#     #             vectors=rays_d.view(-1, 3) * 5
#     #         )
#     #
#     # Cam().run()
#
#     # rho, color = nerf(x, rays_d)
#     # alpha = 1.0 - torch.exp(-rho)[..., 0]
#     # weights = trans_int(alpha)
#     # out = (color * weights[..., None]).sum(-2)
#     color, depth = tensorf(torch.cat([rays_o, rays_d], -1), is_train=True, white_bg=True, ndc_ray=False, N_samples=128)
#
#     loss = (rgb.view(-1, 3) - color).abs().mean()
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if i % 10 == 0:
#         pbar.set_postfix({"Loss": loss.item()})
#     if i % 100 == 0:
#         with torch.no_grad():
#             s = torch.linspace(-1, 1, 100).cuda()
#             uv = torch.stack(torch.meshgrid([s, s]), -1)
#             uv_tex = uv_to_tex(uv)
#             rays_d, rays_o = rend_util.get_camera_params(uv_tex.view(1, -1, 2), pose[:1], intrinsics[:1])
#             rays_o = rays_o.unsqueeze(1).expand(rays_d.shape)
#             rays_d = rays_d.reshape(-1, 3)
#             rays_o = rays_o.reshape(-1, 3)
#             color, depth = tensorf(torch.cat([rays_o, rays_d], -1), is_train=True, white_bg=True, ndc_ray=False,
#                                    N_samples=128)
#             vis_image(color.view(*uv.shape[:-1], -1).permute(2, 0, 1), i)


if __name__ == '__main__':
    pass
