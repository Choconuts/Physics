import os
import torch
import imgui
import numpy as np
from tqdm import trange
from torchvision.io import write_png
import torch.nn.functional as F
from cluster.match import MatchScene, ui, SynDataset, FocusSampler, Inputable, visualize_field
from utils import rend_util
from cluster.third.model import MyNeRF, VNeRF
from cluster.third.sampler import sample_nerf
from cluster.network import Posterior


def torch_tree_map(fn, obj):
    if isinstance(obj, (list, tuple)):
        res = []
        for i, o in enumerate(obj):
            res.append(torch_tree_map(fn, o))
        try:
            return type(obj)(*res)
        except TypeError:
            return type(obj)(res)

    if isinstance(obj, dict):
        res = {}
        for k, o in obj.items():
            res[k] = torch_tree_map(fn, o)
        return res

    return fn(obj)


class Opt(Inputable):

    def __init__(self, z_val=0.0):
        self.t_val = 3.0
        self.x_val = z_val
        self.y_val = z_val
        self.z_val = z_val
        self.show = True

    def gui(self, label, *args, **kwargs):
        self.changed, self.show = imgui.checkbox("Show", self.show)
        self.changed, self.t_val = imgui.slider_float('t', self.t_val, -5.0, 5.0)
        self.changed, (self.x_val, self.y_val, self.z_val) = \
            imgui.slider_float3('x', self.x_val, self.y_val, self.z_val, -1.5, 1.5)


opt = Opt()


class ObserveScene(MatchScene):

    def __init__(self, n_image):
        self.data = SynDataset(r"E:\BaiduNetdiskDownload\colmap_result", 458 // n_image, blender=False)
        # self.data = SynDataset(r"G:\Repository\nerf-pytorch\data\nerf_synthetic\lego", 100 // n_image)
        self.focus_sampler = FocusSampler(self.data)
        self.nerf = VNeRF()
        self.nerf.cuda()
        self.posterior = Posterior()
        self.posterior.cuda()
        self.optimizer = torch.optim.Adam([{"params": self.nerf.parameters(), "lr": 5e-4},
                                           {"params": self.posterior.parameters(), "lr": 5e-4}], betas=(0.9, 0.99))

    def sample_uv(self, uv):
        pose = torch.stack(self.data.pose_all).cuda()
        intrinsics = torch.stack(self.data.intrinsics_all).cuda()
        uv_tex = self.uv_to_tex(uv)
        mask = self.sample_gt_mask(uv)
        rgb = self.sample_gt(uv)
        rays_d, rays_o = rend_util.get_camera_params(uv_tex, pose, intrinsics)
        rays_o = rays_o.unsqueeze(1).expand(rays_d.shape)
        rays_d = rays_d.reshape(-1, 3)
        rays_o = rays_o.reshape(-1, 3)

        return rays_o, rays_d, rgb.view(-1, 3), mask.view(-1)

    def raw2output(self, raw_rgb, raw_density, t_vals, rays_d,
                   rgb_activation=F.sigmoid,
                   density_bias=0.,
                   density_activation=F.relu,
                   rgb_padding=0.001,
                   white_bkgd=True):
        rgb = rgb_activation(raw_rgb)
        rgb = rgb * (1 + 2 * rgb_padding) - rgb_padding
        density = density_activation(raw_density + density_bias)

        t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])
        t_dists = t_vals[..., 1:] - t_vals[..., :-1]
        delta = t_dists * torch.linalg.norm(rays_d[..., None, :], dim=-1)
        # Note that we're quietly turning density from [..., 0] to [...].
        density_delta = density[..., 0] * delta

        alpha = 1 - torch.exp(-density_delta)
        trans = torch.exp(-torch.cat([
            torch.zeros_like(density_delta[..., :1]),
            torch.cumsum(density_delta[..., :-1], dim=-1)
        ], dim=-1))
        weights = alpha * trans

        comp_rgb = (weights[..., None] * rgb).sum(dim=-2)
        acc = weights.sum(dim=-1)
        distance = (weights * t_mids).sum(dim=-1) / acc
        distance = torch.clip(torch.nan_to_num(distance, torch.inf), t_vals[:, 0], t_vals[:, -1])
        if white_bkgd:
            comp_rgb = comp_rgb + (1. - acc[..., None])

        return comp_rgb, weights, acc, distance

    def save_image(self, img, tag=None):
        if img.shape[-1] == 3:
            img = img.permute(2, 0, 1)
        save_root = f"vis/{self.__class__.__name__.lower().replace('scene', '')}"
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        img = img.expand(3, -1, -1).cpu() * 255
        img = img.type(torch.uint8)
        save_name = "img.png" if tag is None else f"img-{tag}.png"
        write_png(img, os.path.join(save_root, save_name))

    def query_model(self, x, rays_d, t_vals):
        raw_rgb, raw_a = self.nerf(x, rays_d)
        dirs = rays_d[..., None, :].expand(x.shape).reshape(-1, 3)
        rgb_obs, mask_obs = self.sample_neighbor(x.view(-1, 3), dirs, -1)
        raw_a = self.posterior(x, rgb_obs, dirs, mask_obs)
        raw_a = raw_a.view(*raw_rgb.shape[:-1], 1)
        rgb, weights, acc, distance = self.raw2output(raw_rgb, raw_a, t_vals, rays_d, white_bkgd=False)
        return rgb, weights, acc, distance

    def volume_render(self, rays_o, rays_d, num_sample=128):
        t_vals, x = sample_nerf(rays_o, rays_d)
        rgb, weights, acc, distance = self.query_model(x, rays_d, t_vals)
        t_vals, x = sample_nerf(rays_o, rays_d, t_vals, weights, n_sample=num_sample)
        rgb, weights, acc, distance = self.query_model(x, rays_d, t_vals)

        return rgb

    @ui
    def show_sample(self):
        batch_size = 2048
        num_sample = 128
        rays_o, rays_d, rgb_gt, mask = self.sample_batch(batch_size)
        t_vals, x = sample_nerf(rays_o, rays_d)
        raw_rgb, raw_a = self.nerf(x, rays_d)
        rgb, weights, acc, distance = self.raw2output(raw_rgb, raw_a, t_vals, rays_d)
        t_vals, x = sample_nerf(rays_o, rays_d, t_vals, weights, n_sample=num_sample)
        rgb, weights, acc, distance = self.raw2output(raw_rgb, raw_a, t_vals, rays_d)

        visualize_field(x.view(-1, 3), x.view(-1, 3))

    @ui(opt)
    def train_nerf(self):
        chunk = 8192
        batch_size = 2048 // 4
        num_sample = 128

        def field(x):
            dirs = -x / (x.norm(dim=-1, keepdim=True) + 1e-5)
            rgb, a = self.nerf(x, dirs)
            return a, torch.sigmoid(rgb)

        pbar = trange(100001)
        for i in pbar:
            rays_o, rays_d, rgb_gt, mask = self.sample_batch(batch_size)
            rgb = self.volume_render(rays_o, rays_d, num_sample=num_sample)

            mask = mask[..., None]
            mask_sum = mask.sum() + 1e-5
            color_error = (rgb - rgb_gt) * mask
            color_fine_loss = F.mse_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((rgb - rgb_gt) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())
            loss = color_fine_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if i % 10 == 0:
                pbar.set_postfix({"Loss": loss.item(), "PSNR": psnr.item()})
            if opt.changed or i % 100 == 0:
                self.vis_field(field, opt.t_val, radius=1.5)
            # if i % 1000 == 50:
            #     idx = np.random.choice(self.data.n_cameras, []).item()
            #     data_all = self.sample_image(idx, res=-1)
            #
            #     results = []
            #     for j in range(0, data_all[0].shape[0], chunk):
            #         chunk_data = torch_tree_map(lambda r: r[j:j + chunk], data_all)
            #         with torch.no_grad():
            #             chunk_results = self.volume_render(chunk_data[0], chunk_data[1], num_sample=num_sample)
            #         ret = torch_tree_map(lambda x: x, chunk_results)
            #         results.append(ret)
            #
            #     img = torch.cat(results, 0)
            #     img = img.view(*self.data.img_res, opt.t_val)
            #     self.save_image(img, i)
            yield


if __name__ == '__main__':
    scene = ObserveScene(100)
    scene.train_nerf()


