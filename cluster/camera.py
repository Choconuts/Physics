import imgui
import torch
import numpy as np
import torch.nn.functional as F
from utils import rend_util
from cluster.third.syn_dataset import SynDataset
from cluster.focus_sampler import FocusSampler
from interface import *
from cluster.third.neus_model import ImplicitNetworkMy


class NeRF:

    def __init__(self):
        self.neus = ImplicitNetworkMy()
        self.neus.cuda()

        from cluster.nerf import NeRF as NeRF0
        # self.tensorf = TensorVM(torch.tensor([[-1.5] * 3, [1.5] * 3]).cuda(), [128] * 3, 'cuda', shadingMode="MLP_Fea")
        # self.tensorf.cuda()
        self.nerf0 = NeRF0()
        self.nerf0.cuda()
        # self.optimizer = torch.optim.Adam(params=self.tensorf.get_optparam_groups(), betas=(0.9, 0.999))
        self.optimizer = torch.optim.Adam(params=self.nerf0.parameters(), lr=5e-5, betas=(0.9, 0.999))

    def __call__(self, rays_o, rays_d):
        t = torch.linspace(2.0, 6.0, 128).cuda()

        t_vals = t[None].expand(rays_d.shape[0], -1)
        t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])
        t_dists = t_vals[..., 1:] - t_vals[..., :-1]
        delta = t_dists * torch.linalg.norm(rays_d[..., None, :], dim=-1)
        # Note that we're quietly turning density from [..., 0] to [...].

        x = rays_d[:, None, :] * t_mids[:, :, None] + rays_o[:, None, :]
        density, color = self.nerf0(x, rays_d)

        density_delta = density[..., 0] * delta

        alpha = 1 - torch.exp(-density_delta)
        trans = torch.exp(-torch.cat([
            torch.zeros_like(density_delta[..., :1]),
            torch.cumsum(density_delta[..., :-1], dim=-1)
        ], dim=-1))
        weights = alpha * trans
        comp_rgb = (weights[..., None] * color).sum(dim=-2)
        acc = weights.sum(dim=-1)
        comp_rgb = comp_rgb + (1. - acc[..., None])

        return comp_rgb
        color, depth = self.tensorf(torch.cat([rays_o, rays_d], -1), is_train=True, white_bg=True, ndc_ray=False, N_samples=128)
        return color

    def density(self, x):
        return -self.neus(x)[..., :1]
        return self.nerf0.density(x)
        mask_outbbox = ((self.tensorf.aabb[0] > x) | (x > self.tensorf.aabb[1])).any(dim=-1)
        density = torch.ones_like(x[..., 0])
        sigma_feature = self.tensorf.compute_densityfeature(x[~mask_outbbox])
        validsigma = self.tensorf.feature2density(sigma_feature)
        density[~mask_outbbox] = validsigma
        return density

    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def match_score(all_rgb, all_mask, dirs, all_dirs, ):
    score = (all_dirs - dirs).norm(dim=-1)
    return score


def ln_var(x, sharpness=10):
    if isinstance(x, torch.Tensor):
        return -0.2 * torch.log(1 + torch.exp(sharpness * x - 3))
    return -np.log(1 + np.exp(sharpness * x - 3))


def posterior(all_rgb, all_mask, dirs, all_dirs, ):
    # score = match_score(all_rgb, all_mask, dirs, all_dirs)
    # _, idx = torch.sort(score, 0)
    # mask = all_mask.gather(0, idx[:12])
    # rgb = all_rgb.gather(0, idx[:12, :, None].expand(-1, -1, 3))
    # mask_sum = torch.sum(mask[..., None], dim=0) + 1e-5
    # mean_rgb = torch.sum(mask[..., None] * rgb, dim=0) / mask_sum
    # mean_var = (ln_var((rgb - mean_rgb).norm(dim=-1)) * mask).sum(0) / mask_sum[..., 0]
    # prob = torch.exp(mean_var * 2)

    # prob[mask.prod(0) == 0] = 0
    # return prob

    return all_mask.prod(0)


class Opt(Inputable):

    def __init__(self, z_val=0.0):
        self.x_val = z_val
        self.y_val = z_val
        self.z_val = z_val

    def gui(self, label, *args, **kwargs):
        self.changed, self.x_val = imgui.slider_float('x', self.x_val, -1.5, 1.5)
        self.changed, self.y_val = imgui.slider_float('y', self.y_val, -1.5, 1.5)
        self.changed, self.z_val = imgui.slider_float('z', self.z_val, -1.5, 1.5)


opt = Opt()


class Scene:

    def __init__(self, n_image=100):
        self.data = SynDataset(r"G:\Repository\nerf-pytorch\data\nerf_synthetic\lego", 100 // n_image)
        self.focus_sampler = FocusSampler(self.data)
        self.nerf = NeRF()

    def vis_axis(self):
        pnts = torch.stack(self.data.pose_all, 0)[:, :3, 3] * 0.5
        x, y, z = torch.zeros(3, *pnts.shape)
        x[..., 0] = 1
        y[..., 1] = 1
        z[..., 2] = -1
        dir_x = torch.stack(self.data.pose_all, 0)[:, :3, :3] @ x.view(-1, 3, 1)
        dir_y = torch.stack(self.data.pose_all, 0)[:, :3, :3] @ y.view(-1, 3, 1)
        dir_z = torch.stack(self.data.pose_all, 0)[:, :3, :3] @ z.view(-1, 3, 1) * 2
        visualize_field(pnts.numpy(), vectors={
            'x': torch.cat([dir_x[..., 0], x], -1).numpy(),
            'y': torch.cat([dir_y[..., 0], y], -1).numpy(),
            'z': torch.cat([dir_z[..., 0], z], -1).numpy(),
        }, len_limit=5.0)

    def vis_rays(self):
        selector = lambda arr: arr[50:51]

        pose = torch.stack(self.data.pose_all).cuda()
        intrinsics = torch.stack(self.data.intrinsics_all).cuda()
        s = torch.linspace(-1, 1, 100).cuda()
        uv = torch.stack(torch.meshgrid([s, s]), -1)
        uv_tex = self.uv_to_tex(uv)
        rays_d, rays_o = rend_util.get_camera_params(uv_tex.view(1, -1, 2), selector(pose), selector(intrinsics))
        rays_o = rays_o.unsqueeze(1).expand(rays_d.shape)
        rays_d = rays_d.reshape(-1, 3) * 2
        rays_o = rays_o.reshape(-1, 3)

        print(torch.norm(rays_o, dim=-1).mean())
        print(torch.norm(rays_d, dim=-1).mean())

        color = self.sample_gt(uv.view(1, -1, 2), selector)
        mask = self.sample_gt_mask(uv.view(1, -1, 2), selector)[0]
        rays_d[~mask] *= 0
        rays_d = torch.cat([rays_d, color.view(-1, 3)], -1)

        visualize_field(rays_o.cpu().numpy(), vectors={
            'x': rays_d.cpu().numpy()
        }, len_limit=-1)

    def vis_field(self, field, thr=0.1, res=100):
        with torch.no_grad():
            s = torch.linspace(-0.5, 0.5, res).cuda()
            sz = torch.linspace(opt.z_val, opt.z_val, 2).cuda()
            x = torch.stack(torch.meshgrid([s, s, s]), -1).view(-1, 3)
            pack = field(x)
            if isinstance(pack, tuple):
                a, rgb = pack
            else:
                a = pack
                rgb = x
            a = a.view(-1)
            rgb = rgb.view(-1 ,3)
            visualize_field(x[a > thr].cpu().detach(), scalars=rgb[a > thr].cpu().detach())

    @ui
    def show_rays(self):
        o, d, c, a = self.sample_batch()
        o = o[a]
        d = d[a] * 4
        c = c[a]

        visualize_field(o.cpu().numpy(), vectors={
            'x': torch.cat([d, c], -1).cpu().numpy()
        }, len_limit=-1)

    @ui
    def train(self):
        def field(x):
            dist = (torch.norm(x, dim=-1, keepdim=True) - 0.5) ** 2
            a = torch.exp(-dist * 50)
            return a

        for i in range(10000):
            x = torch.rand(4096, 3).cuda() * 3 - 1.5
            loss = (self.nerf.density(x) - field(x)).abs().mean()
            self.nerf.step(loss)

            print(loss.item())

            yield self.vis_field(self.nerf.density)

    @ui
    def train_nerf(self):
        for i in range(10000):
            rays_o, rays_d, rgb, mask = self.sample_batch(128)
            color = self.nerf(rays_o.view(-1, 3), rays_d.view(-1, 3))
            rgb[~mask] = 1.0
            loss = (color - rgb.view(-1, 3)).abs().mean()
            self.nerf.step(loss)

            print(loss.item())

            yield self.vis_field(self.nerf.density, 0.1)

    @ui(opt)
    def show_match(self):
        def field(x):
            sample, gt = self.focus_sampler.scatter_sample(x)
            mask = sample['object_mask']
            rgb = gt['rgb']
            prob = posterior(rgb, mask, x / (x.norm(dim=-1, keepdim=True) + 1e-5), sample['view_dir'])

            return prob, x

        while True:
            if opt.changed:
                yield self.vis_field(field, opt.x_val, res=60)
            else:
                yield

    @ui(opt)
    def show_focus(self):
        while True:
            if opt.changed:
                sample, gt = self.focus_sampler.scatter_sample(torch.tensor([opt.x_val, opt.y_val, opt.z_val]).cuda().view(1, 3))
                o = sample['pose'][:, :3, 3]
                d = sample['view_dir'][:, 0] * 4
                c = gt['rgb'][:, 0]

                a = sample['object_mask'][..., 0]
                o = o[a]
                d = d[a]
                c = c[a]

                visualize_field(o.cpu().numpy(), vectors={
                    'x': torch.cat([d, c], -1).cpu().numpy()
                }, len_limit=-1)
            yield

    @ui
    def show_neus(self):
        self.vis_rays()
        yield self.vis_field(self.nerf.density, 0.001)

    def uv_to_tex(self, uv):
        return torch.stack([(uv[..., 0] + 1) / 2 * self.data.img_res[0],
                            (uv[..., 1] + 1) / 2 * self.data.img_res[1]], dim=-1)

    def sample_gt_mask(self, uv, selector=None):
        if selector is not None:
            masks = torch.cat(selector(self.data.object_masks), 0).view(-1, *self.data.img_res, 1).permute(0, 3, 1, 2)
        else:
            masks = torch.cat(self.data.object_masks, 0).view(-1, *self.data.img_res, 1).permute(0, 3, 1, 2)
        mask = F.grid_sample(masks.cuda().float(), uv.view(uv.shape[0], 1, -1, 2)).view(uv.shape[:-1])
        return mask > 0.5

    def sample_gt(self, uv, selector=None):
        if selector is not None:
            images = torch.cat(selector(self.data.rgb_images), 0).view(-1, *self.data.img_res, 3).permute(0, 3, 1, 2)
        else:
            images = torch.cat(self.data.rgb_images, 0).view(-1, *self.data.img_res, 3).permute(0, 3, 1, 2)
        rgb = F.grid_sample(images.cuda().float(), uv.view(uv.shape[0], 1, -1, 2)).view(uv.shape[0], 3, -1)
        return rgb.permute(0, 2, 1).contiguous()

    def sample_batch(self, batch_size=5000):
        uv = torch.rand(self.data.n_cameras, batch_size // self.data.n_cameras, 2).cuda() * 2 - 1
        mask = self.sample_gt_mask(uv)
        rgb = self.sample_gt(uv)

        # v_uv = []
        # v_mask = []
        # v_rgb = []
        # for i in range(self.data.n_cameras):
        #     v_uv.append(uv[i][mask[i]][:batch_size])
        #     v_mask.append(mask[i][mask[i]][:batch_size])
        #     v_rgb.append(rgb[i][mask[i]][:batch_size])
        # uv = torch.stack(v_uv, 0)
        # mask = torch.stack(v_mask, 0)
        # rgb = torch.stack(v_rgb, 0)

        pose = torch.stack(self.data.pose_all).cuda()
        intrinsics = torch.stack(self.data.intrinsics_all).cuda()
        uv_tex = self.uv_to_tex(uv)
        rays_d, rays_o = rend_util.get_camera_params(uv_tex, pose, intrinsics)
        rays_o = rays_o.unsqueeze(1).expand(rays_d.shape)
        rays_d = rays_d.reshape(-1, 3)
        rays_o = rays_o.reshape(-1, 3)

        return rays_o, rays_d, rgb.view(-1, 3), mask.view(-1)


scene = Scene(100)
scene.show_match()


if __name__ == '__main__':
    pass
