import imgui
import torch
import numpy as np
import torch.nn.functional as F
from utils import rend_util
from cluster.third.syn_dataset import SynDataset
from cluster.third.focus_sampler import FocusSampler
from interface import *


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

    rgb = (all_rgb * all_mask[..., None]).sum(0) / (all_mask[..., None].sum(0) + 0.0001)
    return all_mask.prod(0), rgb


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
        # self.data = SynDataset(r"G:\Repository\nerf-pytorch\data\nerf_synthetic\drums", 100 // n_image)
        self.focus_sampler = FocusSampler(self.data)

    def vis_axis(self, poses=None):
        if poses is None:
            poses = self.data.pose_all
        pnts = torch.stack(poses, 0)[:, :3, 3]
        x, y, z = torch.zeros(3, *pnts.shape)
        x[..., 0] = 1
        y[..., 1] = 1
        z[..., 2] = -1
        dir_x = torch.stack(poses, 0)[:, :3, :3] @ x.view(-1, 3, 1)
        dir_y = torch.stack(poses, 0)[:, :3, :3] @ y.view(-1, 3, 1)
        dir_z = torch.stack(poses, 0)[:, :3, :3] @ z.view(-1, 3, 1) * 2
        visualize_field(pnts.numpy(), vectors={
            'x': torch.cat([dir_x[..., 0], x], -1).numpy(),
            'y': torch.cat([dir_y[..., 0], y], -1).numpy(),
            'z': torch.cat([dir_z[..., 0], z], -1).numpy(),
        }, len_limit=5.0)

    def vis_rays(self, selector=None, res=100, far=2.0):
        if selector is None:
            selector = lambda arr: arr[0:1]

        pose = torch.stack(self.data.pose_all).cuda()
        intrinsics = torch.stack(self.data.intrinsics_all).cuda()
        s = torch.linspace(-1, 1, res).cuda()
        uv = torch.stack(torch.meshgrid([s, s], indexing='xy'), -1)
        pose_selected = selector(pose)
        uv = uv.view(1, -1, 2).expand(len(pose_selected), -1, -1)
        uv_tex = self.uv_to_tex(uv)

        rays_d, rays_o = rend_util.get_camera_params(uv_tex, pose_selected, selector(intrinsics))
        rays_o = rays_o.unsqueeze(1).expand(rays_d.shape)
        rays_d = rays_d.reshape(-1, 3) * far
        rays_o = rays_o.reshape(-1, 3)

        print(torch.norm(rays_o, dim=-1).mean())
        print(torch.norm(rays_d, dim=-1).mean())

        color = self.sample_gt(uv, selector)
        mask = self.sample_gt_mask(uv, selector).view(-1)
        rays_d[~mask] *= 0
        rays_d = torch.cat([rays_d, color.view(-1, 3)], -1)

        visualize_field(rays_o.cpu().numpy(), vectors={
            'x': rays_d.cpu().numpy()
        }, len_limit=-1)

    def vis_field(self, field, thr=0.1, res=100, radius=1.5):
        with torch.no_grad():
            s = torch.linspace(-radius, radius, res).cuda()
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

    @ui(opt)
    def show_solid(self):
        def field(x):
            sample, gt = self.focus_sampler.scatter_sample(x)
            mask = sample['object_mask']
            rgb = gt['rgb']
            prob = posterior(rgb, mask, x / (x.norm(dim=-1, keepdim=True) + 1e-5), sample['view_dir'])

            return prob

        self.vis_field(field, opt.x_val, res=80, radius=0.6)
        while True:
            if opt.changed:
                yield self.vis_field(field, opt.x_val, res=80, radius=0.6)
            else:
                yield

    @ui(opt)
    def show_solid_dense(self):

        out_x = None
        out_c = None

        for i in range(1024):
            x = torch.rand(64 * 64 * 64, 3).cuda() * 1.6 - 0.8
            sample, gt = self.focus_sampler.scatter_sample(x)
            mask = sample['object_mask']
            rgb = gt['rgb']
            p, c = posterior(rgb, mask, x / (x.norm(dim=-1, keepdim=True) + 1e-5), sample['view_dir'])

            x = x[p > 0.5]
            c = c[p > 0.5]
            if out_x is None:
                out_x = x
                out_c = c
            else:
                out_x = torch.cat([out_x, x], 0)
                out_c = torch.cat([out_c, c], 0)

        visualize_field(out_x.cpu().detach(), scalars=out_c.cpu().detach())
        # while True:
        #     if opt.changed:
        #         yield self.vis_field(field, opt.x_val, res=80, radius=0.6)
        #     else:
        #         yield

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

    def uv_to_tex(self, uv):
        return torch.stack([(uv[..., 0] + 1) / 2 * self.data.img_res[1],
                            (uv[..., 1] + 1) / 2 * self.data.img_res[0]], dim=-1)

    def sample_gt_mask(self, uv, selector=None):
        if selector is not None:
            masks = selector(torch.stack(self.data.object_masks, 0)).view(-1, *self.data.img_res, 1).permute(0, 3, 1, 2)
        else:
            masks = torch.cat(self.data.object_masks, 0).view(-1, *self.data.img_res, 1).permute(0, 3, 1, 2)
        mask = F.grid_sample(masks.cuda().float(), uv.view(uv.shape[0], 1, -1, 2)).view(uv.shape[:-1])
        return mask > 0.5

    def sample_gt(self, uv, selector=None):
        if selector is not None:
            images = selector(torch.stack(self.data.rgb_images, 0)).view(-1, *self.data.img_res, 3).permute(0, 3, 1, 2)
        else:
            images = torch.cat(self.data.rgb_images, 0).view(-1, *self.data.img_res, 3).permute(0, 3, 1, 2)
        rgb = F.grid_sample(images.cuda().float(), uv.view(uv.shape[0], 1, -1, 2)).view(uv.shape[0], 3, -1)
        return rgb.permute(0, 2, 1).contiguous()

    def sample_batch(self, batch_size=5000):
        uv = torch.rand(self.data.n_cameras, batch_size // self.data.n_cameras, 2).cuda() * 2 - 1
        mask = self.sample_gt_mask(uv)
        rgb = self.sample_gt(uv)

        pose = torch.stack(self.data.pose_all).cuda()
        intrinsics = torch.stack(self.data.intrinsics_all).cuda()
        uv_tex = self.uv_to_tex(uv)
        rays_d, rays_o = rend_util.get_camera_params(uv_tex, pose, intrinsics)
        rays_o = rays_o.unsqueeze(1).expand(rays_d.shape)
        rays_d = rays_d.reshape(-1, 3)
        rays_o = rays_o.reshape(-1, 3)

        return rays_o, rays_d, rgb.view(-1, 3), mask.view(-1)


if __name__ == '__main__':
    scene = Scene(100)
    scene.show_solid_dense()
