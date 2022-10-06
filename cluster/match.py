import torch
import imgui
import numpy as np
from cluster.camera import Scene, ui, SynDataset, FocusSampler, Inputable, visualize_field
from utils import rend_util
from cluster.third.model import MyNeRF
from cluster.third.neus_model import ImplicitNetworkMy


def ln_var(x, sharpness=6):
    if isinstance(x, torch.Tensor):
        return -0.2 * torch.log(1 + torch.exp(sharpness * x - 3))
    return -np.log(1 + np.exp(sharpness * x - 3))


def mask_expect_fn(x, expect=40):
    return torch.sigmoid(12 / expect * (x - expect * 0.6))


class Opt(Inputable):

    def __init__(self, t=0.0, x=0.0, y=0.0, z=0.0):
        self.t_val = t
        self.x_val = x
        self.y_val = y
        self.z_val = z
        self.show = True

    def gui(self, label, *args, **kwargs):
        self.changed, self.show = imgui.checkbox("Show", self.show)
        self.changed, self.t_val = imgui.slider_float('t', self.t_val, 0.2, 5.0)
        self.changed, (self.x_val, self.y_val, self.z_val) = \
            imgui.slider_float3('x', self.x_val, self.y_val, self.z_val, -1.5, 1.5)


opt = Opt(t=2.5, x=0.27)


class MatchScene(Scene):

    def __init__(self, n_image):
        self.data = SynDataset(r"E:\BaiduNetdiskDownload\colmap_result", 458 // n_image, blender=False)
        # self.data = SynDataset(r"G:\Repository\nerf-pytorch\data\nerf_synthetic\lego", 100 // n_image)
        self.focus_sampler = FocusSampler(self.data)

    def sample_gt_mask(self, uv, selector=None):
        return torch.ones_like(uv[..., 0]).bool()

    def sample_image(self, idx, res=100):
        if isinstance(idx, int):
            selector = lambda arr: arr[idx:idx+1]
        else:
            selector = idx

        pose = torch.stack(self.data.pose_all).cuda()
        intrinsics = torch.stack(self.data.intrinsics_all).cuda()
        if res > 0:
            s = torch.linspace(-1, 1, res).cuda()
            uv = torch.stack(torch.meshgrid([s, s], indexing="xy"), -1).view(-1, 2)
        else:
            sx = torch.linspace(-1, 1, self.data.img_res[0]).cuda()
            sy = torch.linspace(-1, 1, self.data.img_res[1]).cuda()
            uv = torch.stack(torch.meshgrid([sy, sx], indexing="xy"), -1).view(-1, 2)
        pose_selected = selector(pose)
        uv = uv.view(1, -1, 2).expand(len(pose_selected), -1, -1)
        uv_tex = self.uv_to_tex(uv)
        mask = self.sample_gt_mask(uv, selector)
        rgb = self.sample_gt(uv, selector)
        rays_d, rays_o = rend_util.get_camera_params(uv_tex, pose_selected, selector(intrinsics))
        rays_o = rays_o.unsqueeze(1).expand(rays_d.shape)
        rays_d = rays_d.reshape(-1, 3)
        rays_o = rays_o.reshape(-1, 3)

        return rays_o, rays_d, rgb.view(-1, 3), mask.view(-1)

    def sample_neighbor(self, x, dirs, best_n=20):
        sample, gt = self.focus_sampler.scatter_sample(x)
        all_mask = sample['object_mask']
        all_rgb = gt['rgb']
        all_dirs = sample['view_dir']

        all_mask[(all_dirs * dirs).sum(-1) < 0] = 0

        if best_n <= 0:
            return all_rgb, all_mask

        score = (all_dirs - dirs).norm(dim=-1)
        _, idx = torch.sort(score, 0)
        mask = all_mask.gather(0, idx[:best_n])
        rgb = all_rgb.gather(0, idx[:best_n, :, None].expand(-1, -1, 3))
        return rgb, mask

    def likelihood(self, x, obs_dirs, obs_rgb=None, degree=2):
        rgb, mask = self.sample_neighbor(x, obs_dirs, -1)
        mask = mask[..., None]
        mask_exp = mask.sum(0)[..., 0]

        if obs_rgb is None:
            obs_rgb = torch.sum(rgb * mask, dim=0) / (mask.sum(0) + 1e-4)
        mask_prob = mask_expect_fn(mask_exp, expect=self.data.n_cameras / 2.5)
        mean_var = (ln_var((rgb - obs_rgb).norm(dim=-1)) * mask[..., 0]).sum(0)
        prob = torch.exp(mean_var * degree)
        prob = prob * mask_prob

        if prob.isnan().any():
            print(prob.isnan().nonzero())

        return prob

    def var(self, x, obs_dirs, obs_rgb=None, degree=2):
        rgb, mask = self.sample_neighbor(x, obs_dirs, -1)

        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-5)

        mask = mask[..., None]
        mask_exp = mask.sum(0)[..., 0]

        if obs_rgb is None:
            obs_rgb = torch.sum(rgb * mask, dim=0) / (mask.sum(0) + 1e-4)
        mask_prob = mask_expect_fn(mask_exp, expect=self.data.n_cameras / 2.5)
        mean_var = ((rgb - obs_rgb) ** 2 * mask).sum(0) / (mask.sum(0) + 1e-4)

        return mean_var.mean(-1)

    def scatter_rgb(self, x, obs_dirs, obs_rgb=None, degree=2):

        x = torch.stack([x[-1], x[-1] + obs_dirs[-1] * 0.01], 0)

        rgb, mask = self.sample_neighbor(x, obs_dirs[-2:], -1)

        rgb = (rgb - 0.05) / 0.6

        return rgb[:, -1], mask[:, 1]

    def vis_inv_rays(self, x):
        sample, gt = self.focus_sampler.scatter_sample(x.view(1, 3))
        all_dirs = sample['view_dir'][:, 0]
        all_rgb = gt['rgb'][:, 0]
        all_rgb = (all_rgb - 0.05) / 0.5

        all_mask = sample['object_mask'][:, 0]
        cam = self.focus_sampler.cam_loc

        t = torch.linspace(0.1, 5.0, 256).cuda()[:, None]
        xx = cam[:, None, :] + t * all_dirs[:, None, :]
        _, alpha = self.my_nerf(xx, all_dirs)
        my_mask = (alpha.view(*xx.shape[:-1]) > 3.0).cumsum(-1).bool()
        last = xx[~my_mask][-1]
        rays = all_dirs * torch.norm(last - cam, dim=-1, keepdim=True)

        rays = torch.cat([rays, all_rgb], -1)
        visualize_field(cam[all_mask].cpu().numpy(), vectors={
            'x': rays[all_mask].cpu().numpy()
        }, len_limit=-1)

    @ui(opt)
    def show_match_deprecated(self):
        def field(x):
            sample, gt = self.focus_sampler.scatter_sample(x)
            all_mask = sample['object_mask']
            all_rgb = gt['rgb']
            all_dirs = sample['view_dir']
            dirs = x / (x.norm(dim=-1, keepdim=True) + 1e-5)

            score = (all_dirs - dirs).norm(dim=-1)
            _, idx = torch.sort(score, 0)
            mask = all_mask.gather(0, idx[:10])
            rgb = all_rgb.gather(0, idx[:10, :, None].expand(-1, -1, 3))
            mean_rgb = torch.mean(rgb, dim=0)
            mean_var = torch.var(rgb, dim=0)
            # mean_var = (ln_var((rgb - mean_rgb).norm(dim=-1))).mean(0)
            # prob = torch.exp(mean_var * 2)

            return -mean_var, mean_rgb

        while True:
            if opt.changed:
                self.vis_field(field, opt.x_val, res=60)
            yield

    @ui(opt)
    def show_best(self):

        while True:
            if opt.changed:
                best_n = 20
                x = torch.rand(50000, 3).cuda().view(-1, 3) * 2 - 1
                x[:, 2] += 0.5
                # x[:, 0] = opt.x_val
                dirs = -x / (x.norm(dim=-1, keepdim=True) + 1e-5)

                rgb, mask = self.sample_neighbor(x, dirs, best_n)
                mask = mask[..., None]

                mean_rgb = torch.sum(rgb * mask, dim=0) / mask.sum(0)
                mask_exp = mask.sum(0)[..., 0]
                mask_prob = mask_expect_fn(mask_exp)
                mean_var = (ln_var((rgb - mean_rgb).norm(dim=-1)) * mask[..., 0]).sum(0)
                # mean_var = (ln_var((rgb - rgb0).norm(dim=-1))).mean(0)
                prob = torch.exp(mean_var)
                prob = prob * mask_prob

                show_mask = prob > opt.t_val / 5
                if show_mask.any():
                    visualize_field(x[show_mask], scalars=mean_rgb[show_mask])
                self.vis_axis()
                # self.vis_rays(lambda x: x[idx[2:3, 0],], 20)
            yield

    @ui(opt)
    def show_hit(self):

        while True:
            if opt.changed:
                rays_o, rays_d, rgb0, mask0 = self.sample_image(2)
                x = rays_o + rays_d * opt.t_val

                prob = self.likelihood(x, rays_d, rgb0)

                visualize_field(x, scalars=prob)
                self.vis_axis()
                # self.vis_rays(lambda x: x[idx[2:3, 0],], 20)
            yield

    @ui(opt)
    def show_heat(self):
        self.my_nerf = MyNeRF()
        def field(x):
            dirs = -x / (x.norm(dim=-1, keepdim=True) + 1e-5)
            rgb, a = self.my_nerf(x, dirs)
            return a, torch.sigmoid(rgb)
        while True:
            if opt.changed:
                pose_id = 2
                rays_o, rays_d, rgb0, mask0 = self.sample_image(pose_id, res=80)
                t = torch.linspace(0.1, 2.5, 80).cuda()[:, None]
                x = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * t
                rgb0 = rgb0.unsqueeze(-2).expand(x.shape).reshape(-1, 3)

                prob = self.likelihood(x.reshape(-1, 3), rays_d.unsqueeze(-2).expand(x.shape).reshape(-1, 3))

                x = x.reshape(-1, 3)[prob > opt.t_val / 5]
                self.vis_axis(self.data.pose_all[pose_id:pose_id + 1])
                self.vis_field(field, 3, radius=1.5)
                if opt.show:
                    visualize_field(x, scalars=rgb0[prob > opt.t_val / 5])
            yield

    @ui(opt)
    def show_nerf(self):
        self.my_nerf = MyNeRF()

        def field(x):
            dirs = -x / (x.norm(dim=-1, keepdim=True) + 1e-5)
            rgb, a = self.my_nerf(x, dirs)
            return a, torch.sigmoid(rgb)

        while True:
            if opt.changed:
                self.vis_field(field, 3, radius=1.5)
                self.vis_rays(far=5.0)
            yield

    @ui(opt)
    def show_length(self):
        self.my_nerf = MyNeRF()
        self.my_neus = ImplicitNetworkMy()
        self.my_neus.cuda()

        def field(x):
            dirs = -x / (x.norm(dim=-1, keepdim=True) + 1e-5)
            rgb, a = self.my_nerf(x, dirs)
            return a, torch.sigmoid(rgb)

        def field_sdf(x):
            dirs = -x / (x.norm(dim=-1, keepdim=True) + 1e-5)
            sdf = self.my_neus(x, dirs)[..., :1]
            return -sdf, x

        # self.vis_field(field, 3, radius=1.5)

        while True:
            if opt.changed:

                selector = lambda arr: arr[1:2]
                pose = torch.stack(self.data.pose_all).cuda()
                intrinsics = torch.stack(self.data.intrinsics_all).cuda()
                pose_selected = selector(pose)
                uv = torch.tensor([[[opt.x_val, opt.y_val]]]).expand(len(pose_selected), -1, -1).cuda()
                uv_tex = self.uv_to_tex(uv)

                rays_d, rays_o = rend_util.get_camera_params(uv_tex, pose_selected, selector(intrinsics))
                rays_o = rays_o.unsqueeze(1).expand(rays_d.shape)

                rays_d = rays_d.reshape(-1, 3)
                rays_o = rays_o.reshape(-1, 3)

                color = self.sample_gt(uv, selector)
                mask = self.sample_gt_mask(uv, selector).view(-1)
                rays_d[~mask] *= 0
                # color_rays_d = torch.cat([rays_d, color.view(-1, 3)], -1)

                t = torch.linspace(0.1, (opt.t_val * 0.2 + 1.4), 64).cuda()[:, None]
                x = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * t

                rgb, msk = self.scatter_rgb(x.reshape(-1, 3), rays_d.unsqueeze(-2).expand(x.shape).reshape(-1, 3))
                # rgb = rgb[rgb.sum(-1) < 0.1] * 10
                ids = torch.linspace(0, 1, rgb.shape[0]).cuda()

                my_mask = (field(x.view(-1, 3))[0].view(*x.shape[:-1]) > 3.0).cumsum(-1).bool()
                if ~my_mask[0, -2] and my_mask[0, -1]:
                    ids[:] = 1

                # x = x[~my_mask]
                # prob = prob[~my_mask[0]]

                if not opt.show:
                    # self.vis_field(field, 3, radius=1.5)
                    self.vis_field(field_sdf, 0, radius=1.5)
                else:
                    visualize_field(x[:1, -1], x[:1, -1])

                self.vis_inv_rays(x[0, -1])
                # self.vis_rays(far=opt.t_val)

                # visualize_field(rays_o.cpu().numpy(), vectors={
                #     'x': color_rays_d.cpu().numpy()
                # }, len_limit=-1)
            yield


if __name__ == '__main__':
    scene = MatchScene(100)
    scene.show_heat()

