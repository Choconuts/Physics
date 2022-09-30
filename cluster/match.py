import torch
import imgui
import numpy as np
from cluster.camera import Scene, ui, SynDataset, FocusSampler, Inputable, visualize_field


def ln_var(x, sharpness=6):
    if isinstance(x, torch.Tensor):
        return -0.2 * torch.log(1 + torch.exp(sharpness * x - 3))
    return -np.log(1 + np.exp(sharpness * x - 3))


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


class MatchScene(Scene):

    def __init__(self, n_image):
        self.data = SynDataset(r"E:\BaiduNetdiskDownload\colmap_result", 458 // n_image, blender=False)
        # self.data = SynDataset(r"G:\Repository\nerf-pytorch\data\nerf_synthetic\lego", 100 // n_image)
        self.focus_sampler = FocusSampler(self.data)

    def sample_gt_mask(self, uv, selector=None):
        return torch.ones_like(uv[..., 0]).bool()

    def sample_neighbor(self, x, dirs, best_n=20):
        sample, gt = self.focus_sampler.scatter_sample(x)
        all_mask = sample['object_mask']
        all_rgb = gt['rgb']
        all_dirs = sample['view_dir']

        score = (all_dirs - dirs).norm(dim=-1)
        _, idx = torch.sort(score, 0)
        mask = all_mask.gather(0, idx[:best_n])
        rgb = all_rgb.gather(0, idx[:best_n, :, None].expand(-1, -1, 3))
        return rgb, mask

    @ui(opt)
    def show_match(self):
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
                x = torch.rand(10000, 3).cuda().view(-1, 3) * 2 - 1
                x[:, 2] += 0.5
                x[:, 0] = opt.x_val
                dirs = -x / (x.norm(dim=-1, keepdim=True) + 1e-5)

                rgb, mask = self.sample_neighbor(x, dirs, best_n)
                mask = mask[..., None]
                # poses = [self.data.pose_all[i] for i in idx[:best_n, 0]]

                mean_rgb = torch.sum(rgb * mask, dim=0) / mask.sum(0)
                mean_var = torch.sum((rgb - mean_rgb) ** 2 * mask, dim=0) / mask.sum(0)
                # mean_var = (ln_var((rgb - mean_rgb).norm(dim=-1))).mean(0)
                # prob = torch.exp(mean_var * 2)

                visualize_field(x, scalars=mean_var * 10)
                self.vis_axis()
                # self.vis_rays(lambda x: x[idx[2:3, 0],], 20)
            yield

    @ui(opt)
    def show_hit(self):

        while True:
            if opt.changed:
                best_n = 20
                rays_o, rays_d, rgb0, mask0 = self.sample_batch(2048)
                t = torch.linspace(0.0, 3.0, 128).cuda()[:, None]
                x = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * t
                x = x[97]
                rays_d = rays_d[97]
                rays_o = rays_o[97]

                rgb, mask = self.sample_neighbor(x, rays_d, best_n)
                mask = mask[..., None]
                # poses = [self.data.pose_all[i] for i in idx[:best_n, 0]]

                mean_rgb = torch.sum(rgb * mask, dim=0) / mask.sum(0)
                mean_var = torch.sum((rgb - mean_rgb) ** 2 * mask, dim=0) / mask.sum(0)
                # mean_var = (ln_var((rgb - mean_rgb).norm(dim=-1))).mean(0)
                # prob = torch.exp(mean_var * 2)

                visualize_field(x, scalars=mean_var * 10)
                self.vis_axis()
                # self.vis_rays(lambda x: x[idx[2:3, 0],], 20)
            yield

scene = MatchScene(100)
scene.show_hit()

if __name__ == '__main__':
    pass
