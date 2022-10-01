import torch
import imgui
import numpy as np
from tqdm import trange
from cluster.match import MatchScene, ui, SynDataset, FocusSampler, Inputable, visualize_field
from cluster.alld import DepthGaussian
from utils import rend_util
from cluster.third.model import MyNeRF


def weight_fn(x):
    d = x.abs() - 0.5
    sdf = torch.clamp(d, min=0).norm(dim=-1) + torch.clamp(torch.max(d, dim=-1)[0], max=0.0)
    return torch.exp(-(sdf[..., None]) ** 2 * 5).detach()


def trans_int(probs):
    assert (probs <= 1).all()
    alpha = probs       # torch.exp(-density_delta)
    trans = torch.cat([
        torch.ones_like(probs[..., :1]),
        torch.cumprod(1 - probs[..., :-1] + 1e-4, dim=-1)
    ], dim=-1)
    weights = alpha * trans

    return weights


class Opt(Inputable):

    def __init__(self, z_val=0.0):
        self.t_val = z_val
        self.x_val = z_val
        self.y_val = z_val
        self.z_val = z_val
        self.show = True

    def gui(self, label, *args, **kwargs):
        self.changed, self.show = imgui.checkbox("Show", self.show)
        self.changed, self.t_val = imgui.slider_float('t', self.t_val, 0.2, 5.0)
        self.changed, (self.x_val, self.y_val, self.z_val) = \
            imgui.slider_float3('x', self.x_val, self.y_val, self.z_val, -1.5, 1.5)


opt = Opt()


class GaussianScene(MatchScene):

    def __init__(self, n_image):
        self.data = SynDataset(r"E:\BaiduNetdiskDownload\colmap_result", 458 // n_image, blender=False)
        # self.data = SynDataset(r"G:\Repository\nerf-pytorch\data\nerf_synthetic\lego", 100 // n_image)
        self.focus_sampler = FocusSampler(self.data)
        self.gaussian = DepthGaussian(self.data.n_cameras, *self.data.img_res, init_mu=3.0, init_sigma=1.0)
        self.gaussian.cuda()
        self.optimizer = torch.optim.Adam(params=self.gaussian.parameters(), lr=0.005)

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

    @ui(opt)
    def show_gaussian(self):
        batch_size = 1
        num_sample = 128
        uv = torch.rand(self.data.n_cameras, batch_size, 2).cuda() * 2 - 1

        while True:
            if opt.changed:
                self.gaussian.mu_sigma.data[:, 1, :, :] = opt.t_val * 2

                t = self.gaussian.sample(num_sample + 1, uv)
                t = torch.sort(t[..., 0], dim=-1)[0]
                dist = t[..., 1:] - t[..., :-1]
                mid_t = (t[..., 1:] + t[..., :-1]) / 2

                norm_prob = self.gaussian.probability(mid_t[..., None].detach(), uv) * dist.detach()

                q = weight_fn(mid_t, uv)
                w = trans_int(norm_prob * q)

                rays_o, rays_d, rgb, mask = self.sample_uv(uv)
                x = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * mid_t.view(-1, num_sample, 1)
                visualize_field(x.view(-1, 3), scalars=w.view(-1))
                self.vis_axis()
            yield

    @ui(opt)
    def train_gaussian(self):
        batch_size = 128
        num_sample = 64

        pbar = trange(10001)
        for i in pbar:
            uv = torch.rand(self.data.n_cameras, batch_size, 2).cuda() * 2 - 1
            t = self.gaussian.sample(num_sample + 1, uv)
            t = torch.sort(t[..., 0], dim=-1)[0]
            dist = t[..., 1:] - t[..., :-1]
            mid_t = (t[..., 1:] + t[..., :-1]) / 2

            norm_prob = self.gaussian.probability(mid_t[..., None].detach(), uv) * dist.detach()
            rays_o, rays_d, rgb, mask = self.sample_uv(uv)
            x = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * mid_t.view(-1, num_sample, 1)
            rays_d = rays_d.unsqueeze(-2).expand(x.shape).reshape(-1, 3)
            rgb = rgb.unsqueeze(-2).expand(x.shape).reshape(-1, 3)
            x = x.view(-1, 3)

            q = weight_fn(x)            # self.likelihood(x, rays_d, rgb)
            w = trans_int(norm_prob * q.view(norm_prob.shape))

            loss = ((w.sum(-1) - 1.0) ** 2).mean() + torch.var(t, -1).mean() * 0.1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if i % 10 == 0:
                pbar.set_postfix({"Loss": loss.item(), "Var": torch.var(t, -1).mean().item()})

            w_show = w.view(-1)
            if (w_show > 0.03).any():
                visualize_field(x[w_show > 0.03], scalars=w_show[w_show > 0.03])
            yield


if __name__ == '__main__':
    scene = GaussianScene(50)
    scene.train_gaussian()

