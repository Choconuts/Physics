import torch
from cluster.third.neus_model import ImplicitNetworkMy
from cluster.camera import Scene, ui, visualize_field


class TVLoss(torch.nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class NeRF:

    def __init__(self):
        self.neus = ImplicitNetworkMy()
        self.neus.cuda()

        from tensorf.tensoRF import TensorVMSplit
        self.tensorf = TensorVMSplit(torch.tensor([[-1.5] * 3, [1.5] * 3]).cuda(), [128] * 3, 'cuda',
                                     shadingMode="MLP_Fea",
                                     density_n_comp=[8, 8, 8],
                                     appearance_n_comp=[24, 24, 24])
        self.tensorf.cuda()
        self.tv = TVLoss()

        # from cluster.third.nerf import NeRF as NeRF0
        # self.nerf0 = NeRF0()
        # self.nerf0.cuda()

        self.optimizer = torch.optim.Adam(params=self.tensorf.get_optparam_groups(), betas=(0.5, 0.99))
        # self.optimizer = torch.optim.Adam(params=self.nerf0.parameters(), lr=5e-4)

    def tv_loss(self):
        loss_tv = self.tensorf.TV_loss_density(self.tv) * 0.1
        return loss_tv

    def __call__(self, rays_o, rays_d):
        t = torch.linspace(1.0, 3.0, 128).cuda()

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
        # return -self.neus(x)[..., :1]
        # return self.nerf0.density(x)
        x = self.tensorf.normalize_coord(x)
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


class NeRFScene(Scene):

    def __init__(self, *args, **kwargs):
        super(NeRFScene, self).__init__(*args, **kwargs)
        self.nerf = NeRF()

    @ui
    def show_neus(self):
        self.vis_rays()
        yield self.vis_field(self.nerf.density, 0.001)

    @ui
    def train(self):
        def field(x):
            dist = (torch.norm(x, dim=-1, keepdim=True) - 0.5) ** 2
            a = torch.exp(-dist * 10)
            return a

        for i in range(10000):
            x = torch.rand(4096, 3).cuda() * 3 - 1.5
            loss = ((self.nerf.density(x) - field(x)) ** 2).mean() + self.nerf.tv_loss()
            self.nerf.step(loss)
            self.vis_field(self.nerf.density)
            print(loss.item())

            yield

    @ui
    def train_nerf(self):
        for i in range(10000):
            rays_o, rays_d, rgb, mask = self.sample_batch(2048)
            color = self.nerf(rays_o.view(-1, 3), rays_d.view(-1, 3))
            rgb[~mask] = 1.0
            loss = (color - rgb.view(-1, 3)).abs().mean()
            self.nerf.step(loss)

            print(loss.item())

            yield self.vis_field(self.nerf.density, 0.1)

    @ui
    def check_rays(self):
        o, d, c, a = self.sample_batch()
        o = o[a]
        d = d[a] * 4
        c = c[a]

        self.vis_field(lambda x: -self.nerf.neus(x)[..., :1], 0.001)
        visualize_field(o.cpu().numpy(), vectors={
            'x': torch.cat([d, c], -1).cpu().numpy()
        }, len_limit=-1)


if __name__ == '__main__':
    scene = NeRFScene(1)
    scene.check_rays()
