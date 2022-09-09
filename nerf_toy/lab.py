import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import cv2
from tqdm import trange, tqdm

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

N_cell = 400
device = 'cuda'
use_similarity = True
use_anchor_sampling = False
max_sigma = 1000


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
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, in_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': in_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class ColorGrid(nn.Module):

    def __init__(self):
        super(ColorGrid, self).__init__()

        # color_img = cv2.imread("color.png")
        #
        # def transform(arr):
        #     return torch.tensor(cv2.resize(arr / 255., (N_cell, N_cell))).permute(2, 0, 1)
        #
        # color = transform(color_img).unsqueeze(0).float()
        self.color = nn.Parameter(torch.randn(1, 3, N_cell, N_cell))
        self.grid = nn.Parameter(torch.randn(1, 3, N_cell, N_cell))
        # g = torch.linspace(0, 1, N_cell)
        # data = torch.stack(torch.meshgrid(g, g), 0).view(1, 2, N_cell, N_cell)
        # data = torch.cat([data, data[:, 0:1] * data[:, 1:2]], 1)
        # self.grid = nn.Parameter(data)

    def forward(self, x):
        N_sample = None
        if len(x.shape) == 3:
            N_sample = x.shape[1]
        x = x * 2 - 1
        chan = self.grid.size(1)
        val = F.grid_sample(self.grid, x.view(1, -1, 1, 2))
        val = val.view(chan, -1).permute(1, 0)

        col = F.grid_sample(self.color, x.view(1, -1, 1, 2))
        col = torch.sigmoid(col).view(3, -1).permute(1, 0)
        ret = torch.cat([col, val], -1)

        if N_sample is not None:
            return ret.view(-1, N_sample, ret.size(-1))
        return ret


class PEGrid(nn.Module):

    def __init__(self):
        super(PEGrid, self).__init__()
        self.embed, embed_dim = get_embedder(10, 2)
        W = 256
        D = 3
        self.skips = [0]
        self.pts_linears = nn.ModuleList(
            [nn.Linear(embed_dim, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + embed_dim, W) for i in range(D-1)])

        self.output_linear = nn.Linear(W, 3 + 3)

        def init_weight(m):
            if type(m) == nn.Linear:
                nn.init.orthogonal_(m.weight, 1)

        self.apply(init_weight)

    def forward(self, x):
        N_sample = None
        if len(x.shape) == 3:
            N_sample = x.shape[1]

        input_pts = self.embed(x.reshape(-1, 2))
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.leaky_relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        outputs = self.output_linear(h)
        if N_sample is not None:
            return outputs.view(-1, N_sample, outputs.size(-1))
        return outputs


class RefGrid:

    def __init__(self, color_img=None, sigma_img=None):
        if color_img is None:
            color_img = cv2.imread("color.png")
        if sigma_img is None:
            sigma_img = cv2.imread("sigma.png")

        def transform(arr):
            return torch.tensor(cv2.resize(arr / 255., (N_cell, N_cell))).permute(2, 0, 1)

        self.color = transform(color_img).unsqueeze(0).float().to(device)
        self.sigma = 1. - transform(sigma_img).unsqueeze(0)[:, 0:1].float().to(device)

    def __call__(self, x):
        """

        RGB is in [0, 1] and do not need sigmoid
        T is in [0, 500]

        :param x:
        :return: RGB T
        """
        N_sample = None
        if len(x.shape) == 3:
            N_sample = x.shape[1]

        x = x * 2 - 1
        col = F.grid_sample(self.color, x.view(1, -1, 1, 2))
        col = col.view(3, -1).permute(1, 0)
        sig = F.grid_sample(self.sigma, x.view(1, -1, 1, 2))
        sig = sig.view(-1, 1) * max_sigma

        ret = torch.cat([col, sig], -1)

        if N_sample is not None:
            return ret.view(-1, N_sample, 4)

        return ret.view(-1,  4)


def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


def log_tensor(tensor):
    print("[Tensor]", torch.max(tensor).item(), torch.min(tensor).item(), torch.mean(tensor).item())


def vis_arr(grid, save_tag=None, show=True, scale=0.1, sig_as_rgb=False):
    g = torch.linspace(0, 1, N_cell)[:-1]
    i, j = torch.meshgrid(g, g)
    i = i.flatten()
    j = j.flatten()
    x = torch.stack([i, j], -1).to(device)
    res = grid(x)
    res[..., -1:] /= max_sigma
    res = torch.clamp(res, 0, 1)
    res = res.detach().cpu().numpy()
    col = res[:, :3]
    sig = res[:, 3:4]
    if sig_as_rgb:
        sig = res[:, 3:6]
    if show:
        plt.scatter(i, j, c=col, s=scale)
    if save_tag is None:
        if show:
            plt.show()
    else:
        plt.savefig(f"logs/color-{save_tag}.png")
    plt.scatter(i, j, c=sig, s=scale)
    if save_tag is None:
        if show:
            plt.show()
    else:
        plt.savefig(f"logs/sigma-{save_tag}.png")


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0., white_bkgd=False, pytest=False, next_seg=None, **kwargs):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    # if 'double' in kwargs and kwargs['double']:
    #     N_sample = raw.shape[1] // 2
    #     ref_raw = raw[:, N_sample:, :]
    #     raw = raw[:, :N_sample, :]

    dists = z_vals[...,1:] - z_vals[...,:-1]
    # my test
    if next_seg is None:
        # next_seg = torch.Tensor([1e10]).cuda().expand(dists[..., :1].shape)
        next_seg = torch.Tensor([0]).cuda().expand(dists[..., :1].shape)
    dists = torch.cat([dists, next_seg], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = raw[...,:3]  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    if "use_sigma_as_weights" in kwargs and kwargs['use_sigma_as_weights']:
        weights = raw[...,3]
    else:
        if "use_sigma_as_alpha" in kwargs and kwargs['use_sigma_as_alpha']:
            alpha = raw[...,3]

            """ 开根号试试 """
            # alpha = torch.pow(alpha, dists * 200)
        else:
            """ alpha 代表的是不透明度，可以相乘的是可见性（透明度） """
            alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

        if torch.min(alpha) < -1:
            log_tensor(alpha)
            log_tensor(raw[...,3])
            log_tensor(-F.relu(raw[...,3])*dists)

        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        Ts = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1.-alpha + 1e-10], -1), -1)
        weights = alpha * Ts[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_fine,
                network_query_fn,
                N_samples,
                N_importance=0,
                double_pts=False,
                retraw=False,
                lindisp=False,
                perturb=1.,
                white_bkgd=False,
                raw_noise_std=0.,
                pytest=False, **kwargs):
    """Volumetric rendering.
     Args:
       ray_batch: array of shape [batch_size, ...]. All information necessary
         for sampling along a ray, including: ray origin, ray direction, min
         dist, max dist, and unit-magnitude viewing direction.
       network_fn: function. Model for predicting RGB and density at each point
         in space.
       network_query_fn: function used for passing queries to network_fn.
       N_samples: int. Number of different times to sample along each ray.
       retraw: bool. If True, include model's raw, unprocessed predictions.
       lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
       perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
         random points in time.
       N_importance: int. Number of additional times to sample along each ray.
         These samples are only passed to network_fine.
       network_fine: "fine" network with same spec as network_fn.
       white_bkgd: bool. If True, assume a white background.
       raw_noise_std: ...
       verbose: bool. If True, print more debugging info.
     Returns:
       rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
       disp_map: [num_rays]. Disparity map. 1 / depth.
       acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
       raw: [num_rays, num_samples, 4]. Raw predictions from model.
       rgb0: See rgb_map. Output for coarse model.
       disp0: See disp_map. Output for coarse model.
       acc0: See acc_map. Output for coarse model.
       z_std: [num_rays]. Standard deviation of distances along ray for each
         sample.
     """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples, device=device)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=device)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand, device=device)

        z_vals = lower + (upper - lower) * t_rand

    # pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]          # [N_rays, N_samples, 3]
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]     # [N_rays, N_samples, 3]

    if double_pts:
        ref_pts = rays_o[..., None, :] + rays_d[..., None, :] * (z_vals[..., :, None] - 1 / 50)  # [N_rays, N_samples, 3]
        pair_pts = torch.cat([pts, ref_pts], 1)
    else:
        pair_pts = pts

#     raw = run_network(pts)
#     raw = network_query_fn(pts, viewdirs, network_fn)
    raw = network_query_fn(pair_pts, viewdirs, network_fn)

    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, **kwargs)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                            None]  # [N_rays, N_samples + N_importance, 3]

        """ anchored sampling """
        if use_anchor_sampling:
            anchor_idx = torch.floor((z_vals - z_vals[..., :1]) / (z_vals[..., -1:] - z_vals[..., :1]) * (N_samples - 1))
            anchor_idx = torch.clip(anchor_idx - 1, 0, N_samples).long()
            anchor_alpha = raw[..., 3]
            if (anchor_idx > 63).any() or (anchor_idx < 0).any():
                print(1)
            kwargs['anchor_idx'] = anchor_idx
            kwargs['anchor_alpha'] = anchor_alpha
            kwargs['anchor_pos'] = pair_pts
            kwargs['use_sigma_as_weights'] = True

        run_fn = network_fn if network_fine is None else network_fine
        #         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn, **kwargs)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                     pytest=pytest, **kwargs)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    return ret


def ray3d_to_2d(rays):
    rays[:, 2] = 0
    rays[:, 5] = 0
    rays[:, -1:] = 0
    rays[:, 8:] = rays[:, 8:] / torch.norm(rays[:, 8:], dim=-1)
    return rays


def sample_rays(batch_size):
    x = torch.rand(batch_size, 3, device=device)
    y = torch.rand(batch_size, 3, device=device)
    x_y = torch.norm(x, dim=-1, keepdim=True) - torch.norm(y, dim=-1, keepdim=True)
    x = torch.where(x_y == 0, x + 1e-4, x)
    starts = torch.where(x_y > 0, x, y)
    ends = torch.where(x_y <= 0, x, y)
    far = torch.norm(starts - ends, dim=-1, keepdim=True)

    far = torch.ones_like(far) * 0.8

    near = torch.ones_like(far) * 0.1
    dirs = (starts - ends) / far
    return torch.cat([starts, dirs, near, far, dirs], -1)


img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(device))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


def train():
    N_iters = 200000 + 1

    gt_model = RefGrid()
    # model = ColorGrid()
    model = PEGrid()
    # fine_model = PEGrid()
    model.cuda()
    # fine_model.cuda()

    import itertools
    optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-4, betas=(0.9, 0.999))
    # optimizer = torch.optim.Adam(params=itertools.chain(model.parameters(), fine_model.parameters()), lr=1e-3, betas=(0.9, 0.999))

    os.makedirs('logs', exist_ok=True)

    closure_loss = [0]

    def network_query_fn(pts, viewdirs, network_fn, anchor_idx=None, anchor_alpha=None, anchor_pos=None, **kwargs):
        val = network_fn(pts[..., :2])
        a_val = val[:, 1:]
        b_val = val[:, :-1]
        col = (a_val[..., :3] + b_val[..., :3]) / 2
        # padding
        col = torch.cat([col, col[:, -1:]], 1)

        def cos_sim(u, v):
            return (u * v).sum(-1) / (torch.norm(u, dim=-1) + 1e-6) / (torch.norm(v, dim=-1) + 1e-6)

        if use_similarity and use_anchor_sampling and anchor_idx is not None:
            anchor_theta = network_fn(anchor_pos[..., :2])[..., 3:]      # N, 64, 3
            theta = val[..., 3:]                                # N, 128, 3
            # 直接输出 weight 吧，那边改成在raw2out里处理

            anchor_T = torch.cumprod(torch.cat([torch.ones((anchor_alpha.shape[0], 1), device=device), 1. - anchor_alpha + 1e-10], -1), -1)
            # anchor_T: [N, 65], 索引61为光线抵达60号线段终点（61号线段起点）时的结果

            alpha = cos_sim(theta, torch.gather(anchor_theta, 1, anchor_idx.unsqueeze(-1).expand(-1, -1, theta.size(-1))))

            alpha = F.relu(1. - F.relu(alpha))
            # alpha: [N, 128], 索引121为121号线段起点与其锚点之间的不透明度

            T = torch.gather(anchor_T, 1, anchor_idx) * (1. - alpha + 1e-10)     # N, 128, 索引121为光线抵达121号线段起点时的结果
            last_alpha = 1 - T[:, 1:] / (torch.where(T[:, :-1] > T[:, 1:], T[:, :-1], T[:, 1:]) + 1e-10)
            weights = T[:, :-1] * last_alpha

            # padding
            sig = torch.cat([weights, weights[:, -1:]], 1)
            return torch.cat([col, sig.unsqueeze(-1)], -1)

        if use_similarity:
            sim = cos_sim(a_val[..., 3:], b_val[..., 3:])
            sig = 1.-F.relu(sim).unsqueeze(-1)
            closure_loss[0] = ((1 - sim) ** 2).mean()
        else:
            # sig = a_val       # 这样居然是不行的！
            sig = (a_val[..., 3:4] + b_val[..., 3:4]) / 2
            closure_loss[0] = torch.log(1 + 2 * sig ** 2).mean()

        # padding
        sig = torch.cat([sig, sig[:, -1:]], 1)

        return torch.cat([col, sig], -1)

    def network_double_query_fn(pts, viewdirs, network_fn, **kwargs):
        N_samples = pts.shape[1] // 2
        pos = pts[:, :N_samples, :2]
        ref = pts[:, N_samples:, :2]
        pos_val = network_fn(pos[..., :2])
        ref_val = network_fn(ref[..., :2])
        col = (pos_val[..., :3] + ref_val[..., :3]) / 2
        assert use_similarity

        def cos_sim(u, v):
            return (u * v).sum(-1) / (torch.norm(u, dim=-1) + 1e-6) / (torch.norm(v, dim=-1) + 1e-6)

        sim = cos_sim(ref_val[..., 3:], pos_val[..., 3:])
        sig = 1.-F.relu(sim).unsqueeze(-1)
        closure_loss[0] = ((1 - sim) ** 2).mean()

        return torch.cat([col, sig], -1)

    def network_query_fn_0(pts, viewdirs, network_fn, **kwargs):
        return network_fn(pts[..., :2])

    for i in trange(0, N_iters):
        rays = sample_rays(1024)

        # random_num = np.random.randint(32, 33, 1).item()
        random_num = 32
        #####  Core optimization loop  #####
        net_ret = render_rays(rays, model, None, network_query_fn, random_num, 64, use_sigma_as_alpha=use_similarity, double_pts=False, perturb=0)
        gt_rgb = render_rays(rays, gt_model, None, network_query_fn_0, 128, perturb=0)['rgb_map']

        rgb = net_ret['rgb_map']

        optimizer.zero_grad()
        img_loss = img2mse(rgb, gt_rgb)
        loss = img_loss + 0.01 * closure_loss[0]

        if 'rgb0' in net_ret:
            img_loss0 = img2mse(net_ret['rgb0'], gt_rgb)
            loss = loss + img_loss0

        psnr = mse2psnr(img_loss)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        if i % 5000 == 0 and i > 0:
            vis_arr(model, i, sig_as_rgb=use_similarity)
        if i % 10000 == 0 and i > 0:
            path = os.path.join("logs", f'{i:06d}-PSNR{psnr.item():04f}.tar')
            torch.save({
                'global_step': i,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)


def show_diff(grid):
    g = torch.linspace(0, 1, 400)
    i, j = torch.meshgrid(g, g)
    x = torch.stack([i, j], -1).to(device)
    feat = grid(x)[:, :, 3:]

    def cos_sim(u, v):
        return (u * v).sum(-1) / (torch.norm(u, dim=-1) + 1e-6) / (torch.norm(v, dim=-1) + 1e-6)

    def sample_offset(*offset):
        offset = torch.tensor(offset, device=device) * 0.006
        nb_feat = grid(x + offset)[:, :, 3:]
        sim = cos_sim(feat, nb_feat)
        sim = F.relu(-sim)
        return sim

    nb_sim = []
    for u in [-1, 0, 1]:
        for v in [-1, 0, 1]:
                if u == v == 0:
                    continue
                nb_sim.append(sample_offset(u, v))

    tot_sim = torch.sum(torch.stack(nb_sim, 0), 0)
    tot_sim = tot_sim.detach().cpu().numpy()
    tot_sim = tot_sim > tot_sim.mean() * 6

    ii, jj = i.flatten(), j.flatten()
    plt.scatter(ii, jj, c=tot_sim.flatten(), s=2)
    plt.show()


def eval():

    ckpts = [os.path.join('logs', f) for f in sorted(os.listdir(os.path.join('logs'))) if 'tar' in f]

    print('Found ckpts', ckpts)
    assert len(ckpts) > 0
    ckpt_path = ckpts[-1]
    print('Reloading from', ckpt_path)
    ckpt = torch.load(ckpt_path)

    start = ckpt['global_step']
    # optimizer.load_state_dict(ckpt['optimizer'])

    model = ColorGrid()
    # Load model
    model.load_state_dict(ckpt['model'])
    model.cuda()

    show_diff(model)


def tst_grid():
    cg = ColorGrid()
    y = cg(torch.rand(100, 2))
    print(y.shape)
    vis_arr(cg)
    rg = RefGrid()
    y = rg(torch.rand(100, 2))
    print(y.shape)
    vis_arr(rg)


def tst_ray_render():
    rg = RefGrid()
    cg = ColorGrid()

    cg.to(device)

    ray_batch = torch.tensor([
        [0.1, 0.1, 0.1,
        0.6, 0.6, 0.6,
        0.2,
        1.2,
        0.6, 0.6, 0.6],
    ], device=device)

    ray_batch = ray3d_to_2d(ray_batch)
    print(ray_batch)

    # ray_batch = sample_rays(1024)
    # print(ray_batch.shape)

    def network_query_fn(pts, viewdirs, network_fn):

        return network_fn(pts[..., :2])

    res = render_rays(ray_batch, rg, None, network_query_fn, 64)
    print(res)


if __name__ == '__main__':
    train()
