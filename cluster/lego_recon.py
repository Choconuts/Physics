import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import trange
from cluster.syn_dataset import *
from cluster.focus_sampler import *
from cluster.alld import DepthGaussian, vis_image, weight_fn, trans_int
from cluster.posterior import posterior_prob

torch.random.manual_seed(0)

data = SynDataset(r"G:\Repository\nerf-pytorch\data\nerf_synthetic\lego", 1)
focus_sampler = FocusSampler(data)

depth = DepthGaussian(data.n_cameras, 256, 256, 3.0, 1.0)
depth.cuda()


def uv_to_tex(uv):
    return torch.stack([(uv[..., 0] + 1) / 2 * data.img_res[0], (uv[..., 1] + 1) / 2 * data.img_res[1]], dim=-1)


def sample_gt_mask(uv):
    masks = torch.cat(data.object_masks, 0).view(-1, *data.img_res, 1).permute(0, 3, 1, 2)
    mask = F.grid_sample(masks.cuda().float(), uv.view(depth.n, 1, -1, 2)).view(uv.shape[:-1])
    return mask > 0.5


def weight_fn(t, uv, mask):
    pose = torch.stack(data.pose_all).cuda()
    intrinsics = torch.stack(data.intrinsics_all).cuda()
    uv_tex = uv_to_tex(uv)
    rays_d, rays_o = rend_util.get_camera_params(uv_tex, pose, intrinsics)
    x = rays_d[:, :, None, :] * t[:, :, :, None] + rays_o[:, None, None, :]
    sample, ground_truth = focus_sampler.scatter_sample(x.view(-1, 3))
    prob = posterior_prob(ground_truth['rgb'], sample['object_mask'])
    g = torch.autograd.grad(prob.sum(), t, retain_graph=True)[0]
    if g.isnan().any():
        print(g.isnan().nonzero())
    return prob.view(*x.shape[:-1])


# def weight_fn(t, uv, mask):
#     masks = torch.cat(data.object_masks, 0).view(-1, *data.img_res, 1).permute(0, 3, 1, 2)
#     mask = F.grid_sample(masks.cuda().float(), uv.view(depth.n, 1, -1, 2)).view(t.shape[:-1])
#     d = mask * 2 + 2.0
#     return torch.exp(-(t - d[..., None]) ** 2 * 5)


optimizer = torch.optim.Adam(params=depth.parameters(), lr=0.005)

batch_size = 1024
sample_num = 64
pbar = trange(10001)
for i in pbar:
    uv = torch.rand(depth.n, batch_size, 2).cuda() * 2 - 1
    mask = sample_gt_mask(uv)

    t = F.softplus(depth.sample(sample_num + 1, uv))
    t = torch.sort(t[..., 0], dim=-1)[0]

    dist = t[..., 1:] - t[..., :-1]
    mid_t = (t[..., 1:] + t[..., :-1]) / 2

    norm_prob = depth.probability(mid_t[..., None].detach(), uv) * dist.detach()

    q = weight_fn(mid_t, uv, mask)
    w = trans_int(norm_prob * q)

    loss = ((w.sum(-1)[mask] - 1.0) ** 2).mean()            # TODO: remove  is better, why?

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        pbar.set_postfix({"Loss": loss.item()})
    if i % 200 == 0:
        plt.xlim(0, 6.0)
        # plt.plot(t.cpu().detach().numpy(), w.cpu().detach().numpy())
        # plt.show()
        print(w.max().item(), w.sum().item())
        # plot_fn(g)
        vis_image(depth, i)


if __name__ == '__main__':
    pass
