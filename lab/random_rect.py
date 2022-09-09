import torch
import numpy as np
import matplotlib.pyplot as plt


def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

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


def sample_random_rect_idx(H, W, A, B):
    coords = torch.stack(torch.meshgrid(torch.linspace(0, A - 1, A), torch.linspace(0, B - 1, B)), -1)  # (A, B, 2)
    coords = torch.reshape(coords, [-1, 2])  # (A * B, 2)
    HS = torch.randint(0, H - A - 1, [1])
    WS = torch.randint(0, W - B - 1, [1])
    coords[:, 0] += HS
    coords[:, 1] += WS
    return coords.long()


bins = torch.linspace(0, 1, 100)
weights = - (bins - 0.5) ** 2 * 2 + 0.5


# plt.plot(bins, weights)
# plt.show()

sps = sample_pdf(bins.unsqueeze(0), weights[:-1].unsqueeze(0), 18)
sps = sps.view(-1)

plt.scatter(sps, torch.ones_like(sps))
plt.show()

if __name__ == '__main__':
    a = torch.linspace(0, 100, 100).view(10, 10)
    b = sample_random_rect_idx(10, 10, 3, 4)
    print(b)
    print(a[b[:, 0], b[:, 1]])
