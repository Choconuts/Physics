from functional import Simulatable, visualize_field, time_recorded
from geometry import Strand, Square, Strand3D, ParamSquare
import numpy as np
import torch
from kornia import create_meshgrid


def get_rays(starts, direction, z_as_up=True):
    new_z = direction
    new_y = torch.zeros_like(direction)
    if z_as_up:
        new_y[..., 2:3] = 1
    else:
        new_y[..., 1:2] = 1

    def dot(a, b):
        return (a * b).sum(dim=-1, keepdim=True)

    def cross(a, b):
        return torch.cross(a, b, dim=-1)

    new_z = new_z / torch.sqrt(dot(new_z, new_z))

    new_x = cross(new_y, new_z)
    new_x_sqr = dot(new_x, new_x)

    new_y[new_x_sqr[..., 0] <= 1e-4] += 0.001
    new_x = cross(new_y, new_z)
    new_x_sqr = dot(new_x, new_x)

    new_x /= torch.sqrt(new_x_sqr)
    new_y = cross(new_z, new_x)
    new_y = new_y / torch.sqrt(dot(new_y, new_y))

    c2w = torch.stack([new_x, new_y, new_z], -1)

    det = torch.det(c2w)
    rays_o = c2w @ starts[..., None]  # (H, W, 3)
    rays_o = rays_o.view(-1, 3)

    rays_d = new_z.expand(rays_o.shape)  # (H, W, 3)
    rays_d = rays_d.view(-1, 3)

    return rays_o, rays_d


def get_ray_starts(H, W, scale, center=None):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5
    i, j = grid.unbind(-1)
    cent = center if center is not None else [W / 2, H / 2]
    starts = torch.stack([(i - cent[0]) * scale[1] / W, (j - cent[1]) * scale[0] / H, torch.zeros_like(i)], -1)    # (H, W, 3)
    return starts


def get_orthogonal_rays(direction, img_size=(800, 800), scale=(5, 5), distance=5, center=None):
    # size、scale are in H, W form
    if center is None:
        center = [0, 0, 0]
    assert isinstance(direction, torch.Tensor)
    if isinstance(scale, (float, int)):
        scale = [scale, scale]
    if not isinstance(center, torch.Tensor):
        center = torch.tensor(center, dtype=direction.dtype)
    center = center.to(direction.device)
    starts = get_ray_starts(img_size[0], img_size[1], scale)
    rays_o, rays_d = get_rays(starts, direction)
    rays_o = rays_o - rays_d * distance
    rays_o = rays_o + center
    return rays_o, rays_d


class Sampler(Simulatable):

    scale = 1
    center = [0, 0, 0]
    distance = 1
    n_cell = 8
    direction = [-1, -1, 0]

    def __init__(self):
        super().__init__()
        batch_size = 1000
        self.batch_size=  batch_size

    def step(self, dt):
        direction = torch.tensor(self.direction, dtype=torch.float32)
        rays_o, rays_d = get_orthogonal_rays(direction, (self.n_cell, self.n_cell), self.scale, self.distance, self.center)

        visualize_field(rays_o, vectors=rays_d * 50)


if __name__ == '__main__':
    Sampler().run()
