#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : spherical_sampling.py
@Author: Chen Yanzhen
@Date  : 2022/3/22 16:08
@Desc  : 
"""

import torch
from torch import nn
from torch.autograd.functional import jacobian
from functional import Simulatable, visualize_field, time_recorded
from geometry import Strand, Square, Strand3D, ParamSquare
import numpy as np

device = torch.device("cpu")


def cylinder_to_world(dirs, thetas, rs, cursors):
    rot_mat = torch.tensor([
        [0, -1, 0],
        [1, 0, 0.],
        [0, 0, 1.]
    ], device=device)

    rot_dirs = rot_mat @ dirs.view(-1, 3, 1)
    u = torch.cross(dirs, rot_dirs.view(-1, 3))
    u = u / torch.norm(u, dim=-1, keepdim=True)
    v = torch.cross(dirs, u)
    v = v / torch.norm(v, dim=-1, keepdim=True)

    w = torch.rand(self.n_samples, 1, device=device) * torch.pi - torch.pi / 2
    w = torch.tan(w) / compact

    x, y = circle_uniform(self.n_samples)
    x *= radius
    y *= radius
    pos = u * x.view(-1, 1) + v * y.view(-1, 1) + w.view(-1, 1) * dirs


class Sampler(Simulatable):

    n_samples = 512
    interval = 10.0
    mode = 9
    seed = 1

    near = 0.1
    far = 1.0
    grid = False
    reduce = 1
    direction = False
    start = True
    scale = 0.01
    pose = -1

    radius = 0.5
    compact = 100

    def __init__(self):
        super().__init__()
        batch_size = 1000

        def sample_rays():
            pnts = torch.rand(batch_size, 3, device=device) * 2 - 1
            u = torch.rand(batch_size, device=device) * 2 - 1
            t = torch.rand(batch_size, device=device) * torch.pi * 2
            x = (1 - u ** 2) ** 0.5 * torch.cos(t)
            y = (1 - u ** 2) ** 0.5 * torch.sin(t)
            z = u
            dirs = torch.stack([x, y, z], -1)
            fars = torch.ones_like(pnts[..., :1]) * 2
            rays = torch.cat([pnts, dirs, fars * 0, fars, dirs], -1)
            return rays

        self.rays = sample_rays()

        self.rays = torch.tensor([[0.3, 1.5, 0.9, 1, 1, 1, 0, 1, 0, -0.707, -0.707]])

    def make_rays(self):
        def sample_rays(batch_size, near, far, scene_max=None, scene_min=None, grid_wise=False):
            if grid_wise:
                n_bs = int(batch_size ** 0.333) + 1
                arr = torch.linspace(0, 1, n_bs, device=device)
                x, y, z = torch.meshgrid(arr, arr, arr, indexing='ij')
                pnts = torch.stack([x, y, z], -1).view(-1, 3)
                pnts = pnts[:batch_size]
            else:
                pnts = torch.rand(batch_size, 3, device=device)
                if scene_max is not None and scene_min is not None:
                    pnts = pnts * (scene_max - scene_min) + scene_min

            u = torch.rand(pnts.shape[0], device=device) * 2 - 1
            t = torch.rand(pnts.shape[0], device=device) * torch.pi * 2
            x = (1 - u ** 2) ** 0.5 * torch.cos(t)
            y = (1 - u ** 2) ** 0.5 * torch.sin(t)
            z = u

            dirs = torch.stack([x, y, z], -1)
            fars = torch.ones_like(pnts[..., :1]) * far
            nears = torch.ones_like(pnts[..., :1]) * near
            rays = torch.cat([pnts, dirs, nears, fars, dirs], -1)
            return rays

        def sample_camera_rays(H, W, K, render_pose, near, far, reduce=1):
            c2w = render_pose[:3, :4]
            reduce = max(reduce, 1)

            def my_get_rays(H, W, K, c2w):
                i, j = torch.meshgrid(torch.linspace(0, W - 1, W // reduce),
                                      torch.linspace(0, H - 1, H // reduce))  # pytorch's meshgrid has indexing='ij'
                i = i.t()
                j = j.t()
                dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
                noise = torch.rand_like(dirs) * 0.001
                dirs = dirs + noise
                # Rotate ray directions from camera frame to the world frame
                rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                                   -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
                # Translate camera frame's origin to the world frame. It is the origin of all rays.
                rays_o = c2w[:3, -1].expand(rays_d.shape)
                return rays_o, rays_d

            rays_o, rays_d = my_get_rays(H, W, K, c2w)
            viewdirs = rays_d
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
            rays_o = torch.reshape(rays_o, [-1, 3]).float()
            rays_d = torch.reshape(rays_d, [-1, 3]).float()

            nears, fars = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
            rays = torch.cat([rays_o, viewdirs, nears, fars, viewdirs], -1)
            return rays

        def noised_rays(rays, scale, start=True, direction=False):
            pnts = rays[..., :3]
            noise = torch.rand_like(pnts) * scale
            if start:
                rays[..., :3] = pnts + noise
            if direction:
                rays[..., -3:] = rays[..., -3:] + noise
            return rays

        poses = \
        [[[1.0000e+00, 6.1257e-17, -1.0614e-16, -4.2457e-16],
          [-1.2251e-16, 5.0000e-01, -8.6621e-01, -3.4648e+00],
          [0.0000e+00, 8.6621e-01, 5.0000e-01, 2.0000e+00],
          [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]],

         [[9.8779e-01, 7.8247e-02, -1.3550e-01, -5.4199e-01],
          [-1.5649e-01, 4.9390e-01, -8.5547e-01, -3.4219e+00],
          [0.0000e+00, 8.6621e-01, 5.0000e-01, 2.0000e+00],
          [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]],

         [[9.5117e-01, 1.5454e-01, -2.6782e-01, -1.0713e+00],
          [-3.0908e-01, 4.7559e-01, -8.2373e-01, -3.2949e+00],
          [0.0000e+00, 8.6621e-01, 5.0000e-01, 2.0000e+00],
          [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]]]
        poses = torch.tensor(poses, device=device)

        K = torch.tensor([
            [555.55552, 0, 200],
            [0, 555.55552, 200],
            [0, 0, 1]
        ], device=device)

        if self.pose < 0:
            return noised_rays(sample_rays(self.n_samples, self.near, self.far, grid_wise=self.grid), self.scale, self.start, self.direction)
        pose = poses[self.pose]
        return noised_rays(sample_camera_rays(400, 400, K, pose, self.near, self.far, self.reduce), self.scale, self.start, self.direction)

    def cylinder_random(self, n_samples, radius, compact):

        def circle_uniform(n):
            r = torch.rand(n, device=device) ** 0.5
            t = torch.rand(n, device=device) * torch.pi * 2
            x = r * torch.cos(t)
            y = r * torch.sin(t)
            return x, y

        def spherical_uniform(n):
            u = torch.rand(n, device=device) * 2 - 1
            t = torch.rand(n, device=device) * torch.pi * 2
            x = (1 - u ** 2) ** 0.5 * torch.cos(t)
            y = (1 - u ** 2) ** 0.5 * torch.sin(t)
            z = u
            return x, y, z

        dirs = torch.stack(spherical_uniform(n_samples), -1)

        rot_mat = torch.tensor([
            [0, -1, 0],
            [1, 0, 0.],
            [0, 0, 1.]
        ], device=device)

        rot_dirs = rot_mat @ dirs.view(-1, 3, 1)
        u = torch.cross(dirs, rot_dirs.view(-1, 3))
        u = u / torch.norm(u, dim=-1, keepdim=True)
        v = torch.cross(dirs, u)
        v = v / torch.norm(v, dim=-1, keepdim=True)

        w = torch.rand(self.n_samples, 1, device=device) * torch.pi - torch.pi / 2
        w = torch.tan(w) / compact

        x, y = circle_uniform(self.n_samples)
        x *= radius
        y *= radius
        pos = u * x.view(-1, 1) + v * y.view(-1, 1) + w.view(-1, 1) * dirs

        return pos, dirs, u, v

    def offset_dirs(self, dirs, cam_up, cam_look, H, W, Fx, Fy):
        assert len(dirs.shape) == 2

        cam_right = torch.cross(cam_look, cam_up)
        f = dirs.view(-1, 1, 3) @ cam_look.view(3, 1)
        f = f.view(-1, 1)

        dx = f / Fx
        dy = f / Fy

        u = cam_right * dx
        v = cam_up * dy

        noise = torch.rand(2, dirs.size(0), 1, device=device)

        dirs = dirs + u * noise[0] + v * noise[1]
        return dirs / torch.norm(dirs)

    def step(self, dt):
        torch.manual_seed(self.seed)
        if self.mode == 1:
            pos = torch.rand(self.n_samples, 3)
        if self.mode == 2:
            r = torch.rand(self.n_samples) ** 0.5
            t = torch.rand(self.n_samples) * torch.pi * 2
            x = r * torch.cos(t)
            y = r * torch.sin(t)
            z = torch.sqrt(1 - x ** 2 - y ** 2)

            pos = torch.stack([x, y, z], -1)
            pos = (pos + 1) / 2
        if self.mode == 3:

            def spherical_uniform(n_samples):
                u = torch.rand(n_samples) * 2 - 1
                t = torch.rand(n_samples) * torch.pi * 2
                x = (1 - u ** 2) ** 0.5 * torch.cos(t)
                y = (1 - u ** 2) ** 0.5 * torch.sin(t)
                z = u
                return torch.stack([x, y, z], -1)

            pos = (spherical_uniform(self.n_samples) + 1) / 2

        if self.mode == 4:
            device = 'cpu'
            ray = self.rays
            def projection(ray):
                def interface(point, normal):
                    r = torch.tensor([point], device=device)
                    n = torch.tensor([normal], device=device)
                    d = ray[..., -3:]
                    print(torch.norm(d, dim=-1))
                    u = d.view(-1, 1, 3) @ n.view(-1, 3, 1)
                    idx = u.nonzero()
                    print(u)
                    p = ray[..., :3]
                    w = (r - p).view(-1, 1, 3) @ n.view(-1, 3, 1)
                    print(w)
                    x = p + w[..., -1] / u[..., -1] * d
                    return x

            p_rays = interface([0, 1, 0.], [0, 1, 0.])
            pos = torch.cat([self.rays[:, :3], p_rays])

            visualize_field(pos)

        if self.mode == 5:
            arr = torch.linspace(0, 1, 10)
            def biased(x): return x + (torch.rand_like(x) * 2 - 1) / self.interval

            x, y, z = torch.meshgrid(arr, arr, arr, indexing='ij')
            pos = torch.stack([x, y, z], -1).view(-1, 3)
            pos = biased(pos)

        if self.mode == 6:
            t_vals = torch.linspace(0., 1., steps=10 + 1)
            z_vals = 1./(1./0.1 * (1.-t_vals) + 1./1. * (t_vals))

            pos = torch.stack([torch.ones_like(z_vals), z_vals, torch.ones_like(z_vals)], -1)

        if self.mode == 7:

            rays = self.make_rays()
            starts = rays[:, :3] + rays[:, 8:11] * rays[:, 6:7]
            ends = rays[:, :3] + rays[:, 8:11] * rays[:, 7:8]
            pos = torch.cat([starts, ends], 0)
            val = torch.cat([torch.ones_like(starts), torch.zeros_like(ends)], 0)[:, 0]
            idx = torch.linspace(0, ends.size(0) - 1, ends.size(0), dtype=torch.long)

            visualize_field(pos, scalars=val, graphs={"Ray": (idx, idx + ends.size(0))})

        if self.mode == 8:
            # rays = self.make_rays()
            #
            # pnts = rays[:, :3]
            # dirs = rays[:, -3:]
            #
            # u0 = -pnts
            # du = torch.tensor([0, 0., -1.])
            # u0[torch.prod(u0 == 0, -1).nonzero()] = du
            #
            # u = torch.cross(dirs, u0)
            # u = u / torch.norm(u, dim=-1, keepdim=True)
            # v = torch.cross(dirs, u)
            # v = v / torch.norm(v, dim=-1, keepdim=True)
            #
            # pos = torch.cat([pnts, pnts, pnts], 0)
            # vec = torch.cat([dirs * 3, u, v], 0) * 5
            # visualize_field(pos, vectors=vec)

            pos, dirs, u, v = self.cylinder_random(self.n_samples, self.radius, self.compact)

            def cylinder_coord(pos, dirs, compact):
                rot_mat = torch.tensor([
                    [0, -1, 0],
                    [1, 0, 0.],
                    [0, 0, 1.]
                ], device=device)

                rot_dirs = rot_mat @ dirs.view(-1, 3, 1)
                u = torch.cross(dirs, rot_dirs.view(-1, 3))
                u = u / torch.norm(u, dim=-1, keepdim=True)
                v = torch.cross(dirs, u)
                v = v / torch.norm(v, dim=-1, keepdim=True)



            pos = torch.cat([pos, pos, pos], 0)
            vec = torch.cat([dirs * 3, u, v], 0) * 5
            visualize_field(pos, vectors=vec)

        if self.mode == 9:
            pnts = torch.tensor([0.5, 0.5, 1]).view(-1, 3).repeat(10, 1)
            dirs = torch.tensor([0, 0, -1]).view(-1, 3).repeat(10, 1)
            cam_up = torch.tensor([0, 1, 0])
            cam_look = torch.tensor([0, 1, -1])
            dirs = self.offset_dirs(dirs, cam_up, cam_look, 100, 100, 50, 50)

            visualize_field(pnts, vectors=dirs * 500)

        # visualize_field(pos)


if __name__ == '__main__':

    Sampler().run()

