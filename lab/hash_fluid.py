#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : hash_fluid.py
@Author: Chen Yanzhen
@Date  : 2022/3/16 14:24
@Desc  : 
"""


import argparse
import commentjson as json
import numpy as np
import os
import sys
import torch
import time
import cv2
import tinycudann as tcnn
from functional import Simulatable, visualize_field, time_recorded
import itertools
from torch.autograd.functional import jacobian
import torch.nn.functional as F


device = torch.device("cuda")

encoding_config = {
    "otype": "HashGrid",
    "n_levels": 16,
    "n_features_per_level": 2,
    "log2_hashmap_size": 15,
    "base_resolution": 16,
    "per_level_scale": 1.5
}

network_config = {
    "otype": "FullyFusedMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 64,
    "n_hidden_layers": 2
}

D_TYPE = torch.float32


class MPMSolver3D:
    E, nu = 0.1e4, 0.2
    mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

    def __init__(self, pos, vel, mat, dt, n_grids):
        def grid_iter(*shape):
            return itertools.product(*[range(dim) for dim in shape])

        # magic numbers
        self.n_particles = pos.shape[0]
        self.n_grids = n_grids
        self.dt = dt
        self.material = mat
        self.p_vol, self.p_rho = (1 / self.n_grids * 0.5) ** 3, 1
        self.p_mass = self.p_vol * self.p_rho
        self.device = device

        # tensors
        self.x = pos.to(device)     # N, 3
        self.v = vel.to(device)    # N, 3
        self.C = torch.zeros(self.mat_shape(), dtype=D_TYPE).to(device)  # N, 3, 3
        self.F = torch.stack([self.eye() for _ in range(self.n_particles)])     # N, 3, 3
        self.Jp = torch.ones(self.n_particles, dtype=D_TYPE).to(device)       # N,

        self.grid_v = torch.zeros((n_grids, n_grids, n_grids, 3), dtype=D_TYPE).to(device)     # G, G, 3 *grid node momentum/velocity
        self.grid_m = torch.zeros((n_grids, n_grids, n_grids), dtype=D_TYPE).to(device)        # G, G *grid node mass
        self.boundary = torch.zeros((n_grids, n_grids, n_grids, 3), dtype=D_TYPE).to(device)   # G, G, 3 *grid boundary vectors
        boundary_width = 4
        for i, j, k in grid_iter(n_grids, n_grids, n_grids):
            def sign(x): return -1 if x < boundary_width else (1 if x > n_grids - boundary_width else 0)
            self.boundary[i, j, k] = torch.tensor(
                [sign(i), sign(j), sign(k)]
            )

    def mat_shape(self):
        return self.n_particles, 3, 3

    def vec_shape(self):
        return self.n_particles, 3

    def eye(self):
        return torch.eye(3, dtype=D_TYPE).to(self.device)

    @property
    def n_neighbors(self):
        return 27

    def step(self):
        def grid_iter(*shape):
            return itertools.product(*[range(dim) for dim in shape])

        # not changed
        grid_v, grid_m, dt, mat, n_particles, device, p_mass \
            = self.grid_v, self.grid_m, self.dt, self.material, self.n_particles, self.device, self.p_mass

        # changed
        x, v, F, C, Jp \
            = self.x, self.v, self.F, self.C, self.Jp

        dx = 1 / self.n_grids
        inv_dx = float(self.n_grids)
        n_neighbors = self.n_neighbors

        # clear
        grid_v.fill_(0)
        grid_m.fill_(0)

        # grid hashing
        base = (x * inv_dx - 0.5).int()         # N, 3 (int)
        diff = x * inv_dx - base.float()        # N, 3
        weights = [0.5 * (1.5 - diff) ** 2,
                   0.75 - (diff - 1) ** 2,
                   0.5 * (diff - 0.5) ** 2]     # 3, N, 3

        # affine
        F = (self.eye() + dt * C) @ F  # deformation gradient
        U, sig, V = torch.svd(F)
        J = torch.ones(n_particles).to(device)
        for i in range(3):
            s = sig[:, i]
            assert (s != 0).all()
            new_s = torch.where(mat == 2, torch.clamp(s, 1 - 2.5e-2, 1 + 4.5e-3), s)
            J *= new_s
            Jp *= s / new_s
            sig[:, i] = new_s
        J = J.reshape(-1, 1, 1)
        F = torch.where(mat.view(n_particles, 1, 1) == 0, self.eye() * J ** (1/3), F)
        F = torch.where(mat.view(n_particles, 1, 1) == 2,
                        U @ (self.eye() * sig.unsqueeze(1)) @ V.transpose(1, 2), F)

        # Hardening coefficient: snow gets harder when compressed
        h = torch.where(mat == 1, torch.ones_like(Jp) * 0.3, torch.exp(10 * (1.0 - Jp)))
        la = self.lambda_0 * h
        mu = torch.where(mat == 0, torch.zeros_like(Jp), h * self.mu_0)
        stress = 2 * mu.view(n_particles, 1, 1) * (F - U @ V.transpose(1, 2)) @ F.transpose(1, 2) \
                 + self.eye() * la.reshape(-1, 1, 1) * J * (J - 1)
        stress = (-dt * self.p_vol * 4 * inv_dx * inv_dx) * stress

        affine = stress + p_mass * C

        #p2g
        offset = torch.stack([torch.tensor([i, j, k]).to(device).int() for i, j, k in grid_iter(3, 3, 3)], 0)
        dists = (offset - diff.unsqueeze(1)) * dx
        weight = torch.stack([weights[i][:, 0] * weights[j][:, 1] * weights[k][:, 2] for i, j, k in grid_iter(3, 3, 3)], 1)  # N,27
        affine_batches = affine.unsqueeze(1).expand(-1, 27, -1, -1).reshape(-1, 3, 3)
        aff_dist_prod = affine_batches @ dists.reshape(-1, 3, 1)
        tmp_vs = p_mass * v.unsqueeze(1) + aff_dist_prod.reshape(n_particles, n_neighbors, -1)  # N, 27, 3
        weighted_vs = weight.unsqueeze(2) * tmp_vs
        weighted_ms = weight * p_mass
        tmp = base.unsqueeze(1) + offset.unsqueeze(0)
        tmp = tmp.reshape(n_particles * n_neighbors, 3).long()
        x_idx, y_idx, z_idx = tmp[:, 0], tmp[:, 1], tmp[:, 2]  # N * 27
        grid_v.index_put_(indices=[x_idx, y_idx, z_idx], values=weighted_vs.reshape(n_particles * n_neighbors, 3), accumulate=True)
        grid_m.index_put_(indices=[x_idx, y_idx, z_idx], values=weighted_ms.reshape(n_particles * n_neighbors), accumulate=True)

        # momentum conservation
        non_zero_grid_m = torch.where(grid_m > 1e-23, grid_m, 1e-23 * torch.ones_like(grid_m) * p_mass)
        grid_v *= 1 / non_zero_grid_m.unsqueeze(-1)

        grid_v[:, :, :, 1] -= dt * 50

        vel_bound_dot = torch.sign(grid_v) * self.boundary
        grid_v = torch.where(vel_bound_dot > 0, torch.zeros_like(grid_v), grid_v)

        # G2P
        v0 = v.detach()
        # 27, N, 2
        selected = grid_v[x_idx, y_idx, z_idx].reshape(n_particles, n_neighbors, -1)
        weighted_sel = weight.unsqueeze(2) * selected
        v = torch.sum(weighted_sel, 1)

        out = weighted_sel.reshape(-1, n_neighbors, 3, 1) * dists.reshape(-1, n_neighbors, 1, 3)
        C = 4 * inv_dx * inv_dx * torch.sum(out, 1)

        x += dt * v

        self.x, self.v, self.F, self.C, self.Jp \
            = x, v, F, C, Jp


class HashFluid(Simulatable):

    batch_size = 2 ** 16
    n_steps = 1000
    dt = 0.0001

    def __init__(self):
        super().__init__()
        self.phi = tcnn.NetworkWithInputEncoding(3, 3, encoding_config, network_config)
        self.vel = tcnn.NetworkWithInputEncoding(3, 3, encoding_config, network_config)
        self.optimizer = torch.optim.Adam(itertools.chain(self.phi.parameters(), self.vel.parameters()), lr=0.005)
        self.initialize()

        self.solver = self.make_mpm_solver(40)

        self.samples = torch.rand([self.batch_size, 3], device=device, dtype=torch.float32)

    def make_mpm_solver(self, grid_size):
        x = torch.rand([self.batch_size // 4, 3], device=device, dtype=torch.float32)
        x = x * 0.5 + 0.3
        v = torch.zeros_like(x)
        m = torch.ones_like(v[:, 0], dtype=torch.long) * 0
        return MPMSolver3D(x, v, m, self.dt, grid_size)

    def initialize(self):
        start = time.time()
        for i in range(self.n_steps):
            batch = torch.rand([self.batch_size, 3], device=device, dtype=torch.float32)
            pos = self.phi(batch)
            pos_gt = batch * 0.5 + 0.3
            relative_l2_error = (pos - pos_gt.to(pos.dtype))**2 / (pos.detach()**2 + 0.01)
            loss = relative_l2_error.mean()

            vel = self.vel(batch)
            relative_l2_error = vel ** 2 / (vel.detach() ** 2 + 0.01)
            loss = loss + relative_l2_error.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print("[Initialize]", time.time() - start)

    def advance(self, dt):
        for i in range(self.n_steps):
            start = time.time()
            X = torch.rand([1000, 3], device=device, dtype=torch.float32)
            x = self.phi(X)
            # F = jacobian(self.phi, X)
            # df = jacobian(self.vel, x)

            # pos = self.phi(batch)
            # pos_gt = batch * 0.5 + 0.5
            # relative_l2_error = (pos - pos_gt.to(pos.dtype)) ** 2 / (pos.detach() ** 2 + 0.01)
            # loss = relative_l2_error.mean()
            #
            # vel = self.vel(batch)
            # relative_l2_error = vel ** 2 / (vel.detach() ** 2 + 0.01)
            # loss = loss + relative_l2_error.mean()
            #
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()
            print("[Jacobian]", time.time() - start)

    def step(self, dt):
        # self.advance(dt)
        self.solver.step()
        # visualize_field(self.phi(self.samples))
        visualize_field(self.solver.x)


class NaiveNNFluid(Simulatable):

    batch_size = 2 ** 16
    n_steps = 100
    dt = 0.0001
    model_path = "save/naive_nn_fluid.pth"
    use_predict = False
    seq_dir = r"F:\Temp\simulation\watercube"
    seq_id = 0

    def __init__(self):
        super().__init__()
        self.enc = tcnn.Encoding(6, encoding_config)
        self.net = tcnn.Network(self.enc.n_output_dims, 3, network_config)
        self.optimizer = torch.optim.Adam(itertools.chain(self.enc.parameters(), self.net.parameters()), lr=0.01)
        self.solver = self.make_mpm_solver(40)
        self.samples = torch.rand([self.batch_size, 3], device=device, dtype=torch.float32)

    def init(self):
        self.solver = self.make_mpm_solver(40)
        # self.load()

    def load_seq(self):
        file_name = os.path.join(self.seq_dir, f"{self.seq_id:03d}.pth")
        obj = torch.load(file_name)
        return obj["x0"], obj["seq"]

    def make_mpm_solver(self, grid_size):
        x = torch.rand([self.batch_size // 4, 3], device=device, dtype=torch.float32)
        self.x0 = x.detach()
        x = x * 0.5 + 0.3
        v = torch.zeros_like(x)
        m = torch.ones_like(v[:, 0], dtype=torch.long) * 0
        return MPMSolver3D(x, v, m, self.dt, grid_size)

    def save(self):
        model = {
            "enc": self.enc.state_dict(),
            "net": self.net.state_dict()
        }
        torch.save(model, self.model_path)

    def load(self):
        model = torch.load(self.model_path)
        self.enc.load_state_dict(model["enc"])
        self.net.load_state_dict(model["net"])

    def predict(self, t):
        inputs = self.samples
        y = self.enc(inputs)
        outputs = self.net(y)

        self.solver.x = outputs

        # x0, xs = self.load_seq()
        # batch = torch.rand(self.batch_size, 2, device=device, dtype=torch.float32)
        # idx = (batch[:, 0] * x0.size(0)).type(torch.long)
        # xyz = x0[idx]
        # t = batch[:, 1:] * (xs.size(0) - 1) * 0.1
        # t = torch.ones_like(t) * 0.5
        # inputs = xyz
        # outputs = self.net(self.enc(inputs))
        #
        # t0 = (t / 0.01).type(torch.long)
        # t1 = t0 + 1
        # alpha = t / 0.01 - t0
        #
        # def x_at(t):
        #     return xs[t.squeeze().cpu(), idx.cpu()].to(device)
        #
        # self.solver.x = outputs

    def simulate(self, t):
        for i in range(int(100)):
            self.solver.step()
        inputs = torch.cat([self.x0, torch.ones_like(self.x0[:, 0:1]) * t], 1)
        for i in range(self.n_steps):
            y = self.enc(inputs)
            out = self.net(y)
            gt = self.solver.x

            relative_l2_error = (out - gt.to(out.dtype)) ** 2 / (out.detach() ** 2 + 0.01)
            loss = relative_l2_error.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if i == 0 or i == self.n_steps - 1:
                print("[Loss]", loss.item())

    def train(self):
        x0, xs = self.load_seq()
        batch = torch.rand(self.batch_size, 2, device=device, dtype=torch.float32)
        idx = (batch[:, 0] * x0.size(0)).type(torch.long)
        xyz = x0[idx]
        t = batch[:, 1:] * (xs.size(0) - 1) * 0.1
        t = torch.ones_like(t) * 0.5
        inputs = torch.cat([xyz, xyz], -1)
        outputs = self.net(self.enc(inputs))

        t0 = (t / 0.01).type(torch.long)
        t1 = t0 + 1
        alpha = t / 0.01 - t0

        def x_at(t):
            return xs[t.squeeze().cpu(), idx.cpu()].to(device)

        # gt = (1 - alpha) * x_at(t0) + alpha * x_at(t1)
        gt = x_at(t0)

        relative_l2_error = (outputs - gt.to(outputs.dtype)) ** 2 / (outputs.detach() ** 2 + 0.01)
        loss = relative_l2_error.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print("[Loss]", loss.item())

    def step(self, dt):
        try:
            self.__t
        except AttributeError:
            self.__t = 0
        self.__t = self.__t + 0.01

        if self.use_predict:
            self.predict(self.__t)
        else:
            self.train()
            self.save()
        visualize_field(self.solver.x)


class FluidSeqGen(Simulatable):
    batch_size = 2 ** 16
    dt = 0.0001
    seq_dir = r"F:\Temp\simulation\watercube"
    seq_id = 0
    playing = True

    def __init__(self):
        super().__init__()
        self.time = 0
        x_lag = torch.rand([self.batch_size // 4, 3], device=device, dtype=torch.float32)
        self.x_lag = x_lag.detach()
        self.solver = self.make_mpm_solver(x_lag, 40)
        self.xs = []

        if not self.playing:
            if self.seq_id < 0:
                index = 0
                while True:
                    file_name = os.path.join(self.seq_dir, f"{index:03d}.pth")
                    if not os.path.exists(file_name):
                        break
                self.seq_id = index
        else:
            self.load_seq()

    def make_mpm_solver(self, x_lag, grid_size):
        x = x_lag * 0.5 + 0.3
        v = torch.zeros_like(x)
        m = torch.ones_like(v[:, 0], dtype=torch.long) * 0
        return MPMSolver3D(x, v, m, self.dt, grid_size)

    def save_seq(self):
        file_name = os.path.join(self.seq_dir, f"{self.seq_id:03d}.pth")
        obj = {
            "x0": self.x_lag,
            "seq": torch.stack(self.xs, 0),
        }
        torch.save(obj, file_name)

    def load_seq(self):
        file_name = os.path.join(self.seq_dir, f"{self.seq_id:03d}.pth")
        obj = torch.load(file_name)
        self.x_lag = obj["x0"]
        self.xs = obj["seq"]

    def simulate(self):
        for i in range(int(100)):
            self.solver.step()
        x = self.solver.x
        self.xs.append(x.detach().cpu())

    def step(self, dt):
        self.time = self.time + 0.01
        if self.playing:
            self.solver.x = self.xs[int(self.time / 0.01) % len(self.xs)]
        else:
            self.simulate()
            self.save_seq()
        visualize_field(self.solver.x)


if __name__ == '__main__':
    NaiveNNFluid().run()
























