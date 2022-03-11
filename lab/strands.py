#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : strands.py
@Author: Chen Yanzhen
@Date  : 2022/3/10 15:19
@Desc  : 
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from torch import nn
import numpy as np
from visilab import *
from torch.autograd.functional import jacobian
import time


class Lag:

    def __init__(self):
        super().__init__()
        self.q = nn.Parameter(torch.tensor([0.5, 1]))
        self.v = nn.Parameter(torch.tensor([0., 0.]))

    def M(self):
        return torch.Tensor([
            [1, 0],
            [0, 1],
        ])

    def T(self):
        return 0.5 * self.v.t() @ self.M() @ self.v

    def V(self):
        g = torch.tensor([0, -9.8])
        return -torch.dot(g, self.q)


class StrandsExplicitSimulator:

    def __init__(self, dt=0.001):
        self.dt = dt

    def step(self, lag: Lag):
        L = lag.T() - lag.V()
        F = torch.autograd.grad(L, lag.q)[0]
        M = lag.M()
        Mv = M @ lag.v
        Mv_ = Mv + self.dt * F
        v_ = torch.inverse(M) @ Mv_

        v_ = v_ * 0.999

        lag.v = v_
        lag.q = lag.q + v_ * self.dt

        lag.q[0] = 0.
        lag.q[1] = 1.
        lag.q[-1] = 1.
        lag.q[-2] = 1.

        point = lag.q.detach().numpy()
        # return [[0, 1], point, [1, 1.]],

        return np.reshape(point, [lag.n_node, 2])[:, :2],


class ImplicitSimulator:

    def __init__(self, dt=0.01):
        self.dt = dt

        self.running = 0.

    def step(self, lag: Lag):

        def calcF(q, v):
            L = lag.T(v) - lag.V(q)
            F = torch.autograd.grad(L, q, retain_graph=True, create_graph=True)[0]
            return F

        # L = lag.T() - lag.V()
        # F = torch.autograd.grad(L, lag.q, retain_graph=True, create_graph=True)[0]
        F = calcF(lag.q, lag.v)
        st = time.time()
        F_q, F_v = jacobian(calcF, (lag.q, lag.v), vectorize=True)
        ed = time.time()
        M = lag.M()

        N = M.size(0)
        G = torch.zeros(4, N, dtype=M.dtype)
        G[0, 0] = 1
        G[1, 1] = 1
        G[2, -2] = 1
        G[3, -1] = 1
        A = torch.zeros(N + 4, N + 4, dtype=M.dtype)
        A[:N, :N] = M - F_q * self.dt ** 2
        A[N:, :N] = G
        A[:N, N:] = G.t()

        b = torch.zeros(N + 4, dtype=M.dtype)
        b[:N] = M @ lag.v + F * self.dt

        # v_ = (torch.inverse(A) @ b)[:N]
        v_ = torch.linalg.solve(A, b)[:N]

        self.running += ed - st
        # print(self.running)

        # A = M - F_q * self.dt ** 2 - F_v * self.dt
        # b = M @ lag.v + F * self.dt - F_v @ lag.v * self.dt
        #
        # v_ = torch.inverse(A) @ b

        # A = M - F_q * self.dt ** 2
        # b = M @ lag.v + F * self.dt
        #
        # v__ = torch.inverse(A) @ b
        #
        # A = M
        # b = M @ lag.v + F * self.dt
        #
        # v___ = torch.inverse(A) @ b

        # dq = torch.rand_like(lag.q) * 0.001
        # dv = torch.rand_like(lag.q) * 0.001

        # F0 = calcF(lag.q + dq, lag.v)
        # F1 = calcF(lag.q + dq, lag.v + dv)
        # F2 = F + F_q @ dq + F_v @ dv
        # print(v__ - v___)
        # print(jacobian(lag.T, lag.v))

        # Mv = M @ lag.v
        # Mv_ = Mv + self.dt * F
        # v_ = torch.inverse(M) @ Mv_

        # v__[1] = v__[1] * 0

        lag.v = v_
        lag.q = lag.q + 0.5 * (lag.v + v_) * self.dt

        point = lag.q.detach().numpy()

        return np.reshape(point, [lag.n_node, 2])[:, :2],


def show_simulation(sim: StrandsExplicitSimulator, MyLag):
    lab = LabViewer((1200, 800))

    p_thick = 0.5

    def expand(arr, scale=0.3, offset=0.5, thick=p_thick):
        return np.concatenate((arr, np.ones_like(arr)[:, 0:1] * thick * scale + offset - scale * 0.5), axis=1)

    vf = VisibleField("Str")

    @TreeNode.as_builder
    class LabOp(TreeNode):

        def __init__(self, label="Stiffness"):
            super().__init__(label)
            self.value = 1000.0

        def inspect(self, label, *args, **kwargs):
            import imgui
            self.changed, self.value = imgui.input_float(label, self.value)

    op = LabOp()

    lab.inputting("Op", op)

    @lab.coroutine
    def roll():
        lag = MyLag()
        yield vf.ui_window()
        while True:
            list_of_points = sim.step(lag)
            all_points = [
                np.concatenate([np.array(points), np.ones_like(np.array(points)[:, 0:1]) * 0.5], 1)
                for points in list_of_points
            ]
            all_points = np.concatenate(all_points, 0)

            edges = [[], []]
            total_points = 0
            for i, points, in enumerate(list_of_points):
                n_points = len(points)
                starts = np.linspace(0, n_points - 1, n_points - 1, endpoint=False, dtype=np.int)
                edges[0].extend(starts + total_points)
                edges[1].extend(starts + 1 + total_points)
                total_points += n_points

            vf.update("", all_points)
            vf.scalar("node", np.ones_like(all_points[:, 0]))
            vf.graph("string", edges, colors=None)

            yield vf.displayables()


class StrandLag(Lag):

    def __init__(self):
        super().__init__()
        self.q = nn.Parameter(torch.tensor([0.3, 0.7]))
        self.v = nn.Parameter(torch.tensor([0., 0.]))
        self.p0 = torch.tensor([0., 1])
        self.p1 = torch.tensor([1., 1])
        self.l0 = self.length().detach()
        self.k = 10

    def length(self):
        return torch.norm(self.q - self.p0) + torch.norm(self.q - self.p1)

    def M(self):
        return torch.Tensor([
            [1, 0],
            [0, 1],
        ])

    def T(self):
        return 0.5 * self.v.t() @ self.M() @ self.v

    def V(self):
        g = torch.tensor([0, -9.8])
        return -torch.dot(g, self.q) + self.k * torch.clip(self.length() - self.l0, 0) ** 2


class MassStrandLag(Lag):

    def __init__(self, n_seg=20):
        super().__init__()
        x = np.linspace(0, 1, n_seg + 1)
        y = np.ones_like(x)
        q = np.stack([x, y])
        q = q.transpose().flatten()     # [ x0, y0,  x1, y1,  ...,  xn, yn ]
        self.q = nn.Parameter(torch.tensor(q))
        self.v = nn.Parameter(torch.zeros_like(self.q))

        self.n_seg = n_seg
        self.n_node = n_seg + 1

        self.rho = 1.0
        self.k = 5000

    def M(self):
        M = torch.zeros(self.n_node * 2, self.n_node * 2, dtype=self.q.dtype)
        for i in range(self.n_seg):
            a = i * 2
            b = (i + 1) * 2
            x0 = self.q[a]
            x1 = self.q[b]
            y0 = self.q[a + 1]
            y1 = self.q[b + 1]
            s0 = 0
            s1 = 1 / self.n_seg

            block = torch.tensor([
                [1, 0],
                [0, 1],
            ]) * 1 / 6 * (s1 - s0) * self.rho

            def set_block(c, d):
                M[c:c + 2, d:d + 2] = M[c:c + 2, d:d + 2] + block

            set_block(a, a)
            set_block(a, a)
            set_block(b, b)
            set_block(b, b)
            set_block(a, b)
            set_block(b, a)

        torch.inverse(M)
        return M

    def T(self, v=None):
        if v is None:
            v = self.v
        return 0.5 * v.t() @ self.M() @ v

    def V(self, q=None):
        if q is None:
            q = self.q
        V = 0
        g = -9.8
        for i in range(self.n_seg):
            a = i * 2
            b = (i + 1) * 2
            x0 = q[a]
            x1 = q[b]
            y0 = q[a + 1]
            y1 = q[b + 1]
            s0 = 0
            s1 = 1 / self.n_seg

            wx = (x1 - x0) / (s1 - s0)
            wy = (y1 - y0) / (s1 - s0)

            w = torch.sqrt(wx ** 2 + wy ** 2)

            V = V - self.rho * (s1 - s0) * g * (y0 + y1) / 2
            V = V + 1 / 2 * self.k * (s1 - s0) * (w - 1) ** 2

            # no bending

        return V


class EoLMassStrandLag(Lag):

    def __init__(self, n_seg=10):
        super().__init__()
        x = np.linspace(0, 1, n_seg + 1)
        y = np.ones_like(x)
        s = np.linspace(0, 1, n_seg + 1)
        q = np.stack([x, y, s])
        q = q.transpose().flatten()     # [ x0, y0, s0,  x1, y1, s1,  ...,  xn, yn, sn ]
        self.q = nn.Parameter(torch.tensor(q))
        self.v = nn.Parameter(torch.zeros_like(self.q))

        self.n_seg = n_seg
        self.n_node = n_seg + 1

        self.rho = 1.0
        self.k = 100

    def M(self):
        M = torch.zeros(self.n_node * 3, self.n_node * 3, dtype=self.q.dtype)
        for i in range(self.n_seg):
            a = i * 3
            b = (i + 1) * 3
            x0 = self.q[a]
            x1 = self.q[b]
            y0 = self.q[a + 1]
            y1 = self.q[b + 1]
            s0 = self.q[a + 2]
            s1 = self.q[b + 2]

            wx = (x1 - x0) / (s1 - s0)
            wy = (y1 - y0) / (s1 - s0)

            block = torch.tensor([
                [1, 0, -wx],
                [0, 1, -wy],
                [-wx, -wy, wx*wx + wy*wy]
            ]) * 1 / 6 * (s1 - s0) * self.rho

            def set_block(c, d):
                M[c:c + 3, d:d + 3] = M[c:c + 3, d:d + 3] + block

            set_block(a, a)
            set_block(a, a)
            set_block(b, b)
            set_block(b, b)
            set_block(a, b)
            set_block(b, a)

        # idx = torch.tensor([k for k in range(3 * self.n_node) if k % 3 != 2], dtype=torch.long)
        # M = M[idx][:, idx]
        return M

    def T(self):
        return 0.5 * self.v.t() @ self.M() @ self.v

    def V(self):
        V = 0
        g = -9.8
        for i in range(self.n_seg):
            a = i * 3
            b = (i + 1) * 3
            x0 = self.q[a]
            x1 = self.q[b]
            y0 = self.q[a + 1]
            y1 = self.q[b + 1]
            s0 = self.q[a + 2]
            s1 = self.q[b + 2]

            wx = (x1 - x0) / (s1 - s0)
            wy = (y1 - y0) / (s1 - s0)

            w = torch.sqrt(wx ** 2 + wy ** 2)

            V = V + self.rho * (s1 - s0) * g * (y0 + y1) / 2
            V = V + 1 / 2 * self.k * (s1 - s0) * (w - 1) ** 2

            # no bending

        return V


show_simulation(ImplicitSimulator(), MassStrandLag)


if __name__ == '__main__':
    pass





