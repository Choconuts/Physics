#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : easy_lab.py
@Author: Chen Yanzhen
@Date  : 2022/3/11 14:39
@Desc  : 
"""

import numpy as np
import torch
from torch import nn
from torch.autograd.functional import jacobian
from functional import Simulatable, visualize_field, time_recorded
from geometry import Strand

torch.set_default_dtype(torch.float64)
DEVICE = 'cuda'


class MassStrand(Simulatable):
    k = 5000.
    rho = 1.
    dt = 0.01

    def __init__(self, n_seg=4000):
        super().__init__(n_seg=4000)
        self.strand = Strand(n_seg)
        self.q = nn.Parameter(torch.tensor(self.strand.points().flatten(), device=DEVICE))
        self.v = nn.Parameter(torch.zeros_like(self.q, device=DEVICE))

    def M(self):
        n_node = self.strand.n_node
        n_seg = self.strand.n_seg
        M = torch.zeros(n_node * 2, n_node * 2, device=DEVICE)
        for i in range(n_seg):
            a = i * 2
            b = (i + 1) * 2
            s0 = 0
            s1 = 1 / n_seg

            block = torch.tensor([
                [1, 0],
                [0, 1],
            ], device=DEVICE) * 1 / 6 * (s1 - s0) * self.rho

            def set_block(c, d):
                M[c:c + 2, d:d + 2] = M[c:c + 2, d:d + 2] + block

            set_block(a, a)
            set_block(a, a)
            set_block(b, b)
            set_block(b, b)
            set_block(a, b)
            set_block(b, a)

        return M

    def T(self, v=None):
        if v is None:
            v = self.v
        return 0.5 * v.t() @ self.M() @ v

    def V(self, q=None):
        n_seg = self.strand.n_seg
        if q is None:
            q = self.q
        V = 0
        g = -9.8
        for i in range(n_seg):
            a = i * 2
            b = (i + 1) * 2
            x0 = q[a]
            x1 = q[b]
            y0 = q[a + 1]
            y1 = q[b + 1]
            s0 = 0
            s1 = 1 / n_seg

            wx = (x1 - x0) / (s1 - s0)
            wy = (y1 - y0) / (s1 - s0)

            w = torch.sqrt(wx ** 2 + wy ** 2)

            V = V - self.rho * (s1 - s0) * g * (y0 + y1) / 2
            V = V + 1 / 2 * self.k * (s1 - s0) * (w - 1) ** 2

        return V

    def F(self, q, v):
        L = self.T(v) - self.V(q)
        F = torch.autograd.grad(L, q, retain_graph=True, create_graph=True)[0]
        return F

    def step(self, dt):

        with time_recorded("All", True):
            F = self.F(self.q, self.v)
            with time_recorded("Jacobi", True):
                F_q, F_v = jacobian(self.F, (self.q, self.v), vectorize=True)

            M = self.M()

            N = M.size(0)
            G = torch.zeros(4, N, device=DEVICE)
            G[0, 0] = G[1, 1] = G[2, -2] = G[3, -1] = 1
            A = torch.zeros(N + 4, N + 4, device=DEVICE)
            A[:N, :N] = M - F_q * dt ** 2
            A[N:, :N] = G
            A[:N, N:] = G.t()

            b = torch.zeros(N + 4, device=DEVICE)
            b[:N] = M @ self.v + F * dt
            with time_recorded("Solve", True):
                v_ = torch.linalg.solve(A, b)[:N]

            self.v = v_
            self.q = self.q + 0.5 * (self.v + v_) * dt

            with time_recorded("Visualize"):
                visualize_field(
                    self.q.view(-1, 2)[:, :2],
                    graphs=self.strand.edges()
                )


class MassCloth(Simulatable):
    k = 5000.
    rho = 1.
    dt = 0.01

    def __init__(self, n_seg=20):
        super().__init__(n_seg=20)
        self.strand = Strand(n_seg)
        self.q = nn.Parameter(torch.tensor(self.strand.points().flatten()))
        self.v = nn.Parameter(torch.zeros_like(self.q))

    def M(self):
        n_node = self.strand.n_node
        n_seg = self.strand.n_seg
        M = torch.zeros(n_node * 2, n_node * 2)
        for i in range(n_seg):
            a = i * 2
            b = (i + 1) * 2
            s0 = 0
            s1 = 1 / n_seg

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
        n_seg = self.strand.n_seg
        if q is None:
            q = self.q
        V = 0
        g = -9.8
        for i in range(n_seg):
            a = i * 2
            b = (i + 1) * 2
            x0 = q[a]
            x1 = q[b]
            y0 = q[a + 1]
            y1 = q[b + 1]
            s0 = 0
            s1 = 1 / n_seg

            wx = (x1 - x0) / (s1 - s0)
            wy = (y1 - y0) / (s1 - s0)

            w = torch.sqrt(wx ** 2 + wy ** 2)

            V = V - self.rho * (s1 - s0) * g * (y0 + y1) / 2
            V = V + 1 / 2 * self.k * (s1 - s0) * (w - 1) ** 2

        return V

    def F(self, q, v):
        L = self.T(v) - self.V(q)
        F = torch.autograd.grad(L, q, retain_graph=True, create_graph=True)[0]
        return F

    def step(self, dt):

        F = self.F(self.q, self.v)
        F_q, F_v = jacobian(self.F, (self.q, self.v), vectorize=True)
        M = self.M()

        N = M.size(0)
        G = torch.zeros(4, N)
        G[0, 0] = G[1, 1] = G[2, -2] = G[3, -1] = 1
        A = torch.zeros(N + 4, N + 4)
        A[:N, :N] = M - F_q * dt ** 2
        A[N:, :N] = G
        A[:N, N:] = G.t()

        b = torch.zeros(N + 4)
        b[:N] = M @ self.v + F * dt

        v_ = torch.linalg.solve(A, b)[:N]

        self.v = v_
        self.q = self.q + 0.5 * (self.v + v_) * dt

        visualize_field(
            self.q.view(-1, 2)[:, :2],
            graphs=self.strand.edges()
        )


if __name__ == '__main__':
    MassStrand().run()
