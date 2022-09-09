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
from geometry import Strand, Square, Strand3D, ParamSquare


torch.set_default_dtype(torch.float64)
DEVICE = 'cpu'


class MassStrand(Simulatable):
    k = 5000.
    rho = 1.
    dt = 0.01

    def __init__(self, n_seg=40):
        super().__init__(n_seg=40)
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

        with time_recorded("All"):
            F = self.F(self.q, self.v)
            with time_recorded("Jacobi"):
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
            with time_recorded("Solve"):
                v_ = torch.linalg.solve(A, b)[:N]

            self.v = v_
            self.q = self.q + 0.5 * (self.v + v_) * dt

            with time_recorded("Visualize"):
                visualize_field(
                    self.q.view(-1, 2)[:, :2],
                    graphs=self.strand.edges()
                )


class MassStrand(Simulatable):
    k = 5000.
    rho = 1.
    dt = 0.01

    def __init__(self, n_seg=10):
        super().__init__(n_seg)
        self.strand = Strand3D(n_seg)
        self.n_node0 = (n_seg) * 3
        self.q = nn.Parameter(torch.tensor(self.strand.points().flatten(), device=DEVICE))
        self.v = nn.Parameter(torch.zeros_like(self.q, device=DEVICE))

    def M(self):
        n_node = self.strand.n_node
        n_seg = self.strand.n_seg
        s_idx, e_idx = self.strand.edges()

        M = torch.zeros(n_node, n_node, 3, 3)

        a_idx = torch.tensor(s_idx, dtype=torch.long)
        b_idx = torch.tensor(e_idx, dtype=torch.long)

        all_idx = torch.linspace(0, n_node - 1, n_node, dtype=torch.long)

        s0 = 0
        s1 = 1 / n_seg

        block = torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]) * 1 / 6 * (s1 - s0) * self.rho

        M = M.index_put((a_idx, b_idx), block)
        M = M.index_put((b_idx, a_idx), block)
        M = M.index_put((all_idx, all_idx), 2 * block)

        M = M.permute(0, 2, 1, 3).reshape(n_node * 3, n_node * 3)
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

        s_idx, e_idx = self.strand.edges()
        a_idx = s_idx
        b_idx = e_idx

        x0 = q.view(-1, 3)[a_idx]
        x1 = q.view(-1, 3)[b_idx]
        s0 = 0
        s1 = 1 / n_seg

        w = (x1 - x0) / (s1 - s0)
        w = torch.linalg.norm(w, dim=1)

        V = V - self.rho * (s1 - s0) * (-9.8 * (x0 + x1)[:, 1] / 2).sum()
        V = V + 1 / 2 * (self.k * (s1 - s0) * (w - 1) ** 2).sum()

        return V

    def F(self, q, v):
        L = self.T(v) - self.V(q)
        F = torch.autograd.grad(L, q, retain_graph=True, create_graph=True)[0]
        return F

    def step(self, dt):

        with time_recorded("All"):
            F = self.F(self.q, self.v)
            with time_recorded("Jacobi"):
                F_q, F_v = jacobian(self.F, (self.q, self.v), vectorize=True)

            M = self.M()

            N_Cons = 9
            N = M.size(0)
            G = torch.zeros(N_Cons, N, device=DEVICE)
            G[0, 0] = G[1, 1] = G[2, 2] = 1
            G[3, self.n_node0] = G[4, self.n_node0 + 1] = G[5, self.n_node0 + 2] = 1
            G[6, -3] = G[7, -2] = G[8, -1] = 1
            A = torch.zeros(N + N_Cons, N + N_Cons, device=DEVICE)
            A[:N, :N] = M - F_q * dt ** 2
            A[N:, :N] = G
            A[:N, N:] = G.t()

            b = torch.zeros(N + N_Cons, device=DEVICE)
            b[:N] = M @ self.v + F * dt
            with time_recorded("Solve"):
                v_ = torch.linalg.solve(A, b)[:N]

            self.v = v_
            self.q = self.q + 0.5 * (self.v + v_) * dt

            with time_recorded("Visualize"):
                visualize_field(
                    self.q.view(-1, 3)[:, :3],
                    graphs=self.strand.edges()
                )


class MassCloth(Simulatable):
    k = 5000.
    rho = 1.
    dt = 0.01
    g = -9.8

    n_segment = 10

    def __init__(self, n_seg=40):
        super().__init__(n_seg=40)
        n_seg = self.n_segment
        self.square = Square(n_seg)
        self.q = nn.Parameter(torch.tensor(self.square.points().flatten()))
        self.v = nn.Parameter(torch.zeros_like(self.q))

    def M(self):

        s_idx, e_idx = self.square.edges()

        n_node = self.square.n_node
        n_node_tot = n_node * n_node
        n_seg = self.square.n_seg

        M = torch.zeros(n_node_tot, n_node_tot, 3, 3)

        a_idx = torch.tensor(s_idx, dtype=torch.long)
        b_idx = torch.tensor(e_idx, dtype=torch.long)

        s0 = 0
        s1 = 1 / n_seg

        block = torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]) * 1 / 6 * (s1 - s0) * self.rho

        M = M.index_put((a_idx, b_idx), block)
        M = M.index_put((b_idx, a_idx), block)
        M = M.index_put((a_idx, a_idx), 2 * block, accumulate=True)
        M = M.index_put((b_idx, b_idx), 2 * block, accumulate=True)

        M = M.permute(0, 2, 1, 3).reshape(n_node_tot * 3, n_node_tot * 3)
        # print(M.det())
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

        s_idx, e_idx = self.square.edges()
        n_seg = self.square.n_seg
        a_idx = s_idx
        b_idx = e_idx

        x0 = q.view(-1, 3)[a_idx]
        x1 = q.view(-1, 3)[b_idx]
        s0 = 0
        s1 = 1 / n_seg

        w = (x1 - x0) / (s1 - s0)
        w = torch.linalg.norm(w, dim=1)

        V = V - self.rho * (s1 - s0) * (self.g * (x0 + x1)[:, 1] / 2).sum()
        # V = V - self.rho * (s1 - s0) * (q.view(-1, 3)[:, 1] * self.g).sum()
        V = V + 1 / 2 * (self.k * (s1 - s0) * (w - 1) ** 2).sum()

        return V

    def F(self, q, v):
        L = self.T(v) - self.V(q)
        F = torch.autograd.grad(L, q, retain_graph=True, create_graph=True)[0]
        return F

    def step(self, dt):
        with time_recorded("ALL"):
            F = self.F(self.q, self.v)
            F_q, F_v = jacobian(self.F, (self.q, self.v), vectorize=True)
            M = self.M()

            V_ = self.V(self.q + M.inverse() @ F * dt)
            V = self.V()
            # assert V_ <= V

            N = M.size(0)

            N_Cons = 6
            G = torch.zeros(N_Cons, N)
            left_top = self.square.n_seg * 3
            G[0, 0] = G[1, 1] = G[2, 2] = 1
            G[3, left_top] = G[4, left_top + 1] = G[5, left_top + 2] = 1
            A = torch.zeros(N + N_Cons, N + N_Cons)
            A[:N, :N] = M - F_q * dt ** 2
            A[N:, :N] = G
            A[:N, N:] = G.t()
            b = torch.zeros(N + N_Cons)
            b[:N] = M @ self.v + F * dt

            v_ = torch.linalg.solve(A, b)[:N]

            self.v = v_
            self.q = self.q + 0.5 * (self.v + v_) * dt

            # assert self.V() <= V

            visualize_field(
                self.q.view(-1, 3)[:, :3],
                graphs=self.square.edges()
            )


class MassCloth(Simulatable):
    k = 5000.
    rho = 1.
    dt = 0.01
    g = -9.8
    force = 10

    n_segment = 4

    def __init__(self, n_seg=40):
        super().__init__(n_seg=40)
        n_seg = self.n_segment
        self.DoF = 5

        self.square = ParamSquare(n_seg)
        self.q = nn.Parameter(torch.tensor(self.square.points().flatten())).to(DEVICE)
        self.v = nn.Parameter(torch.zeros_like(self.q)).to(DEVICE)

    def M(self, q=None):
        if q is None:
            q = self.q
        s_idx, e_idx = self.square.edges()

        n_node = self.square.n_node
        n_node_tot = n_node * n_node
        n_seg = self.square.n_seg
        n_edge = s_idx.shape[0] // 2

        M = torch.zeros(n_node_tot, n_node_tot, self.DoF, self.DoF, device=DEVICE)

        # vertical
        a_idx = torch.tensor(s_idx, dtype=torch.long, device=DEVICE)[:n_edge]
        b_idx = torch.tensor(e_idx, dtype=torch.long, device=DEVICE)[:n_edge]

        x0 = q.view(-1, self.DoF)[a_idx][:, :3]
        x1 = q.view(-1, self.DoF)[b_idx][:, :3]
        s0 = q.view(-1, self.DoF)[a_idx][:, 4]
        s1 = q.view(-1, self.DoF)[b_idx][:, 4]

        du = (s1 - s0).view(-1, 1)
        w = (x1 - x0) / du

        tmp = torch.zeros(n_edge, self.DoF, self.DoF, device=DEVICE)
        tmp[:, :3, :3] = torch.eye(3, device=DEVICE)
        tmp[:, 4:5, :3] = w.view(-1, 1, 3)
        tmp[:, :3, 4:5] = w.view(-1, 3, 1)
        tmp[:, 4:5, 4:5] = (w * w).sum(-1).view(-1, 1, 1)
        tmp = tmp * du.view(-1, 1, 1) * 1 / 6 * self.rho

        M = M.index_put((a_idx, b_idx), tmp, accumulate=True)
        M = M.index_put((b_idx, a_idx), tmp, accumulate=True)
        M = M.index_put((a_idx, a_idx), 2 * tmp, accumulate=True)
        M = M.index_put((b_idx, b_idx), 2 * tmp, accumulate=True)

        # horizontal
        a_idx = torch.tensor(s_idx, dtype=torch.long)[n_edge:]
        b_idx = torch.tensor(e_idx, dtype=torch.long)[n_edge:]

        x0 = q.view(-1, self.DoF)[a_idx][:, :3]
        x1 = q.view(-1, self.DoF)[b_idx][:, :3]
        s0 = q.view(-1, self.DoF)[a_idx][:, 3]
        s1 = q.view(-1, self.DoF)[b_idx][:, 3]

        du = (s1 - s0).view(-1, 1)
        w = (x1 - x0) / du

        tmp = torch.zeros(n_edge, self.DoF, self.DoF, device=DEVICE)
        tmp[:, :3, :3] = torch.eye(3, device=DEVICE)
        tmp[:, 3:4, :3] = w.view(-1, 1, 3)
        tmp[:, :3, 3:4] = w.view(-1, 3, 1)
        tmp[:, 3:4, 3:4] = (w * w).sum(-1).view(-1, 1, 1)
        tmp = tmp * du.view(-1, 1, 1) * 1 / 6 * self.rho

        M = M.index_put((a_idx, b_idx), tmp, accumulate=True)
        M = M.index_put((b_idx, a_idx), tmp, accumulate=True)
        M = M.index_put((a_idx, a_idx), 2 * tmp, accumulate=True)
        M = M.index_put((b_idx, b_idx), 2 * tmp, accumulate=True)

        M = M.permute(0, 2, 1, 3).reshape(n_node_tot * self.DoF, n_node_tot * self.DoF)
        M = M + torch.eye(n_node_tot * self.DoF, device=DEVICE) * 0.1
        # print(M.det())
        torch.inverse(M)
        return M

    def T(self, v=None, q=None):
        if v is None:
            v = self.v
        return 0.5 * v.t() @ self.M(q) @ v

    def V(self, q=None):
        if q is None:
            q = self.q

        V = 0
        s_idx, e_idx = self.square.edges()
        n_seg = self.square.n_seg
        n_edge = s_idx.shape[0] // 2

        # # Gravity
        # a_idx = s_idx[:n_edge]
        # b_idx = e_idx[:n_edge]
        # x0 = q.view(-1, self.DoF)[a_idx][:, :3]
        # x1 = q.view(-1, self.DoF)[b_idx][:, :3]
        # s0 = q.view(-1, self.DoF)[a_idx][:, 4]
        # s1 = q.view(-1, self.DoF)[b_idx][:, 4]
        #
        # V = V - self.rho * ((s1 - s0) * self.g * (x0 + x1)[:, 1] / 2).sum()
        #
        # a_idx = s_idx[n_edge:]
        # b_idx = e_idx[n_edge:]
        # x0 = q.view(-1, self.DoF)[a_idx][:, :3]
        # x1 = q.view(-1, self.DoF)[b_idx][:, :3]
        # s0 = q.view(-1, self.DoF)[a_idx][:, 3]
        # s1 = q.view(-1, self.DoF)[b_idx][:, 3]
        #
        # V = V - self.rho * ((s1 - s0) * self.g * (x0 + x1)[:, 1] / 2).sum()

        # vertical
        a_idx = s_idx[:n_edge]
        b_idx = e_idx[:n_edge]
        x0 = q.view(-1, self.DoF)[a_idx][:, :3]
        x1 = q.view(-1, self.DoF)[b_idx][:, :3]
        s0 = q.view(-1, self.DoF)[a_idx][:, 4]
        s1 = q.view(-1, self.DoF)[b_idx][:, 4]

        du = (s1 - s0).view(-1, 1)
        w = (x1 - x0) / du
        w = torch.linalg.norm(w, dim=1)
        V = V + 1 / 2 * (self.k * du * (w - 1) ** 2).sum()

        # horizontal
        a_idx = s_idx[n_edge:]
        b_idx = e_idx[n_edge:]
        x0 = q.view(-1, self.DoF)[a_idx][:, :3]
        x1 = q.view(-1, self.DoF)[b_idx][:, :3]
        s0 = q.view(-1, self.DoF)[a_idx][:, 3]
        s1 = q.view(-1, self.DoF)[b_idx][:, 3]

        du = (s1 - s0).view(-1, 1)
        w = (x1 - x0) / du
        w = torch.linalg.norm(w, dim=1)
        V = V + 1 / 2 * (self.k * du * (w - 1) ** 2).sum()

        return V

    def F(self, q, v):
        L = self.T(v, q) - self.V(q)
        F = torch.autograd.grad(L, q, retain_graph=True, create_graph=True)[0]
        return F

    def external_force(self):
        F_ex = torch.zeros_like(self.q)
        n_node = self.square.n_node
        xi = n_node // 2
        yi = n_node // 2
        n_idx = yi * n_node + xi
        n_idx *= self.DoF
        F_ex[n_idx] = 0
        F_ex[n_idx + 1] = 0.1 * self.force
        F_ex[n_idx + 2] = self.force
        return F_ex

    def constrained_matrix(self, A0):
        # N = A0.size(0)
        # N_Cons = 6
        # G = torch.zeros(N_Cons, N, device=DEVICE)
        # left_top = self.square.n_seg * self.DoF
        # G[0, 0] = G[1, 1] = G[2, 2] = 1
        # G[3, left_top] = G[4, left_top + 1] = G[5, left_top + 2] = 1
        # A = torch.zeros(N + N_Cons, N + N_Cons, device=DEVICE)
        # A[:N, :N] = A0
        # A[N:, :N] = G
        # A[:N, N:] = G.t()
        #
        # return A

        N = A0.size(0)
        n_node = self.square.n_node
        n_cons = n_node * 4
        G = torch.zeros(n_cons, N, device=DEVICE)
        left_top = self.square.n_seg * self.DoF
        # G[-1, 0] = G[-2, 1] = G[-3, 2] = 1
        # G[-4, left_top] = G[-5, left_top + 1] = G[-6, left_top + 2] = 1

        for i in range(n_node):
            G[i, i * self.DoF + 4] = 1
        for i in range(n_node):
            G[i + n_node, i * n_node * self.DoF + 3] = 1
        for i in range(n_node):
            G[i + n_node * 2, (i * n_node + n_node - 1) * self.DoF + 3] = 1
        for i in range(n_node):
            G[i + n_node * 3, (n_node * n_node - i - 1) * self.DoF + 4] = 1

        A = torch.zeros(N + n_cons, N + n_cons, device=DEVICE)
        A[:N, :N] = A0
        A[N:, :N] = G
        A[:N, N:] = G.t()

        return A

    def post_processing(self):
        s_idx, e_idx = self.square.edges()
        n_edge = s_idx.shape[0] // 2

        a_idx = torch.tensor(s_idx, dtype=torch.long, device=DEVICE)[:n_edge]
        b_idx = torch.tensor(e_idx, dtype=torch.long, device=DEVICE)[:n_edge]
        q = self.q.view(-1, self.DoF)
        v = self.v.view(-1, self.DoF)
        s0 = q[a_idx][:, 4]
        s1 = q[b_idx][:, 4]

        du = s1 - s0
        bad_idx = (du < 5e-2).nonzero().view(-1)

        assert (du < 0).nonzero().view(-1).numel() == 0

        if bad_idx.numel() > 0:
            ai = a_idx[bad_idx]
            bi = b_idx[bad_idx]
            avg_q = (q[ai] + q[bi]) / 2
            avg_v = (v[ai] + v[bi]) / 2
            for t in range(5):
                # self.q[ai * 5 + t] = avg_q[:, t]
                # self.q[bi * 5 + t] = avg_q[:, t]
                self.v[ai * 5 + t] = avg_v[:, t]
                self.v[bi * 5 + t] = avg_v[:, t]

        a_idx = torch.tensor(s_idx, dtype=torch.long, device=DEVICE)[n_edge:]
        b_idx = torch.tensor(e_idx, dtype=torch.long, device=DEVICE)[n_edge:]
        q = self.q.view(-1, self.DoF)
        v = self.v.view(-1, self.DoF)
        s0 = q[a_idx][:, 3]
        s1 = q[b_idx][:, 3]

        du = s1 - s0
        bad_idx = (du < 5e-2).nonzero().view(-1)

        assert (du < 0).nonzero().view(-1).numel() == 0

        if bad_idx.numel() > 0:
            ai = a_idx[bad_idx]
            bi = b_idx[bad_idx]
            avg_q = (q[ai] + q[bi]) / 2
            avg_v = (v[ai] + v[bi]) / 2
            for t in range(5):
                # self.q[ai * 5 + t] = avg_q[:, t]
                # self.q[bi * 5 + t] = avg_q[:, t]
                self.v[ai * 5 + t] = avg_v[:, t]
                self.v[bi * 5 + t] = avg_v[:, t]


    def step(self, dt):
        with time_recorded("ALL"):
            F = self.F(self.q, self.v)
            F_q, F_v = jacobian(self.F, (self.q, self.v), vectorize=True)
            M = self.M()

            V = self.V()
            # V.backward()

            N = M.size(0)
            A = self.constrained_matrix(M - F_q * dt ** 2)
            b = torch.zeros(A.size(0), device=DEVICE)
            b[:N] = M @ self.v + F * dt

            # external
            F_ex = self.external_force()
            b[:N] = b[:N] + F_ex * dt

            v_ = torch.linalg.solve(A, b)[:N]

            self.v = v_
            self.post_processing()
            self.q = self.q + 0.5 * (self.v + v_) * dt

            visualize_field(
                self.q.view(-1, self.DoF)[:, [0, 1, 2]],
                scalars=self.q.view(-1, self.DoF)[:, 4],
                graphs=self.square.edges()
            )


if __name__ == '__main__':
    MassCloth().run()
