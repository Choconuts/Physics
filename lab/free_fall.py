#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : free_fall.py
@Author: Chen Yanzhen
@Date  : 2022/3/11 14:40
@Desc  : 
"""

import torch
from torch import nn


# class PhysicSystem:
#
#     def __init__(self):
#         super().__init__()
#
#     def calc_M(self, q, v, t):
#         raise NotImplementedError
#
#     def calc_T(self, q, v, t):
#         raise NotImplementedError
#
#     def calc_V(self, q, v, t):
#         raise NotImplementedError
#
#     def step(self, dt):
#         raise NotImplementedError


from functional import Simulatable, visualize_field, time_recorded


class FreeFallSystem(Simulatable):

    g = 9.8

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
        g = torch.tensor([0, -self.g])
        return -torch.dot(g, self.q)

    def step(self, dt):
        with time_recorded("step"):
            L = self.T() - self.V()
            F = torch.autograd.grad(L, self.q)[0]
            M = self.M()
            Mv = M @ self.v
            Mv_ = Mv + dt * F
            v_ = torch.inverse(M) @ Mv_

        v_ = v_ * 0.999

        self.v = v_
        self.q = self.q + v_ * dt
        visualize_field(self.q.view(1, 2))


# system = None
#
#
# @init_func
# def init():
#     global system
#     system = FreeFallSystem()
#
#
# @step_func
# def step():
#     with time_recorded("step"):
#         system.step(0.001)
#     visualize_field(system.q.view(1, 2))


if __name__ == '__main__':
    FreeFallSystem().run()
