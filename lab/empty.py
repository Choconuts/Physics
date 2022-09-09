#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : empty.py
@Author: Chen Yanzhen
@Date  : 2022/3/10 22:59
@Desc  : 
"""

import torch
from torch.autograd.functional import jacobian


a = torch.tensor([1., 2, 3])
a.requires_grad_(True)
x = torch.tensor([2., 1, 1])
x.requires_grad_(True)

def calc(a, x):
    return torch.concat([a ** 3 + a.sum(), x[0:1]], 0)

b = calc(a, x)
# b = torch.ones(4)

# c = torch.autograd.grad(b, a, grad_outputs=torch.tensor([1.0, 1.0, 1.0]))


# def jacobian(outputs, inputs):
#     """
#
#     :param outputs: 1D vector (M)
#     :param inputs: 1D vector (N)
#     :return: 2D matrix (M x N)
#     """
#     return torch.stack([
#         torch.autograd.grad([outputs[i].sum()], [inputs], retain_graph=True, create_graph=True, allow_unused=True)[0]
#         for i in range(outputs.size(0))
#     ], dim=0)


# c = jacobian(calc, (a, x))
#
# da = torch.ones_like(a) * 0.001
# db = c[0] @ da + b
#
# b_ = calc(a + da, x)
# #
#
# class A:
#
#     d = 1
#
# class B(A):
#
#     c = 2


# print(B.__dict__)

# B.__dict__['d'] = 3

# print(B.__dict__)

# print(B.mro())

# print(c)
# print(b_ - db)

# import numpy as np
# n_seg = 3
# x, y = np.mgrid[0:1:n_seg * 1j, 0:1:n_seg * 1j]
# print(np.mgrid[0:n_seg:1, 0:n_seg:1])
# print(x)

# a = torch.linspace(1, 9, 9)
# a = a.reshape(3, 3)
#
# i = torch.tensor([1, 2], dtype=torch.long)
# j = torch.tensor([0, 1], dtype=torch.long)
#
# print(a)
# print(a[i, j])
#
# a = a.index_put((i, j), torch.tensor([0., 0.]))
# print(a)
#
# a = torch.linspace(1, 12, 12)
# a = a.reshape(3, 2, 2)
#
# a = a.permute(2, 0, 1)
# print(a)

a = torch.linspace(1, 9, 9).view(3, 3)
x = a[:, None, :].repeat(1, 3, 1)
y = a[:, :, None].repeat(1, 1, 3)

print(x * y)

if __name__ == '__main__':
    pass
