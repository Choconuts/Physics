#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : geometry.py
@Author: Chen Yanzhen
@Date  : 2022/3/11 16:11
@Desc  : 
"""

import numpy as np


class Mesh:

    def points(self) -> np.ndarray:
        raise NotImplementedError

    def uvs(self) -> np.ndarray:
        raise NotImplementedError

    def edges(self) -> np.ndarray:
        raise NotImplementedError


class Strand(Mesh):

    def __init__(self, n_seg):
        self.n_seg = n_seg
        self.n_node = n_seg + 1
        x = np.linspace(0, 1, self.n_seg + 1)
        y = np.ones_like(x)
        q = np.stack([x, y], 1)
        self.positions = q
        idx = np.linspace(0, self.n_node - 1, self.n_node - 1, endpoint=False, dtype=np.int)
        self.links = (idx, idx + 1)
        self.params = x

    def points(self) -> np.ndarray:
        return self.positions

    def uvs(self) -> np.ndarray:
        return self.params

    def edges(self) -> np.ndarray:
        return self.links


class Strand3D(Mesh):

    def __init__(self, n_seg):
        self.n_seg = n_seg
        self.n_node = n_seg + 1
        x = np.linspace(0, 1, self.n_seg + 1)
        y = np.ones_like(x)
        z = np.ones_like(x) * 0.5
        q = np.stack([x, y, z], 1)

        n_seg2 = self.n_seg // 2
        n_node2 = n_seg2 + 1
        z2 = np.linspace(0.5, 0, n_node2)
        x2 = np.ones_like(z2) * 0.5
        y2 = np.ones_like(x2)
        q2 = np.stack([x2, y2, z2], 1)

        self.positions = np.concatenate([q, q2])
        idx = np.linspace(0, self.n_node - 1, self.n_node - 1, endpoint=False, dtype=np.int)

        idx2 = np.linspace(self.n_node, self.n_node + n_node2 - 1, n_node2 - 1, endpoint=False, dtype=np.int)

        idx3_s = np.array([self.n_seg // 2])
        idx3_e = np.array([self.n_node])

        ia = np.concatenate([idx, idx2, idx3_s])
        ib = np.concatenate([idx + 1, idx2 + 1, idx3_e])

        self.links = (ia, ib)
        self.params = x

        self.n_seg += n_seg2
        self.n_node += n_node2

    def points(self) -> np.ndarray:
        return self.positions

    def uvs(self) -> np.ndarray:
        return self.params

    def edges(self) -> np.ndarray:
        return self.links


class Square(Mesh):

    def __init__(self, n_seg, direct='h'):
        self.n_seg = n_seg
        self.n_node = n_seg + 1
        n_node = self.n_node
        x, z = np.mgrid[0:1:n_node * 1j, 0:1:n_node * 1j]
        x = x.flatten()
        z = z.flatten()
        y = np.ones_like(x)
        q = np.stack([z, y, x], 1)
        if direct == 'v':
            x, y = np.mgrid[1:0:n_node * 1j, 1:0:n_node * 1j]
            x = x.flatten()
            y = y.flatten()
            z = np.zeros_like(x)
            q = np.stack([y, x, z], 1)
        self.positions = q
        self.params = np.stack([x, z], 1)

        xi, zi = np.mgrid[0:n_node:1, 0:n_node:1]
        square_idx = xi * n_node + zi
        s1, e1 = square_idx[:n_seg, :], square_idx[1:, :]
        s2, e2 = square_idx[:, :n_seg], square_idx[:, 1:]

        s = np.concatenate([s1.flatten(), s2.flatten()])
        e = np.concatenate([e1.flatten(), e2.flatten()])

        self.links = (s, e)

    def points(self) -> np.ndarray:
        return self.positions

    def uvs(self) -> np.ndarray:
        return self.params

    def edges(self) -> np.ndarray:
        return self.links


class ParamSquare(Mesh):

    def __init__(self, n_seg):
        self.n_seg = n_seg
        self.n_node = n_seg + 1
        n_node = self.n_node
        y, x = np.mgrid[1:0:n_node * 1j, 0:1:n_node * 1j]
        x = x.flatten()
        y = y.flatten()
        z = np.zeros_like(x)

        v, u = np.mgrid[0:1:n_node * 1j, 0:1:n_node * 1j]
        u = u.flatten()
        v = v.flatten()

        q = np.stack([x, y, z, u, v], 1)
        self.positions = q
        self.params = np.stack([x, z], 1)

        xi, zi = np.mgrid[0:n_node:1, 0:n_node:1]
        square_idx = xi * n_node + zi
        s1, e1 = square_idx[:n_seg, :], square_idx[1:, :]
        s2, e2 = square_idx[:, :n_seg], square_idx[:, 1:]

        s = np.concatenate([s1.flatten(), s2.flatten()])
        e = np.concatenate([e1.flatten(), e2.flatten()])

        self.links = (s, e)

    def points(self) -> np.ndarray:
        return self.positions

    def uvs(self) -> np.ndarray:
        return self.params

    def edges(self) -> np.ndarray:
        return self.links

