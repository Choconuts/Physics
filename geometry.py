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

    def points(self) -> np.ndarray:
        return self.positions

    def edges(self) -> np.ndarray:
        return self.links


class Square(Mesh):

    def __init__(self, n_seg):
        self.n_seg = n_seg
        self.n_node = n_seg + 1
        x = np.linspace(0, 1, self.n_seg + 1)
        y = np.ones_like(x)
        z = np.linspace(0, 1, self.n_seg + 1)
        q = np.stack([x, y, z], 1)
        self.positions = q
        idx = np.linspace(0, self.n_node - 1, self.n_node - 1, endpoint=False, dtype=np.int)
        self.links = (idx, idx + 1)

