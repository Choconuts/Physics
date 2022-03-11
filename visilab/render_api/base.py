#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : base.py
@Author: Chen Yanzhen
@Date  : 2020/12/13 13:48
@Desc  : 
"""


from enum import Enum
from visilab.utils import *
import numpy as np


@JsonCode.decoder
class GeoType(JsonCode, Enum):
    Point = 0
    Line = 1
    Triangle = 2


@JsonCode.decoder
class LightMode(JsonCode, Enum):
    NoLight = 0
    Phong = 1


class Inputable(JsonCode):

    def input(self, label, *args, **kwargs) -> bool:
        self.changed = None
        ImguiTool.is_any_window_focused()
        self.gui(label, *args, **kwargs)
        return self.changed

    def gui(self, label, *args, **kwargs):
        raise NotImplementedError

    @property
    def changed(self):
        try:
            return self._changed
        except AttributeError:
            self._changed = False
            return self._changed

    @changed.setter
    def changed(self, value):
        if value is None:
            self._changed = False
        else:
            self._changed = self.changed or value


class Displayable:

    def render_layer(self) -> int:
        raise NotImplementedError

    def can_render(self) -> bool:
        raise NotImplementedError


class MeshLikeDisplayable(Displayable):

    def render_geometry(self) -> GeoType:
        raise NotImplementedError

    def render_positions(self) -> np.ndarray:
        raise NotImplementedError

    def render_indices(self) -> np.ndarray:
        raise NotImplementedError

    def render_lighting(self) -> LightMode:
        raise NotImplementedError

    def render_color(self) -> tuple:
        raise NotImplementedError

    def render_colors(self) -> np.ndarray:
        raise NotImplementedError

    def render_normals(self) -> np.ndarray:
        raise NotImplementedError

    def render_point_width(self) -> int:
        raise NotImplementedError

    def render_occlusion(self) -> bool:
        raise NotImplementedError



