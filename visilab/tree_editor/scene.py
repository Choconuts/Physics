#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : scene.py
@Author: Chen Yanzhen
@Date  : 2020/12/14 23:02
@Desc  : 
"""

from visilab.render_api import *
from .tree_node import *
import numpy as np
import json


_except_from = TreeNode()


class Scene(TreeNode):

    def gui(self, label="Root", *args, **kwargs):
        super().gui(self.label, default_open=True)

    def gen_displayables(self):
        for c in self.all_children():
            if isinstance(c, Displayable):
                yield c

    def actions(self):

        class SaveAction(InputAction):

            def input_value(self) -> bool:
                self.changed, self.default = imgui.input_text("file", self.default, 100)

            def on_confirmed(self):
                self.node.save(self.default)

        return super().actions() + [SaveAction("Save", self, "", "Input a file name")]

    def save(self, file):
        try:
            with open(file, 'w') as fp:
                json.dump(self, fp, default=self.cls_serializer, indent=4)
        except FileNotFoundError:
            pass

    @staticmethod
    def load(file):
        with open(file, 'r') as fp:
            return json.load(fp, object_hook=JsonCode.common_unserialize)


@TreeNode.as_builder("Obj")
class ObjNode(TreeNode, MeshLikeDisplayable):

    def __init__(self, label="Obj"):
        super().__init__(label)
        self.geo = GeoType.Triangle
        self.lighting = LightMode.Phong
        self.color = [1., 1., 1.]
        self.colors = None
        self.point_width = 5
        self.occlusion = True
        self.mesh = Mesh()

    def inspect(self, label, *args, **kwargs):
        self.changed = ImguiTool.input_vars(self, _except_from)

    def input_color(self, label, value):
        c, v = imgui.color_edit3(label, *value)
        return c, list(v)

    def actions(self):

        node = self

        class InputMesh(InputAction):

            def input_value(self) -> bool:
                self.changed, self.default = imgui.input_text("file", self.default, 100)

            def on_confirmed(self):
                self.node.load_mesh(self.default)

        return super().actions() + [InputMesh("LoadObj", node, "", "Input a file")]

    def load_mesh(self, file):
        self.mesh.load(file)

    def can_render(self) -> bool:
        return self.mesh is not None and self.mesh.vertices.__len__() > 0

    def render_geometry(self) -> GeoType:
        return self.geo

    def render_positions(self) -> np.ndarray:
        return self.mesh.vertices

    def render_indices(self) -> np.ndarray:
        return np.array(self.mesh.faces)

    def render_lighting(self) -> LightMode:
        return self.lighting

    def render_occlusion(self) -> bool:
        return self.occlusion

    def render_point_width(self) -> int:
        return self.point_width


@TreeNode.as_builder("Geometry")
class GeoNode(ObjNode, MeshLikeDisplayable):

    def __init__(self, label="Geo"):
        super().__init__(label)
        self.starts = None
        self.ends = None
        self.colors = None
        self.indices = None

    def inspect(self, label, *args, **kwargs):
        ImguiTool.input_vars(self, _except_from)

    def as_points(self, points: np.ndarray, colors=None):
        self.starts = points
        if colors is not None:
            self.colors = colors
        self.geo = GeoType.Point

    def as_lines(self, starts, ends, colors=None):
        self.starts = starts
        self.ends = ends
        if colors is not None:
            self.colors = colors
        self.geo = GeoType.Line

    def can_render(self) -> bool:
        return self.starts is not None and self.geo == GeoType.Point or \
               self.starts is not None and self.ends is not None and self.geo == GeoType.Line

    def render_positions(self) -> np.ndarray:
        if self.geo == GeoType.Point:
            return self.starts
        if self.geo == GeoType.Line:
            return np.concatenate([self.starts, self.ends], axis=0)
        else:
            raise NotImplementedError

    def render_indices(self) -> np.ndarray:
        nv = len(self.starts)
        idx = np.linspace(0, nv, nv, endpoint=False, dtype=np.uint32)
        if self.geo == GeoType.Point:
            return idx
        elif self.geo == GeoType.Line:
            return np.stack([idx, idx + nv], axis=1).flatten()
        else:
            raise NotImplementedError

    def render_colors(self) -> np.ndarray:
        if self.colors is None:
            raise NotImplementedError
        if self.geo == GeoType.Point:
            return self.colors
        if self.geo == GeoType.Line:
            return np.concatenate([self.colors, self.colors], axis=0)
        else:
            raise NotImplementedError

    def render_point_width(self) -> int:
        return self.point_width

    def render_color(self) -> tuple:
        return self.color
