#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : field.py
@Author: Chen Yanzhen
@Date  : 2020/12/18 14:28
@Desc  : 
"""


from visilab.render_api import *
from visilab.utils import *
from visilab.tree_editor import *
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import json
import imgui


_except_from = TreeNode()


class ScalarColorMap(EnumJsonCode):
    Inherit = 0
    Rainbow = 1
    Magma = 2
    Viridis = 3


class ScalarNormalizer(EnumJsonCode):
    CustomMinMax = 0
    AutoMinMax = 1
    ZeroToAutoMax = 2


class ScalarMapper(TreeNode):

    def __init__(self, label="Mapper"):
        super().__init__(label)
        self.color_map = ScalarColorMap.Inherit
        self.normalizer = ScalarNormalizer.AutoMinMax
        self._min_value = 0
        self._max_value = 1

    def clip(self, min_value, max_value):
        self._max_value = max_value
        self._min_value = min_value
        self.normalizer = ScalarNormalizer.CustomMinMax

    def actions(self):
        return []

    def inspect(self, label, *args, **kwargs):
        self.changed = ImguiTool.input_vars(self, _except_from)
        if self.normalizer == ScalarNormalizer.CustomMinMax:
            self.changed, (self._min_value, self._max_value) = \
                imgui.drag_float2("Min/Max", self._min_value, self._max_value)
            if self._min_value > self._max_value:
                self._max_value = self._min_value

    def map_to_colors(self, values):
        vmin, vmax = (values.min(), values.max()) if self.normalizer == ScalarNormalizer.AutoMinMax \
            else ((self._min_value, self._max_value) if self.normalizer == ScalarNormalizer.CustomMinMax else
                  (0, values.max()))
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        colors = self.cmap(self.color_map)(norm(values))
        return colors[:, :3]

    @staticmethod
    def cmap(self):
        if self == ScalarColorMap.Inherit:
            return plt.cm.rainbow
        elif self == ScalarColorMap.Rainbow:
            return plt.cm.rainbow
        elif self == ScalarColorMap.Magma:
            return plt.cm.magma
        elif self == ScalarColorMap.Viridis:
            return plt.cm.viridis
        else:
            raise NotImplementedError


class VisibleField(TreeNode):

    Window_Name = "Visible Fields"
    Always_Propagate = "Always_Propagate"
    Propagate_Only_Needed = "Propagate_Only_Needed"

    _global_decoder_map = JsonCode._global_decoder_map
    _global_decoder_map["ndarray"] = lambda x: np.array([])

    def __init__(self, label="Field"):
        super().__init__(label)
        self.positions = None
        self.visible = True

    def inherit_attribs(self):
        return {
            "positions": self.Always_Propagate,
            "visible": self.Propagate_Only_Needed
        }

    def actions(self):

        if self.__class__ != VisibleField:
            return [super().actions()[0]]

        class SaveAction(InputAction):

            def input_value(self) -> bool:
                self.changed, self.default = imgui.input_text("file", self.default, 100)

            def on_confirmed(self):
                self.node.save(self.default)

        class LoadAction(InputAction):

            def input_value(self) -> bool:
                self.changed, self.default = imgui.input_text("file", self.default, 100)

            def on_confirmed(self):
                try:
                    v = self.node.load(self.default)
                    self.node.children = v.children
                    self.node.visible = v.visible
                except FileNotFoundError:
                    pass

        class ClearAction(NodeAction):

            def gui(self, label, *args, **kwargs):
                self.node.children.clear()

        return [SaveAction("Save", self, "", "Input a file name"), LoadAction("Load", self, "", "Input a file name"),
                ClearAction("Clear", self)]

    def save(self, file):
        try:
            with open(file, 'w') as fp:
                json.dump(self, fp, default=self.cls_serializer, indent=4)
        except FileNotFoundError:
            pass

    @staticmethod
    def load(file):
        with open(file, 'r') as fp:
            return json.load(fp, object_hook=VisibleField.unserialize)

    @staticmethod
    def input_positions(k, v):
        return False, v

    def inspect(self, label, *args, **kwargs):
        changed, self.visible = imgui.checkbox("visible", self.visible)
        if changed:
            self._attrib_propagation()
        self.changed = changed

    def gui(self, label, *args, default_open=False, **kwargs):
        super().gui(self.label, *args, default_open=default_open, **kwargs)

    def _attrib_propagation(self, node=None, needed=[]):
        if node is None:
            [self._attrib_propagation(node, needed) for node in self.all_children()]
        if isinstance(node, VisibleField):
            for c in node.all_children():
                self._attrib_propagation(c, needed)
            for k, d in self.inherit_attribs().items():
                if not hasattr(node, k) or d == self.Propagate_Only_Needed and k not in needed:
                    continue
                if d == self.Always_Propagate and getattr(self, k) is not None \
                        or d == self.Propagate_Only_Needed \
                        or getattr(node, k) is d or d is not None and getattr(node, k) == d:
                    setattr(node, k, getattr(self, k))

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._attrib_propagation(value)

    def deletable(self, k):
        return False

    def _rename_child_for_distinction(self, node):
        super()._rename_child_for_distinction(node)
        self._attrib_propagation(node)

    def displayables(self):
        res = []
        if self.visible:
            for c in self.all_children():
                if isinstance(c, VisibleField) and c.visible:
                    res.extend(c.displayables())
        return res

    def set_pos(self, positions):
        self.positions = positions
        [self._attrib_propagation(node) for node in self.all_children()]
        return self

    def update(self, path, positions):
        if path == "" or path is None:
            self.set_pos(positions)
            return
        exist, key, sub_path, ended = self._pop_path(path)

        if ended:
            if not exist:
                self[key] = VisibleField()
            sf = self[key]
            sf.set_pos(positions)
            self._attrib_propagation()
        else:
            if not exist:
                self[key] = VisibleField()
            self[key].update(sub_path, positions)
        self._attrib_propagation()

    def ui_window(self):
        return self, self.Window_Name

    def _pop_path(self, path):
        sep = "." if "." in path else ("/" if "/" in path else "\\")
        paths = path.split(sep)
        k = paths[0]
        exist = isinstance(self[k], VisibleField)

        if not exist:
            try:
                del self[k]
            except ValueError:
                pass

        return exist, k, ".".join(paths[1:]), len(paths) == 1

    def scalar(self, path: str, values: np.ndarray, clipper=None, non_zero=False):
        exist, key, sub_path, ended = self._pop_path(path)

        if ended:
            if not exist:
                self[key] = ScalarField()
                self._attrib_propagation()
                sf = self[key]
                assert isinstance(sf, ScalarField)
                if clipper is not None:
                    sf.mapper.clip(*clipper)
                sf.non_zero_only = non_zero
            sf = self[key]
            sf.set(values)
        else:
            if not exist:
                self[key] = VisibleField()
            self[key].scalar(sub_path, values)
        self._attrib_propagation()

    def vector(self, path: str, vectors: np.ndarray, clipper=None):
        exist, key, sub_path, ended = self._pop_path(path)

        if ended:
            if not exist:
                self[key] = VectorField()
                self._attrib_propagation()
                sf = self[key]
                assert isinstance(sf, VectorField)
                if clipper is not None:
                    sf.mapper.clip(*clipper)
                else:
                    sf.mapper.normalizer = ScalarNormalizer.ZeroToAutoMax
            self[key].set(vectors)
        else:
            if not exist:
                self[key] = VisibleField()
            self[key].vector(sub_path, vectors)
            self._attrib_propagation()

    @classmethod
    def encoder(cls, x):
        if isinstance(x, np.ndarray):
            return []
        return x

    def graph(self, path, edges, colors):
        if path == "" or path is None:
            self.set(edges, colors)
            return
        exist, key, sub_path, ended = self._pop_path(path)

        if ended:
            if not exist:
                self[key] = GraphField()
                self._attrib_propagation(self[key])
            sf = self[key]
            sf.set(edges, colors)
        else:
            if not exist:
                self[key] = VisibleField()
            self[key].graph(sub_path, edges, colors)
            self._attrib_propagation()


class ScalarField(VisibleField):

    def __init__(self, label="Scalar"):
        super().__init__(label)
        self.mapper = ScalarMapper("Mapper")
        self.non_zero_only = False
        self._geo = GeoNode("geo")

    def inspect(self, label, *args, **kwargs):
        super().inspect(label)
        self.mapper.inspect("mapper")
        self.changed = self.mapper.changed
        self.changed, self.non_zero_only = imgui.checkbox("Non Zero Only", self.non_zero_only)

    def displayables(self):
        return [self._geo]

    def set(self, values):
        if isinstance(values, (int, float)):
            values = np.ones_like(self.positions[:, 0])
        if self.non_zero_only:
            non_zero_idx = np.nonzero(values)
            self._geo.as_points(self.positions[non_zero_idx], self.mapper.map_to_colors(values[non_zero_idx]))
        else:
            self._geo.as_points(self.positions, self.mapper.map_to_colors(values))
        return self


class VectorField(VisibleField):

    def __init__(self, label="Vector"):
        super().__init__(label)
        self.mapper = ScalarMapper("Mapper")
        self.max_length = 0.5
        self._geo = GeoNode("geo")

    def inspect(self, label, *args, **kwargs):
        super().inspect(label)
        self.mapper.inspect("mapper")
        self.changed = self.mapper.changed
        self.changed, self.max_length = imgui.slider_float("max length", self.max_length, 0.1, 1)

    def displayables(self):
        return [self._geo]

    def set(self, vectors):
        length = np.linalg.norm(vectors, axis=1)
        mapped_length = np.arctan(length) / np.pi * 2 * 0.01 * self.max_length
        self._geo.as_lines(self.positions, self.positions + mapped_length.reshape([-1, 1]) * vectors
                           , self.mapper.map_to_colors(length))
        return self


class GraphField(VisibleField):

    def __init__(self, label="Graph"):
        super().__init__(label)
        self.mapper = ScalarMapper("Mapper")
        self.default_color = [0., 0.15, 0.4]
        self._geo = GeoNode("geo")

    def inspect(self, label, *args, **kwargs):
        super().inspect(label)
        self.mapper.inspect("mapper")
        self.changed = self.mapper.changed
        self.changed = ImguiTool.input_color(label, self.default_color)

    def displayables(self):
        return [self._geo]

    def set(self, edges, colors=None):
        if colors is None:
            colors = np.ones([len(edges[0]), 3]) * np.array(self.default_color)
        if len(colors.shape) == 1 or colors.shape[1] == 1:
            colors = self.mapper.map_to_colors(colors)
        self._geo.as_lines(self.positions[edges[0]], self.positions[edges[1]], colors)
        return self


